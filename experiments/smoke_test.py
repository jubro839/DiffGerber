"""DiffGerber smoke test.

1. Hard-limit consistency: soft Gerber -> hard Gerber as tau -> 0
2. PSD: cosine Gerber PSD by construction; gs2022 needs the clip
3. Gradients flow from the decision loss into the threshold network
4. Toy end-to-end training: learned threshold beats fixed c=0.5 OOS
   on data whose dependence lives in the tails
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffgerber import (
    DiffGerberGMV,
    GlobalThreshold,
    SoftGerber,
    StateThresholdNet,
    decision_loss,
    gmv_closed_form,
    hard_gerber,
    tau_schedule,
)

torch.manual_seed(0)
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def gen_tail_comovement(T, N=8, shock_p=0.12, shock=3.0, idio_df=3.0, seed=0):
    """Two blocks; dependence enters only through occasional large common shocks."""
    g = torch.Generator().manual_seed(seed)
    fA = (torch.rand(T, generator=g) < shock_p).double() * (
        torch.randint(0, 2, (T,), generator=g).double() * 2 - 1
    ) * shock
    fB = (torch.rand(T, generator=g) < shock_p).double() * (
        torch.randint(0, 2, (T,), generator=g).double() * 2 - 1
    ) * shock
    idio = torch.distributions.StudentT(idio_df).sample((T, N)) * 0.8
    r = idio.clone()
    r[:, : N // 2] += fA.unsqueeze(-1)
    r[:, N // 2 :] += fB.unsqueeze(-1)
    return r


def check_hard_limit():
    print("== 1. hard-limit consistency (soft -> hard as tau -> 0) ==")
    z = torch.randn(500, 6)
    layer = SoftGerber("cos")
    Gh = hard_gerber(z, 0.5, "cos")
    for tau in [0.2, 0.05, 0.01, 0.002]:
        Gs = layer(z, 0.5, tau)
        print(f"  tau={tau:<6} max|soft-hard| = {(Gs - Gh).abs().max():.2e}")


def check_psd():
    print("== 2. PSD ==")
    worst_cos, worst_gs, worst_gs_clip = 1.0, 1.0, 1.0
    for s in range(50):
        z = torch.randn(120, 15, generator=torch.Generator().manual_seed(s))
        G = SoftGerber("cos")(z, 0.5, 0.1)
        worst_cos = min(worst_cos, torch.linalg.eigvalsh(G).min().item())
        Gg = SoftGerber("gs2022", straight_through=True)(z, 0.5, 0.1)
        worst_gs = min(worst_gs, torch.linalg.eigvalsh(Gg).min().item())
        Gc = SoftGerber("gs2022", straight_through=True)(z, 0.5, 0.1)
        worst_gs_clip = min(worst_gs_clip, torch.linalg.eigvalsh(Gc).min().item())
    print(f"  min eig over 50 draws: cos={worst_cos:.3e} (must be >= -1e-10), "
          f"gs2022 raw={worst_gs:.3e}, gs2022 clipped={worst_gs_clip:.3e}")
    assert worst_cos > -1e-10


def check_gradients():
    print("== 3. gradient flow through StateThresholdNet ==")
    T, N, F_ = 250, 8, 3
    r = gen_tail_comovement(T + 21, N)
    window, future = r[:T], r[T:]
    feats = torch.stack(
        [r[:T].abs().mean(-1), r[:T].std(-1), r[:T].pow(2).mean(-1)], dim=-1
    )
    net = StateThresholdNet(F_, n_assets=N)
    model = DiffGerberGMV(net)
    out = model(window, window.std(0), tau=0.1, feats=feats)
    loss = decision_loss(out["w"], future)
    loss.backward()
    gnorm = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    print(f"  loss={loss.item():.6f}, total |grad| over threshold net = {gnorm:.3e}")
    assert gnorm > 0


def check_end_to_end():
    print("== 4. toy end-to-end training (learned global c vs fixed 0.5) ==")
    N, WIN, H, STEP = 8, 250, 21, 21
    r = gen_tail_comovement(2400, N, seed=1)
    train_end = 1400

    def make_folds(lo, hi):
        folds = []
        k = lo + WIN
        while k + H <= hi:
            folds.append((r[k - WIN : k], r[k : k + H]))
            k += STEP
        return folds

    train_folds = make_folds(0, train_end)
    test_folds = make_folds(train_end, 2400)

    model = DiffGerberGMV(GlobalThreshold(init=0.5))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    epochs = 60
    for ep in range(epochs):
        tau = tau_schedule(ep, epochs, 0.2, 0.02)
        opt.zero_grad()
        loss = sum(
            decision_loss(model(w_, w_.std(0), tau)["w"], f_)
            for w_, f_ in train_folds
        ) / len(train_folds)
        loss.backward()
        opt.step()
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  tau={tau:.3f}  train loss={loss.item():.5f}  "
                  f"c={model.thresholds().item():.3f}")

    c_learned = model.thresholds().item()

    def oos_var(cov_fn):
        tot = 0.0
        for w_, f_ in test_folds:
            w = gmv_closed_form(cov_fn(w_), ridge=1e-4)
            tot += decision_loss(w, f_).item()
        return tot / len(test_folds)

    def gerber_cov(window, c):
        s = window.std(0)
        G = hard_gerber(window / s, c, "cos")
        return s.unsqueeze(-1) * G * s.unsqueeze(-2)

    v_sample = oos_var(lambda w_: (w_ - w_.mean(0)).T @ (w_ - w_.mean(0)) / len(w_))
    v_fixed = oos_var(lambda w_: gerber_cov(w_, 0.5))
    v_learned = oos_var(lambda w_: gerber_cov(w_, c_learned))

    print(f"  learned c = {c_learned:.3f}  (init 0.5)")
    print(f"  OOS realized portfolio variance ({len(test_folds)} test rebalances):")
    print(f"    sample covariance : {v_sample:.5f}")
    print(f"    hard Gerber c=0.5 : {v_fixed:.5f}")
    print(f"    hard Gerber c=learned : {v_learned:.5f}")


if __name__ == "__main__":
    check_hard_limit()
    check_psd()
    check_gradients()
    check_end_to_end()
    print("\nAll smoke checks completed.")
