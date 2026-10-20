"""
CALF (Causally-Aware Layered Fabric) adapted for REAL observational data.

Changes vs. the paper's synthetic setup (each one is deliberate):
  1. Binary outcome (default): BCE on logits; tau = p1 - p0 (risk difference).
  2. No ground-truth ITE exists -> checkpoint selection uses MSE against a
     cross-fitted doubly-robust pseudo-outcome (see dr_eval.py), not PEHE.
  3. Monotonicity prior beta defaults to 0. A baked-in "effects are non-negative"
     prior would make the downstream counterfactual audit partly circular.
  4. Variable number of layers (tiers 0..tier(T)), since a treatment can only be
     adjusted for by covariates upstream of it.
  5. `k_tokens`: the paper's Eq. 3-5 apply MHA to ONE vector per layer. With a
     single key, softmax is identically 1, so MHA collapses to out_proj(v_proj(upstream))
     -- a linear map, not selective attention. With k_tokens>1 each layer is split
     into K latent tokens so attention can actually select among upstream slots.
     k_tokens=1 reproduces the paper-as-written (use it as an ablation).
  6. Large batches / fewer epochs (n is ~1e5-1e6, not 9k).
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Eq. 2: ELU(LN(W2 Dropout(ELU(LN(W1 x))))) + linear skip."""
    def __init__(self, d_in, h, p):
        super().__init__()
        self.l1, self.n1 = nn.Linear(d_in, h), nn.LayerNorm(h)
        self.l2, self.n2 = nn.Linear(h, h), nn.LayerNorm(h)
        self.drop = nn.Dropout(p)
        self.skip = nn.Linear(d_in, h, bias=False)

    def forward(self, x):
        z = self.drop(F.elu(self.n1(self.l1(x))))
        return F.elu(self.n2(self.l2(z))) + self.skip(x)


def head(h, p):
    return nn.Sequential(nn.Linear(h, h), nn.ELU(), nn.Dropout(p),
                         nn.Linear(h, h // 2), nn.ELU(), nn.Dropout(p),
                         nn.Linear(h // 2, 1))


class CALF(nn.Module):
    def __init__(self, dims, h=128, k_tokens=4, heads=4, p=0.3, use_attn=True):
        super().__init__()
        assert h % k_tokens == 0 and (h // k_tokens) % heads == 0
        self.K = k_tokens
        d = h // k_tokens
        self.enc = nn.ModuleList([Block(di, h, p) for di in dims])
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, batch_first=True) for _ in dims[1:]]
        ) if use_attn else None
        self.fuse = Block(h * len(dims), h, p)
        self.e_head, self.y0_head, self.y1_head = head(h, p), head(h, p), head(h, p)

    def forward(self, xs):
        z = [enc(x).view(x.size(0), self.K, -1) for enc, x in zip(self.enc, xs)]
        if self.attn is not None:
            for i, mha in enumerate(self.attn):          # Eq. 3-5: upstream is already updated
                o, _ = mha(z[i + 1], z[i], z[i], need_weights=False)
                z[i + 1] = z[i + 1] + o
        f = self.fuse(torch.cat([t.flatten(1) for t in z], 1))
        return self.e_head(f).squeeze(-1), self.y0_head(f).squeeze(-1), self.y1_head(f).squeeze(-1)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, layers, bs=8192, mc=0):
    """Returns dict(tau, p0, p1, e). mc>0 -> MC-dropout mean/std (UNCALIBRATED; see paper Sec. VI-F)."""
    dev = next(model.parameters()).device
    n = len(layers[0])

    def run():
        out = []
        for i in range(0, n, bs):
            xb = [torch.from_numpy(x[i:i + bs]).to(dev) for x in layers]
            e, l0, l1 = model(xb)
            out.append(torch.stack([torch.sigmoid(e), torch.sigmoid(l0), torch.sigmoid(l1)], 1).cpu())
        return torch.cat(out).numpy()

    if mc <= 0:
        model.eval()
        a = run()
        return dict(e=a[:, 0], p0=a[:, 1], p1=a[:, 2], tau=a[:, 2] - a[:, 1])
    model.train()                                           # dropout on
    draws = np.stack([run() for _ in range(mc)])
    model.eval()
    tau = draws[:, :, 2] - draws[:, :, 1]
    return dict(e=draws[:, :, 0].mean(0), p0=draws[:, :, 1].mean(0), p1=draws[:, :, 2].mean(0),
                tau=tau.mean(0), tau_sd=tau.std(0))


def fit_calf(layers_tr, T_tr, y_tr, layers_va, tau_dr_va, *, alpha=1.0, beta=0.0, h=128,
             k_tokens=4, heads=4, p=0.3, use_attn=True, epochs=40, bs=2048, lr=1e-3,
             wd=1e-4, patience=8, seed=0, verbose=True):
    """Train one CALF. Early-stop / checkpoint on validation MSE(tau_hat, tau_dr)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = _dev()
    model = CALF([x.shape[1] for x in layers_tr], h, k_tokens, heads, p, use_attn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = [torch.from_numpy(x).to(dev) for x in layers_tr]
    T = torch.from_numpy(np.asarray(T_tr, "float32")).to(dev)
    Y = torch.from_numpy(np.asarray(y_tr, "float32")).to(dev)
    n = len(T)
    best, best_state, bad = np.inf, None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            e, l0, l1 = model([x[idx] for x in X])
            tb, yb = T[idx], Y[idx]
            lt = torch.where(tb > 0.5, l1, l0)
            loss = F.binary_cross_entropy_with_logits(lt, yb) \
                 + alpha * F.binary_cross_entropy_with_logits(e, tb)
            if beta > 0:
                loss = loss + beta * F.relu(-(torch.sigmoid(l1) - torch.sigmoid(l0))).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        va = float(np.mean((predict(model, layers_va)["tau"] - tau_dr_va) ** 2))
        if verbose:
            print(f"ep {ep:02d}  train_loss {tot / n:.4f}  val_DR_MSE {va:.5f}")
        if va < best - 1e-7:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def fit_ensemble(*args, seeds=(0, 1, 2), **kw):
    return [fit_calf(*args, seed=s, **kw) for s in seeds]


def predict_ensemble(models, layers):
    """Mean tau and between-seed SD. Cheap, and usually better behaved than MC-dropout."""
    taus = np.stack([predict(m, layers)["tau"] for m in models])
    return taus.mean(0), taus.std(0)


# ----------------------------------------------------------------------------- persistence
def save_ensemble(models, path_dict, key, meta):
    """Collect state_dicts + constructor args into path_dict[key] (then torch.save the dict once)."""
    path_dict[key] = dict(meta=meta, states=[{k: v.cpu() for k, v in m.state_dict().items()} for m in models])


def load_ensemble(entry, device=None):
    dev = device or _dev()
    m = entry["meta"]
    out = []
    for st in entry["states"]:
        net = CALF(m["dims"], m["h"], m["k_tokens"], m["heads"], m["p"], m["use_attn"]).to(dev)
        net.load_state_dict(st); net.eval(); out.append(net)
    return out


def ensemble_outputs(models, layers):
    """Mean over seeds of e, p0, p1, tau (+ between-seed SD of tau)."""
    P = [predict(m, layers) for m in models]
    g = lambda k: np.stack([p[k] for p in P])
    return dict(e=g("e").mean(0), p0=g("p0").mean(0), p1=g("p1").mean(0),
                tau=g("tau").mean(0), tau_sd=g("tau").std(0))
