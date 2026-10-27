"""Train the neural forecaster (D9) with PyTorch; uses the GPU when one is available.

  python scripts/train_neural.py --all                  # one model on all 7 categories
  python scripts/train_neural.py --loco                 # 7 models, each without one category
  python scripts/train_neural.py --loco --all --epochs 30 --device cuda

Self-supervised: every model learns to predict each value from the previous `window` values
of the training series. NAB labels are never read. For honest evaluation on NAB, a file is
scored by the model that did not see its category (models/neural_wo_<category>.npz); new
files use models/neural_all.npz. Weights are saved as .npz for NumPy inference
(src/sentinel/detectors/neural.py), so the detector itself does not need PyTorch.
Writes models/*.npz and results/neural_training.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from sentinel.config import ROOT  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.neural import batch_features  # noqa: E402

MODELS = ROOT / "models"


def build(corpus, names, window):
    X, Y = [], []
    for n in names:
        s = corpus[n]
        x, y = batch_features(s.values, s.seconds, window)
        X.append(x)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def train_one(X, Y, args, device, tag):
    import torch
    from torch import nn
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    n_val = max(int(0.05 * len(X)), 1)
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)
    d_in = X.shape[1]
    model = nn.Sequential(nn.Linear(d_in, args.hidden), nn.GELU(approximate="tanh"),
                          nn.Linear(args.hidden, args.hidden // 2), nn.GELU(approximate="tanh"),
                          nn.Linear(args.hidden // 2, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def nll(xb, yb):
        out = model(xb)
        mu, log_b = out[:, 0], out[:, 1].clamp(-6, 6)
        return (log_b + (yb - mu).abs() / log_b.exp()).mean()

    tr_t = torch.tensor(tr, device=device)
    va_t = torch.tensor(va, device=device)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        tot = 0.0
        for k in range(0, len(perm), args.batch):
            b = perm[k:k + args.batch]
            loss = nll(Xt[b], Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(nll(Xt[va_t], Yt[va_t]))
        hist.append({"epoch": ep + 1, "train_nll": tot / len(tr), "val_nll": vl})
        print(f"  [{tag}] epoch {ep + 1:2d}/{args.epochs}  train {tot / len(tr):.4f}  val {vl:.4f}",
              flush=True)
    lin = [m for m in model if isinstance(m, nn.Linear)]
    arrays = {"window": np.array(args.window)}
    for i, m in enumerate(lin):
        arrays[f"W{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        arrays[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"neural_{tag}.npz"
    np.savez(path, **arrays)
    return {"tag": tag, "samples": int(len(X)), "seconds": round(time.time() - t0, 1),
            "final": hist[-1], "history": hist, "path": str(path.relative_to(ROOT))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="train one model on every category")
    ap.add_argument("--loco", action="store_true", help="train one model per held-out category")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args()
    if not (args.all or args.loco):
        ap.error("pass --all and/or --loco")
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"torch {torch.__version__}, device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)
    corpus = load_corpus()
    cats = sorted({s.category for s in corpus.values()})
    jobs = []
    if args.loco:
        jobs += [(f"wo_{c}", [n for n, s in corpus.items() if s.category != c]) for c in cats]
    if args.all:
        jobs.append(("all", list(corpus)))
    out = {"args": vars(args), "device": device, "torch": torch.__version__, "models": []}
    for tag, names in jobs:
        X, Y = build(corpus, names, args.window)
        print(f"{tag}: {len(names)} files, {len(X):,} samples", flush=True)
        out["models"].append(train_one(X, Y, args, device, tag))
    p = ROOT / "results" / "neural_training.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p.relative_to(ROOT)} and {len(jobs)} model(s) in models/")


if __name__ == "__main__":
    main()
