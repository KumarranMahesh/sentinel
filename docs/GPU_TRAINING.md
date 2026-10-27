# Running Sentinel v3 on your Windows machine (and training the neural channel on the GPU)

Every command below is for **Windows PowerShell**, run from the folder
`D:\kumarran-folder\opportunity-assignments\sentinel`. Each step says what you should see; if you
see something else, stop at that step (the last section lists the usual causes).

Nothing here deletes or overwrites your files: step 1 adds a new git branch `v3` next to the
branch you are on (`wip/lof-reenter`), and you can switch back at any time with
`git checkout wip/lof-reenter`.

## 0. Check what the machine has (1 minute)

```powershell
nvidia-smi
py --list
git --version
```

- `nvidia-smi` prints a table. Read the GPU name and **`CUDA Version: X.Y`** in the top-right
  corner. That number decides the PyTorch build in step 4.
- `py --list` lists the installed Pythons. You need **3.11, 3.12 or 3.13** (the project was
  developed on 3.11.9). If none is listed, install Python 3.11 from python.org and tick
  "Add python.exe to PATH".
- If `nvidia-smi` is "not recognized", the NVIDIA driver is missing: install the current
  driver for your GPU from nvidia.com first. Training still works on the CPU without it (about
  20 minutes instead of a few).

## 1. Get the v3 branch

```powershell
cd D:\kumarran-folder\opportunity-assignments\sentinel\sentinel_repo
git status
git fetch ..\sentinel_repo_v3.bundle v3:v3
git checkout v3
git log --oneline -3
```

- `git status` should end with `nothing to commit, working tree clean`. If it lists modified
  files, run `git stash` first (this keeps your changes; `git stash pop` brings them back).
- `git log` should show the v3 commits at the top.

## 2. Create a virtual environment and install the requirements

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Use `py -3.12` or `py -3.13` instead of `py -3.11` if that is what `py --list` showed.
- After activation the prompt starts with `(.venv)`. Every later step assumes it is active
  (open a new PowerShell → repeat `cd`, the `Set-ExecutionPolicy` line and `Activate.ps1`).

## 3. Install the data and run the tests

```powershell
python scripts\download_nab.py --zip ..\NAB-master.zip
pytest -q
```

- The first command ends with `verified against nab_manifest.json: all data and label files match`.
- `pytest` should end with `passed` and no `failed`. The first run compiles the fast engine
  (Numba), which takes about 20-40 seconds once; later runs reuse the compiled cache.

## 4. Install PyTorch with CUDA (only needed to retrain the neural channel)

Pick the line that matches `CUDA Version` from step 0:

| `nvidia-smi` shows | command |
|---|---|
| 13.0 or higher | `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130` |
| 12.6 to 12.9 | `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126` |
| lower than 12.6 | update the NVIDIA driver, then use the first line |

Check it:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see something like `2.14.1+cu130 True NVIDIA GeForce RTX ...`. If it prints
`False`, the CPU-only build was installed: run `pip uninstall -y torch` and repeat the command
from the table.

## 5. Train the neural forecaster on the GPU

```powershell
python scripts\train_neural.py --loco --all --epochs 20 --device cuda
```

- First line: `torch 2.14.1+cu130, device cuda (NVIDIA ...)`.
- It trains 8 models (one per held-out category, plus one on everything) and prints one line
  per epoch. Ends with `wrote results/neural_training.json and 8 model(s) in models/`.
- This overwrites `models\neural_*.npz` (the copies trained on the CPU are still in git:
  `git checkout -- models` restores them). GPU training is not bit-for-bit deterministic, so
  your weights and the neural-channel numbers will differ slightly from the shipped ones.

## 6. Reproduce the v3 numbers

```powershell
python scripts\run_detector.py --detector sentinel --config configs\final.yaml --out predictions\
python scripts\evaluate.py --pred predictions\ --out results\
python scripts\benchmark_runtime.py --config configs\final.yaml
```

The validation experiments (the adoption decisions in LOGBOOK L-0048 onwards) are re-run
with the commands listed in README "Reproduce every number and figure".

## Usual problems

| Symptom | Cause / fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | run the `Set-ExecutionPolicy -Scope Process ...` line again in this window |
| `pip install -r requirements.txt` fails on `numba` | Python version outside 3.11-3.13: recreate `.venv` with `py -3.11` |
| `fatal: couldn't find remote ref v3` | the bundle is not next to `sentinel_repo` (it must be `..\sentinel_repo_v3.bundle`) |
| `torch.cuda.is_available()` is `False` | CPU wheel installed, or driver older than the CUDA build: see step 4 |
| `CUDA out of memory` | add `--batch 256` to the training command |
| Tests segfault right after you edited `src/sentinel/fast.py` | delete the Numba cache: `Remove-Item src\sentinel\__pycache__\*.nb*` |
