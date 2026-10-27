# Deploy the Sentinel demo on Streamlit Community Cloud

Free hosting; you need a GitHub account. Commands are Windows PowerShell from
`D:\kumarran-folder\opportunity-assignments\sentinel\sentinel_repo` (after
`docs/GPU_TRAINING.md` step 1, so branch `v3` exists locally).

## 1. Put the repo on GitHub

1. On github.com: top-right **+** → **New repository** → name `sentinel` → **Public** →
   do **not** add a README/.gitignore/license → **Create repository**.
2. In PowerShell (replace `YOUR-USER`):

```powershell
git checkout v3
git remote add github https://github.com/YOUR-USER/sentinel.git
git push -u github v3
```

`git push` opens a browser window to sign in to GitHub the first time. Check on github.com that
branch `v3` shows `app/streamlit_app.py` and `requirements.txt`.

## 2. Create the app

1. Open https://share.streamlit.io → **Continue with GitHub** → authorize Streamlit.
2. Click **Create app** (top right) → **Yup, I have an app**.
3. Fill in:
   - **Repository:** `YOUR-USER/sentinel`
   - **Branch:** `v3`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** optional, e.g. `sentinel-utkrishta` → `https://sentinel-utkrishta.streamlit.app`
4. **Advanced settings** → **Python version:** `3.12` → **Save**. No secrets are needed.
5. Click **Deploy**. The build installs `requirements.txt` (a few minutes).

## 3. First start

The data (`data/raw/`) is not in git. On its first start the app downloads the pinned NAB
commit from GitHub (about 50 MB) and checks every file against `configs/nab_manifest.json`;
a spinner shows "First start: downloading the NAB data". The first stream also compiles the
fast engine once.

## If it does not work

| Symptom | Fix |
|---|---|
| Build log: error installing a package | Advanced settings → Python version must be 3.11, 3.12 or 3.13 (Manage app → Settings to change) |
| App shows no NAB files after the first start | The download was blocked. Upload CSV mode still works. Or commit the data: `git add -f data/raw/nab/data data/raw/nab/labels` → commit → `git push github v3` (NAB is AGPL-3.0: keep `data/raw/nab/LICENSE.txt` with it) |
| "You do not have access" when picking the repo | On share.streamlit.io: avatar → Settings → Linked accounts → GitHub → grant access to the repo |
| App sleeps after a few days without visitors | Open the URL and click "Yes, get this app back up" |
