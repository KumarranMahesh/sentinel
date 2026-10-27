"""Save a screenshot of the running Streamlit demo for the slides (Windows/Linux, Edge or Chrome).

  streamlit run app/streamlit_app.py            # in another terminal
  python scripts/screenshot_demo.py --file realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv

Starts a headless browser with a temporary profile, opens the demo deep link
(?file=...&run=1), waits until the run has rendered, and saves figures/demo_streamlit.png.
"""
import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "google-chrome", "chromium", "microsoft-edge"]


def find_browser():
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    sys.exit("no Edge/Chrome found")


async def capture(ws_url, url, width, height, wait_s, out):
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        await call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                   deviceScaleFactor=1.5, mobile=False)
        await call("Page.enable")
        await call("Page.navigate", url=url)
        await asyncio.sleep(wait_s)
        res = await call("Page.captureScreenshot", format="png")
        Path(out).write_bytes(base64.b64decode(res["data"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--out", default=str(ROOT / "figures" / "demo_streamlit.png"))
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1250)
    ap.add_argument("--wait", type=float, default=15.0)
    args = ap.parse_args()
    profile = tempfile.mkdtemp(prefix="sentinel-shot-")
    proc = subprocess.Popen([find_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--user-data-dir={profile}", "--remote-debugging-port=9333", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen("http://localhost:9333/json").read())
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        else:
            sys.exit("browser did not start")
        url = f"http://localhost:{args.port}/?file={args.file}&run=1"
        asyncio.run(capture(ws_url, url, args.width, args.height, args.wait, args.out))
        print("wrote", args.out)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
