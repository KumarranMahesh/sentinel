"""Save a screenshot of the running Streamlit demo for the slides.

  streamlit run app/streamlit_app.py            # in another terminal
  python scripts/screenshot_demo.py --file realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv

Optional tool, not needed to run Sentinel. Needs Playwright:
  pip install playwright && python -m playwright install chromium
Opens the demo deep link (?file=...&run=1) in headless Chromium, waits until the run has
rendered (the evidence section appears), and saves figures/demo_streamlit.png.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv")
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default=str(ROOT / "figures" / "demo_streamlit.png"))
    args = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Playwright is not installed: pip install playwright && python -m playwright install chromium")
    url = f"{args.url}/?file={args.file}&run=1"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(url)
        page.get_by_text("Evidence for each alert").wait_for(timeout=args.timeout * 1000)
        page.wait_for_timeout(2500)                  # let plotly finish drawing
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=args.out, full_page=False)
        browser.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
