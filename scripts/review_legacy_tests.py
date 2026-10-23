"""Inventory standalone regressions without contacting any network service.

Run: python scripts/review_legacy_tests.py
Known legacy failures are reported honestly; exit 1 means a deterministic
script failed. Live tests are classified as blocked, never as passing.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = '''import os, runpy, socket, sys
os.environ.update(MODEL_PROVIDER="cloudflare", CLOUDFLARE_API_TOKEN="offline-placeholder", CLOUDFLARE_ACCOUNT_ID="offline-placeholder", GROQ_API_KEY="offline-placeholder", LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")
for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(name, None)
def blocked(*args, **kwargs):
    raise RuntimeError("NETWORK_BLOCKED_FOR_OFFLINE_REVIEW")
socket.socket.connect = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
import requests, httpx
requests.sessions.Session.request = blocked
httpx.Client.send = blocked
async def async_blocked(*args, **kwargs):
    return blocked()
httpx.AsyncClient.send = async_blocked
sys.path.insert(0, os.getcwd())
runpy.run_path(sys.argv[1], run_name="__main__")
'''


def run(path):
    try:
        process = subprocess.run([sys.executable, "-c", WRAPPER, str(path)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=35)
        output = process.stdout + process.stderr
        status = "PASS" if process.returncode == 0 else (
            "LIVE_BLOCKED" if "NETWORK_BLOCKED_FOR_OFFLINE_REVIEW" in output else "FAIL")
        return {"file": path.relative_to(ROOT).as_posix(), "status": status,
                "exit_code": process.returncode, "detail": output.splitlines()[-1] if output else ""}
    except subprocess.TimeoutExpired:
        return {"file": path.relative_to(ROOT).as_posix(), "status": "TIMEOUT", "exit_code": 124, "detail": "35-second timeout"}


def main():
    files = sorted(ROOT.glob("workflows/*_test.py"))
    files += sorted(p for p in (ROOT / "agents").rglob("*.py") if "test" in p.name.lower())
    files += [ROOT / name for name in ("rate_limit_test.py", "cloudflare_connection_test.py", "cloudflare_structured_test.py")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, files))
    destination = ROOT / "runtime" / "diagnostics" / "legacy_offline_review.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(result["status"], result["file"])
    print("Summary:", {status: sum(item["status"] == status for item in results)
                       for status in ("PASS", "FAIL", "LIVE_BLOCKED", "TIMEOUT")})
    print("Report:", destination)
    return int(any(result["status"] in ("FAIL", "TIMEOUT") for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
