"""Fetch the official SAM2 Python source at one recorded GitHub commit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

DEST = Path.home() / "src" / "sam2-runtime"
API = "https://api.github.com/repos/facebookresearch/sam2"

def fetch(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Earthquake-SAM-environment-setup"})
            with urllib.request.urlopen(req, timeout=40) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

commit = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
tree = json.loads(fetch(API + f"/git/trees/{commit}?recursive=1"))
assert not tree.get("truncated"), "Incomplete source tree"
root_files = {"setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "LICENSE_cctorch"}
entries = [e for e in tree["tree"] if e["type"] == "blob" and
           (e["path"].startswith("sam2/") or e["path"] in root_files)]
DEST.mkdir(parents=True, exist_ok=True)

def download(entry):
    path = DEST / entry["path"]
    data = fetch(f"https://raw.githubusercontent.com/facebookresearch/sam2/{commit}/{entry['path']}")
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    assert digest == entry["sha"], f"Source checksum mismatch: {entry['path']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

print(f"Official SAM2 commit: {commit}; fetching {len(entries)} runtime files", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(download, entries))
(DEST / "SOURCE_COMMIT.txt").write_text(commit + "\n")
print(f"Verified source saved to {DEST}", flush=True)
