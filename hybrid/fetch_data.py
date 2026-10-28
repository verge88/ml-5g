"""Download pinned public captures without executing upstream code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

PROTOCOL = Path(__file__).with_name("protocol.json")


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(destination):
    spec = json.loads(PROTOCOL.read_text())["external_dataset"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expected in spec["files"].items():
        path = destination / name
        url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['ref']}/datasets/{name}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if git_blob_sha(data) != expected:
                raise ValueError(f"Upstream Git blob mismatch: {name}")
            path.write_bytes(data)
        data = path.read_bytes()
        assert git_blob_sha(data) == expected, name
        rows.append({"name": name, "url": url, "bytes": len(data), "git_blob_sha": expected,
                     "sha256": hashlib.sha256(data).hexdigest()})
        print(json.dumps(rows[-1]), flush=True)
    (destination / "manifest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/granomaly")
    fetch(parser.parse_args().output)

