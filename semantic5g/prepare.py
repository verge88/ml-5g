"""Verify frozen local artifacts and decode an existing capture, without replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from .analyze import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tshark", default="tshark")
    a = p.parse_args()
    manifest_path = Path(__file__).with_name("inputs.json")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = a.inputs / relative
        if digest(path) != expected:
            raise ValueError("Frozen artifact file digest mismatch: " + relative)
    a.output.mkdir(parents=True, exist_ok=True)
    pcap = a.inputs / "registration/native-evidence/registration.pcap"
    with (a.output / "http2-decoded.json").open("w", encoding="utf-8") as f:
        subprocess.run([a.tshark, "-r", str(pcap), "-Y", "http2", "-T", "json", "--no-duplicate-keys",
            "-J", "frame ip tcp http2 json"], stdout=f, check=True)
    version = subprocess.check_output([a.tshark, "--version"], text=True).splitlines()[0]
    (a.output / "decoder-provenance.json").write_text(json.dumps(dict(tshark=version,
        input_manifest_sha256=digest(manifest_path), files_verified=len(manifest["files"]),
        decoding="passive saved capture only; no packet transmission"), indent=2))


if __name__ == "__main__":
    main()
