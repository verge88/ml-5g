import json
from pathlib import Path
import numpy as np
from hybrid.diagnose import altered_training
from hybrid.fetch_data import git_blob_sha
from hybrid.simulation import split


def test_original_experiment_sources_are_byte_identical():
    root = Path(__file__).resolve().parents[1] / "hybrid"
    originals = {"benchmark.py": "cdc4dbcb42d9a0fadc6b2766cbade8f663324d66",
                 "simulation.py": "fe452ad035775ff918ffb037b112bdc6ef14002f",
                 "external.py": "c4403ea7b92fee0e1424e5bb6c966115265cef57",
                 "protocol.json": "aebce5a24333d98d38dce8b42f42282a826825f1"}
    for name, expected in originals.items():
        assert git_blob_sha((root / name).read_bytes()) == expected


def test_explanatory_interventions_preserve_positive_count_and_untouched_data():
    runs = split(2, "attack")
    original_x = [r.x.copy() for r in runs]
    exposed, cleaned, changes = altered_training(2, runs)
    assert len(changes) == len(runs)
    for i, (old, new) in enumerate(zip(runs, exposed)):
        assert np.array_equal(old.x, original_x[i])
        assert np.array_equal(old.y, new.y)
        assert np.array_equal(old.context, new.context)
        integrity_points = sum(e["end"] - e["start"] for e in old.events if e["kind"] == "integrity")
        assert int(old.y.sum()) - int(cleaned[i].sum()) == integrity_points
        assert sum(e["kind"] == "sequence" for e in old.events) == 0
        assert sum(e["kind"] == "sequence" for e in new.events) == 1

