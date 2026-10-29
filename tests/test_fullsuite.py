import hashlib
import json
from pathlib import Path
import numpy as np
from fullsuite.common import P,balanced_indices,cutoff,temporal


def test_original_sources_unchanged():
    root=Path(__file__).resolve().parents[1]
    blobs=json.loads((root/"fullsuite/source-blobs.json").read_text())
    for path,expected in blobs.items():
        data=(root/path).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==expected,path


def test_fresh_seeds_are_disjoint_from_previous_series_and_development():
    assert not (set(P["basic_seeds"])|set(P["complex_seeds"])) & (set(range(41,51))|set(range(101,111))|{1,7,9,11})
    assert len(set(P["basic_seeds"]))==10 and len(set(P["complex_seeds"]))==10


def test_balancing_uses_only_given_training_labels():
    y=np.r_[np.zeros(100,dtype=bool),np.ones(7,dtype=bool)]
    i=balanced_indices(y,901)
    assert y[i].sum()==7 and len(i)==42 and len(set(i))==42


def test_temporal_statistics_do_not_see_future():
    signal=np.random.default_rng(900).normal(size=(100,3))
    modified=signal.copy();modified[40:]+=1000
    for mode in ("ewma","cusum"):
        assert np.array_equal(temporal(signal,mode)[:40],temporal(modified,mode)[:40])


def test_tied_normal_calibration_does_not_create_alarms():
    assert cutoff(np.zeros(100))==0
    assert not (np.zeros(100)>cutoff(np.zeros(100))).any()


def test_unimplemented_architectures_are_explicitly_excluded():
    assert "MLP" not in P["not_implemented"]
    assert {"TGN","Transformer","LSTM","GNN"}==set(P["not_implemented"])
