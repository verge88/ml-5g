# Causal neural 5G request detectors

Four independent PyTorch implementations analyze recorded SBI requests:

- **TGN:** persistent per-node memory, raw endpoint messages, chronological last-message aggregation, differentiable GRU memory updates, learned elapsed-time encodings, and attention over recent temporal neighbors. Pending messages from batch k update memory when scoring k+1. Labels never enter online state. This is a mini-batch TGN event classifier, with a documented update lag of 128 events, rather than an exact reproduction of the original link-prediction experiment.
- **GNN:** two mean-aggregation GraphSAGE layers over the graph observed in preceding minibatches. Node features summarize preceding incident edge features. The classifier sees directed current endpoints and current features; the historical neighborhood is undirected.
- **LSTM:** a unidirectional LSTM over the latest 16 requests, ending at the current request.
- **Transformer:** two independently initialized encoder layers, four attention heads, position embeddings, causal masks, and explicit right-padding masks. The readout uses the last valid token.

HGB is refitted on the same current-request features and training requests as an additional baseline. The earlier frozen HGB/MLP packet-trained results remain separate experiments.

## Experiment contract

`protocol.json` fixes 8 epochs, hidden width 48, batch size 128, three seeds, two source/target capture folds, and two feature sets before the Actions experiment. No hyperparameters or thresholds are selected using target labels. This remains a retrospective study on previously inspected captures, not a blind preregistered discovery.

Training uses the first 60% of the sorted normal packets and one whole source attack capture. Calibration uses only the next normal partition after a 32-packet embargo. The remaining normal partition and the other attack capture are held out. Both directions NRF→SMF and SMF→NRF are evaluated. Repeated normal frame numbers do not define training membership: original CSV row identity does.

The **metadata** inputs are the prior 16 numeric packet/context features plus endpoint one-hot identities. The **metadata_api** ablation adds seven HTTP-method indicators, 16 deterministic service-hash bins, and 64 deterministic normalized-resource-hash bins. Hashes do not fit a held-out vocabulary. Subscriber IDs, UUIDs and numeric resource IDs are normalized upstream. Query arguments are excluded. Hash collisions are a limitation of this bounded representation.

All architectures receive the same event information. Graph models additionally organize past events by their endpoints. Identity vocabularies use training-normal and source training captures only; unseen target nodes map to a frozen unknown bucket. Numeric medians/scales and structural reference counts fit training-normal packets only. Absolute epochs, response codes, labels, source file names, frame numbers, tokens and bodies are not input features.

Class weighting is calculated from original normal/non-normal source training labels. Original portscan interval labels are retained for training; label-only and semantic-specific subsets are separated in evaluation. Whole training streams may change order between epochs, but event order inside each stream is preserved. State resets between epochs, captures and normal partitions. Calibration and test inference update graph observations causally with frozen weights; they do not train on test labels.

The threshold is the higher 99th percentile of normal calibration scores, with strict `score > threshold`. Every saved threshold and every evaluation score is independently audited against source rows in the aggregate job. All request-label, semantic-label, producer-candidate and known-edge candidate scopes are exported. The producer-candidate denominator follows `semantic5g`, including its retrospective selection and label-quality limitations.

Three seeds measure initialization variability only. There are still only two attack capture sessions. No claim of improved population-level detection, authorization verification or CU verification follows from successful CI execution.

## Reproduction and outputs

Install the existing `requirements.txt`, then CPU PyTorch:

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests/test_neural5g.py tests/test_semantic5g.py
python -m hybrid.fetch_data --output data/granomaly
python -m neural5g.benchmark --data data/granomaly --model tgn --output results/neural/tgn
```

Replace `tgn` with `gnn`, `lstm`, `transformer` or `hgb`. The workflow `.github/workflows/neural-models.yml` runs all 60 fits and the independent aggregate audit.

Each fit exports learned weights (PyTorch `state_dict` or HGB joblib), preprocessing statistics, node vocabulary, exact configuration, loss history, calibrated threshold, class weighting, prediction arrays and timing. TGN additionally records nonzero gradient evidence for memory/message/attention/time modules. Capture-specific memory is reset rather than exported as learned state. PyTorch weights should be loaded with `torch.load(..., weights_only=True)` into the matching class. Load only trusted HGB joblib files.

Architecture and implementation references:

- [Rossi et al., Temporal Graph Networks](https://arxiv.org/abs/2006.10637), [authors' implementation](https://github.com/twitter-research/tgn).
- [Hamilton et al., GraphSAGE](https://arxiv.org/abs/1706.02216).
- [PyTorch TransformerEncoder documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.TransformerEncoder.html).
- [PyTorch CPU version installation](https://pytorch.org/get-started/previous-versions/).
- [Pinned Granomaly source](https://github.com/TF0x42/granomaly/blob/51df94b4e88d1c9c5e499416e4b8fee5c9b93cd7/README.md).
