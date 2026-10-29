# Passive semantic analysis of recorded 5G messages

This experiment analyzes recorded traffic. It never executes the upstream attack scripts, transmits exploit messages, or replays traffic into a core. It is retrospective and exploratory, rather than a new preregistered replication.

## Inputs and units

Granomaly revision `51df94b4e88d1c9c5e499416e4b8fee5c9b93cd7` provides real Open5GS/UE traffic annotated by the dataset authors. Its published CSVs contain HTTP/2 method/path/status summaries, but no JSON bodies, tokens, authenticated caller identities, or trusted CU evidence. All three CSVs retain their original blob and SHA256 validation.

The separate official Open5GS v2.7.6 registration-suite capture and the frozen ML scores come from run `40902826197`, commit `681f8d234aa0cc0e9b88c7ed55866dadf4f8e92d`. Thirteen extracted files are pinned by SHA256 in `inputs.json`. Registration-suite negative cases are standard functional tests, not malicious attack evidence.

The analysis unit is an observed request hop, not a unique UE, independent attack episode, or arbitrary packet. Original packet-level ML rates must not be compared directly with these request-level rates. `compare.py` joins frozen scores to precisely the source packet of each request, validates labels and frame uniqueness, and retains the original thresholds. ML is neither refitted nor tuned.

## Detectors

All operation banks use the same first 60% of the normal capture's packets as the frozen ML training. The rules never read attack labels while producing alerts. Subscriber IDs, UUIDs, numeric resource identifiers, opaque hex IDs, and the Wireshark frame delimiter are normalized. Query arguments and payload semantics remain outside this CSV policy's observability.

1. Edge novelty: unfamiliar `(source IP, destination IP)`.
2. Edge-operation novelty: unfamiliar `(source IP, destination IP, HTTP method, API resource template)`.
3. Caller-operation novelty: unfamiliar `(source IP, HTTP method, resource template)`.

These are observational normality policies. A source address can aggregate different NF roles. The repository's host inventory is not treated as authoritative capture-time caller identity. SCP forwarded requests receive no inferred authenticated original caller. Authorization outcome and CU verification remain `UNKNOWN`.

## Evaluation scopes

Both denominators are preserved:

- Every request carrying any of the four attack labels, excluding the nonsemantic portscan class. This includes ordinary interleaved procedures in labeled intervals.
- Producer candidates: the request's API operation agrees with its source label and its observed destination is the relevant service producer. Fixed endpoints are NRF `.7`, UDM `.8`, and AMF `.4`, corroborated by ordinary requests. This excludes an interleaved subscription request to SCP under the subscription-attack label. This is a transparent retrospective subset, not an independently reannotated ground truth.

The NRF recording has 58 candidates. Edge and operation policies both flag 58; each also flags three requests labeled normal among 4169. The SMF recording has 71 candidates. Edge policy flags 51, operation policy 71, with zero alerts among 2509 requests labeled normal. At the broader label-only denominator, operation policy flags 71 of 481 SMF requests, not 100%. The missing 410 include interleaved procedures; the experiment cannot independently resolve every label.

The 20 additional SMF alerts are deregistration notifications over the already observed SCP-to-AMF edge, whereas ordinary callbacks have different methods/resources. Their recorded responses are 17 HTTP 204 and three HTTP 404. These are not proofs of UE disconnection, authenticated source identity, or successful authorization bypass.

In the SMF candidate subset, frozen HGB and MLP each flag 51/71. Operation policy flags 71/71. HGB OR operation policy flags 71/71 but retains one normal false alert, compared with zero for operation policy alone. In the NRF subset, HGB and operation policy each flag 58/58; HGB has 92 normal false alerts versus three for operation policy. These two captures do not demonstrate a benefit from adding these ML models to the operation policy.

## Native JSON check

Saved PCAP decoding groups messages by `(tcp.stream, http2.streamid, direction)`, reassembles DATA, and inspects declared JSON content. Full bodies, keys, subscriber IDs, and raw decoded traffic are not included in report artifacts. It checks NFProfile instance ID binding to its registration URI and reads the actual authentication result. This checks observed consistency, not truth of NF state or full 3GPP conformance.

TS 29.509 permits HTTP 200 with `authResult=AUTHENTICATION_FAILURE`; therefore HTTP status alone is insufficient. The current real registration capture contains 96 observed successful auth-result hops and no recorded authResult failure. A parser unit fixture exercises this standard-defined negative result separately; it is never included in empirical attack counts.

## Reproduction

The workflow `.github/workflows/passive-semantic.yml` downloads only pinned recordings/frozen scores, runs six parser/correlation controls, decodes the saved PCAP, evaluates policies, and uploads redacted request tables and summaries. It requires neither a running 5G core nor attack code.

Sources:

- [Granomaly source and dataset description](https://github.com/TF0x42/granomaly/blob/51df94b4e88d1c9c5e499416e4b8fee5c9b93cd7/README.md)
- [ETSI TS 129 509 v17.7.0, section 5.2.2.2.2](https://www.etsi.org/deliver/etsi_ts/129500_129599/129509/17.07.00_60/ts_129509v170700p.pdf)
- [Frozen experiment run](https://github.com/verge88/ml-5g/actions/runs/40902826197)
