# Native CU transport with separately signed runtime witnesses

Build fixed official Open5GS 2.7.6 with two benign instrumentation patches: export process PID/native runtime UUID and transport NF-side CU metadata using NFProfile.customInfo during registration and heartbeat. No attack or malicious response hook is compiled. Run native NRF, AUSF and NSSF.

The unchanged original Stage39 KIERequest/KIEResponse, FreshnessAnchor, FullProfilePolicy and StreamRelationalVerifier are imported from hybrid/vendor/stage39_kie.py, checked against Git blob 050eb43dbb4a4fb0eecaa5b81ef65ab130c8dc28. An independently authored two-slot deployment policy defines expected type, FQDN, IP, service, version and port. The adapter maps actual NFProfile fields and never fills missing fields from expected policy.

NF-side Python agents sign KIE responses bound to controller, subject, cycle, nonce, policy and normalized-profile digest. Native C processes transport this metadata; cryptographic generation is not implemented inside C. A separate witness container signs direct runtime-export/proc/TCP observations. It never reads NRF discovery. Its private key is not mounted into the core container. A host-side lifecycle authority signs only scoped restart permissions; its key is also absent from the core container.

Three deployments each exercise five live challenge cycles and two authorized real NF restarts. A witness-channel absence must produce UNKNOWN and a fresh recovery must succeed. Forged MAC/signature, replay, scope and profile-integrity controls modify offline copies only. They are protocol correctness checks, not live malicious attacks. Reports preserve the distinction.

Trust limits: all containers share a host kernel; the runtime identity file originates in NF instrumentation and is not hardware attestation. PID namespace and mount separation do not protect against a compromised host. Same-container NF/agent/controller placement is a prototype trust assumption, not isolation of mutually hostile processes. No claim of arbitrary NF compromise resistance, production 3GPP deployment, standard CU interoperability, ML gain, scheduler gain or new attack detection rate follows.

The previous full official Open5GS registration-suite run remains a separate experiment; this series has only NRF/AUSF/NSSF and no UE/gNB/PDU-session workload. Reproduce using .github/workflows/native-cu-witness.yml. Artifacts omit all private keys.
