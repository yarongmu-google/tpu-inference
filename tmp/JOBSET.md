# Sweep jobs

Docker provides the runtime. CDK/GKE still provides TPU allocation and a mounted
persistent output directory. Do not use the container's writable layer as the
only copy of results.

The launcher expects an immutable image containing the reviewed repository at
`/opt/tpu-inference`, a pinned InferenceX checkout at `/opt/InferenceX`, and the
complete Python environment. The source must be installed from that same
repository. Do not map an input directory over either installed checkout.

The old `vllm12` package export and source revisions are still needed before
building its replacement. The existing `docker/Dockerfile` is a starting point,
but its default vLLM checkout and unpinned requirements do not reproduce that
environment. Record Python, package versions, editable source revisions and
local changes; do not substitute current upstream requirements for that export.

Before submission, confirm the installed CDK version's recipe discovery,
`IMAGE`/`SCRIPT`/`RESUME_DIR` substitutions and injected writable
`CDK_OUTPUT_DIR` mount. The recipe must resolve `tmp/jobset.yml`. Check the
rendered manifest against the cluster's accelerator labels, resource capacity,
service account, model access and cache mounts. The supplied template targets
one TPU7x host with four physical chips. Verify eight local TPU JAX devices
before spending time on model loading. CPU/RAM/scratch sizing and persistent
model-cache configuration depend on the available node pool.

After the image has been built and published to the chosen registry:

```bash
bash tmp/run.sh REGISTRY/IMAGE@sha256:DIGEST
```

The image runs the remaining EP/4I sweep by default. A second argument replaces
the script with another path inside the image. The current manifest assumes
CDK's template conventions; it has not been validated against the installed CDK.

Each attempt publishes under `$CDK_OUTPUT_DIR/attempts/ATTEMPT_ID`. It contains
`job.log`, `environment.json`, per-config client JSON, server logs, `status.json`
and a checksum manifest. Both successful and failed attempts are retained.
The runner snapshots active logs every 30 seconds and publishes final status
on normal exit or handled termination. A forced kill or storage outage can
leave only an earlier snapshot; it is never classified as a completed run.

For resumption, arrange for a previous attempt directory to be mounted read-only
and supply its absolute mounted path as the third launcher argument. The runner
checks the image identity and JSON hashes before restoring client results.
The sweep then skips only results satisfying its existing completion checks.
No previous attempt is assumed to be visible automatically in a new CDK job.

Collect results on the machine with the local Git checkout:

```bash
bash tmp/cleanup.sh gs://BUCKET/JOB_OUTPUT_PREFIX NAMESPACE JOBSET_NAME
```

Use the actual output prefix and Kubernetes JobSet name reported by CDK. The
last two arguments are optional; with them, collection also attempts to capture
pod state, current/previous container logs and pod events. These help diagnose
failures before the runner starts, such as an image pull or mount failure.
This requires local `gcloud` and, for cluster diagnostics, `kubectl` access.
The image itself does not need gcloud when CDK supplies the filesystem mount.

Downloads go to a fresh `tmp/vllm_logs/jobsets/collection-*` directory. Failed
runs are valid diagnostic bundles and are collected like successful runs.
Missing manifests, changed checksums and download failures cause a nonzero
collector exit while preserving the available local files. Remote data is
never deleted. Review the downloaded results, then stage and commit them
locally; the collector prints the staging command and does not commit or push.
