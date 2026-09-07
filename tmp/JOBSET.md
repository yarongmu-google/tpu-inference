# Sweep jobs

On the CPU VM with `vllm12` active, run the complete workflow:

```bash
bash tmp/run.sh
```

The runner builds or reuses a verified local image, publishes it by digest,
registers a dedicated CDK recipe, submits one JobSet, polls execution and log
archiving, and downloads results and errors. Commands, stdout, stderr, exit
codes and the submitted job ID are retained under
`tmp/vllm_logs/jobset-workflows/run-*`. The terminal transcript is also saved
as `tmp/vllm_logs/jobset-launch-*.log`.

The default sweep runs the remaining 1k/8k C256 DP8_EP point, matched 4I
points at C256 and C128, and the failed 1k/1k C256 DP4TP2_EP retry, in that
order. Earlier successful configurations remain omitted from the sweep.

## Images and registration

The default image destination is the registry documented by CDK:
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/qwen-sweep-USER`,
where USER is the normalized submitting username. Each workflow gets a unique
tag; the JobSet uses the published registry digest. The account must already
have permission to push there, and GKE must be able to pull it. The runner
configures Docker's gcloud credential helper for that registry host. It does
not create a registry, change IAM, or print access tokens.

Images contain tracked vLLM, TPU-inference and InferenceX sources plus the
captured Python environment. Torch 2.10.0 and Torchvision 0.25.0 use CPU wheels.
Editable installation happens at image build time; TPU jobs check installed
source paths and start the sweep without reinstalling packages. CPU import and
CLI checks run during the build and in a fresh container. Existing dependency
conflicts remain reported; those checks do not prove TPU compatibility.

A successful local build can be reused when its source revisions and relevant
image inputs still match. Use `--rebuild` to incorporate environment/package
changes. The InferenceX revision selected by the build stays pinned in that
image. Both sides of a benchmark comparison must use the same image.

CDK reads the registry in its own checkout, not this repository's `tmp` directory.
The runner adds one uniquely named recipe under `recipes/experimental/` and
appends its registration to CDK's `recipes.yml`, preserving existing entries.
A conflicting recipe is rejected. These are local CDK checkout edits; they are
not committed or pushed by the runner. CDK's restrictions are respected: no
explicit service account, no host networking, and restart policy `Never`.
The template requests one TPU7x host with four physical chips (`2x2x1`).

CDK's current agent letter is read before every CDK command, and all acknowledgement
codes are passed in order. Changed instructions stop the workflow for review.
Jobs carry only their unique workflow tag. `kubectl` and direct MongoDB access
are not used. Leave `NO_UPDATE_CHECK` unset.

## Interruptions, failures and another run

Rerun the same command after an interruption. The saved workflow resumes the
same job by its unique tag, including when submission succeeded remotely but
its response was interrupted. An uncertain submission is never automatically
repeated. The runner does not cancel a remote job when local polling stops.
Commands have timeouts; status reads and output synchronization retry transient
failures. A submission whose outcome is still unknown must be investigated
before starting another workflow.

A completed workflow is not automatically submitted again. To launch another:

```bash
bash tmp/run.sh --new
```

Add `--rebuild` to force a fresh image. To reuse verified successful client
results from a previous attempt with the same image:

```bash
bash tmp/run.sh --new --resume-from /absolute/path/to/collected/attempt
```

The attempt directory must contain `manifest.json` and its files. Only client
JSON results and the manifest are copied into the new job's resume input;
checksums and image identity are verified. The sweep's completion checks decide
which configurations can be skipped.

Polling distinguishes `job_status` (execution) from `state` (CDK log archiving).
Success requires successful execution, completed CDK processing, and downloaded
sweep manifests with verified hashes and successful exit codes. Failed jobs,
archive errors, timeouts, missing manifests, and failed downloads return nonzero
while keeping every available diagnostic. A Log Explorer URL is identified as
potentially missing container logs, not mistaken for log text.

`cdk job sync-outputs` collects archived artifacts and available container logs.
Files are copied from CDK's local output directory into the workflow directory.
Partial copies are retained separately on retry. No remote output or job is
deleted. To collect again without submitting:

```bash
bash tmp/cleanup.sh JOB_ID
```

The runner prints the local staging command. Review, stage, commit and push
results manually; neither launcher nor collector performs Git writes.

## Optional settings

- `JOBSET_IMAGE_REPOSITORY`: Artifact Registry image path, without tag or digest.
- `JOBSET_SCRIPT`: relative script path inside the image (default: the sweep above).
- `CDK_SOURCE_DIR` and `CDK_JOB_OUTPUTS_DIR`: path overrides that must match CDK's
  own configuration. Otherwise only those two path settings are read from
  `~/.cdk.ini`, without printing or copying the credential-bearing file.
- `JOBSET_POLL_SECONDS`: 30 by default.
- `JOBSET_WAIT_SECONDS`: 86400 by default, including queue wait.
- `JOBSET_ARCHIVE_SECONDS`: 1200 after execution ends. The submitted job's active
  deadline is 43200 seconds. Polling timeouts leave the remote job available for
  inspection and resumption.

The standalone image builder remains available as `bash tmp/build_jobset_image.sh`.
Model access, TPU device initialization, and the actual sweep are validated on
the cluster; local mocked workflow tests do not establish those results.
