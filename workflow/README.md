# Described jobs

A description supplies code, inputs, arguments, output locations and independent
cases. An environment profile supplies the runtime image or image registry destination,
hardware, cloud project and storage mode. Each submitted job receives
its own storage location and saved execution record. No application-specific
completion rules are built into the controller.

This is a separate entry point from the existing scripts. It does not alter or
resume executions created by another launcher.

## Serving sweep through this workflow

From the repository root on the CPU VM, with the existing Python 3.12
`vllm12` environment active:

```bash
bash workflow/run_sweep.sh
```

This selects the concrete `workflow/sweep.yml` description. It snapshots the
three existing sweep scripts, runs their configured points sequentially in one
TPU job, and sends client results, server logs and installed-source metadata to
the generic output collector. The generic controller has no knowledge of the
sweep's model, configurations or metrics. The first workload is the actual
sweep; the example below remains an optional small storage check.

The checked-in `workflow/sweep-environment.yml` supplies the earlier launcher's
project, registry and TPU settings. The only prompt is the run name; use
`--name my-description` to supply it on the command line. No project, bucket,
image-repository, digest, service-account or IAM questions are asked. Existing
CPU-VM CDK/gcloud authentication and registry permissions are still required.

The saved image prefix is
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/runtime`.
Each run publishes `<prefix>/<run-id>:run`, then submits its immutable digest.
The exact path and digest remain in the local resource and build records.
CDK assigns the Kubernetes service account and mounts its normal output folder;
the sweep does not create a bucket or change IAM. Its files live only under
`gs://cloud-devkit/jobs/<job-id>/outputs/workflow-<run-id>/`.
This preset is used directly; an older `workflow/local/environment.json` is not
consulted or overwritten by the sweep launcher.

Every new TPU job invokes the Docker builder and CPU smoke check, even when the
source and environment match a previous job. The configured image repository is
a prefix: the launcher appends the unique run ID as a separate image path. Its
CDK storage subfolder uses that same run ID. Independent cases each build and own their own
image; no completed image is selected from a shared-image cache. Docker may reuse
build layers, and the local image-preparation lock serializes builds on this CPU
VM while already submitted TPU jobs continue concurrently.

The helper finds editable vLLM in the active CPU-VM environment. It uses
`INFERENCEX_REPO`, an existing `/tmp/InferenceX`, or a persistent checkout under
`workflow/local/images/InferenceX`. That checkout is cloned once when needed;
it is not automatically pulled on later runs. Relevant tracked edits and
untracked source files must be committed before building so they cannot be
silently omitted. Source revisions, a build-input fingerprint and the published
digest are saved for each run. The existing image build installs editable
projects and CPU Torch packages; TPU execution disables installation.

The run's `resources.json` is written before any build or submission. Its
`image/` directory retains build logs, publication progress and early errors.
`image-build.json` records the final image digest and source provenance. A failed
build never submits a TPU job. After an interrupted build, resume can recover an
already saved publication result; it never silently starts another build under
the same run ID. Start a new run to rebuild.

The sweep selects `execution.cleanup: after_collection`. Once the job is terminal
and archiving has completed or failed, the controller verifies the downloaded
final artifact bundle before deleting that run's storage subfolder and complete registry
image package, including its tags and versions. This applies to both successful
and failed workloads. Active jobs, unknown submissions and incomplete or damaged
local collections retain their resources. Resume retries collection and cleanup;
it does not resubmit the completed job. Cleanup of a failed pre-submission build
also removes its owned image resources once local error logs are saved.

Cleanup removes only the two local Docker tags belonging to the run. Shared
Docker build layers, the persistent benchmark-client checkout, older images,
local results and CDK service metadata are preserved. Local Docker build-cache
pruning is a separate operation. The parent Artifact Registry repository is not
deleted or automatically created; it must already exist and permit the CPU VM
to publish, list and delete the run's image packages. The helper checks that the
repository exists and uses Docker format before starting the expensive build.

Cleanup progress is saved after each resource. If deletion completes just before
the controller is interrupted, a successful resource listing confirms absence on
resume. Permission and network errors do not count as absence. `cleanup.json`
keeps the run ID, image digest, deleted resource paths, outcome and cleanup time.
A fresh retry after cleanup always builds a new image under a new run ID.

`--dry-run --name check` can be passed to `run_sweep.sh` to inspect its frozen
scripts and build plan without building, publishing or creating cloud resources.
For descriptions with image preparation, the digest and job recipe remain
unresolved until each job builds its image. Dry-run records cannot be submitted. Each normal
invocation starts a new campaign; use the generic `resume` command with its
printed saved path to reconnect. It does not import results or resume jobs
created by the previous launcher. The profile helper itself runs no cloud
commands and does not read credential files.

## Other descriptions with dedicated buckets

Fill in `examples/environment.yml` once for the CPU VM and cluster. The image
must already exist in a registry and be readable by GKE. The image needs Python
3.11 or newer, the libraries your program imports, and a writable container
filesystem. It needs neither conda activation nor gcloud. The CPU VM needs
Python 3.11 or newer, gcloud, CDK, and their existing authentication.

The `cloud.service_account` field is the Kubernetes service-account name CDK
assigns to a job, as shown in its rendered recipe. `cloud.workload_iam_member`
is the exact IAM principal through which that account accesses Cloud Storage.
They are different identifiers. Use the cluster's configured identity mapping;
do not derive an IAM identity from the account name alone. The submitting
account must be able to create buckets and update their IAM policies in the
specified project. Only the dedicated bucket gets an object-user grant.

Start with the small execution/storage check:

```bash
bash workflow/run.sh workflow/examples/experiment.yml
```

The shell creates a workflow-local Python environment for the pinned YAML
parser; it does not modify your active environment. Its complete transcript,
including setup and early validation failures, is saved under
`workflow/local/logs/`. The runner asks for a name if the description leaves it
empty. `--name` can supply one for noninteractive execution. That display name
is normalized for cloud labels; a random suffix makes each job/bucket unique.
The selected name appears in cloud metadata and the globally visible bucket
namespace, so use a name suitable for that destination.

A local preparation pass is available without any cloud commands:

```bash
bash workflow/run.sh workflow/examples/experiment.yml --name storage-check --dry-run
```

It writes the frozen inputs and rendered submission proposal into the results
root. Dry-run executions cannot later be submitted: invoke the description
again without `--dry-run` when ready. Setup may download the YAML dependency
on the first invocation even in dry-run mode.

## Description contract

All relative CPU-VM paths resolve against the description file, including its
profile path. YAML and JSON are accepted, with duplicate and unknown keys
rejected. See `examples/experiment.yml` for the minimal description.

- `image_build` (optional): a local preparation command with `cwd`, `argv`, and
  optional `timeout_seconds` (default 28800, maximum 86400). `cwd` resolves against
  the description file. The command receives `IMAGE_REPOSITORY`, `IMAGE_RESULT`,
  and `IMAGE_BUILD_LOG_DIR`. It must write a JSON object containing an immutable
  `image` digest in that repository to `IMAGE_RESULT`; other fields record its
  provenance. Output is streamed into local logs. Failures, interruption, timeout,
  or an invalid digest block job preparation. The profile supplies
  `runtime.repository`; without `image_build`, it supplies `runtime.image` as before.
  Preparation runs once per new job, after experiment snapshots. The command
  also receives `IMAGE_RUN_ID` and must return that ID as `owner_run_id`. Its
  destination already includes the unique run ID. Resume reuses only that same
  run's saved result; it never starts a second build under the run ID. Use a command appropriate to your project; the controller contains
  no application-specific build logic.
- `code.directory`: one directory to freeze; `include` optionally selects paths
  relative to it using shell-style file patterns. The default is `['**']`.
  `.git`, `.venv`, and `__pycache__` directories are excluded. Symlinks and
  non-regular files are rejected. Everything else selected is uploaded.
- `code.destination`: where the snapshot is copied inside the container. Use a
  fresh workspace directory for standalone scripts. Copying onto an existing
  source directory replaces matching files and preserves unlisted files,
  including existing native modules. It does not remove deleted source files
  from a base image. Use a fresh source path for exact tree replacement.
- `inputs`: named directories with `source` on the CPU VM and `destination`
  inside the container. This version snapshots local directories only; large
  shared datasets should remain outside these per-run uploads until a shared
  input adapter is added.
- `run.argv`: argument list, such as `[python3, main.py]` or `[bash, run.sh]`.
  `${NAME}` substitutions come from the execution environment. They are not
  evaluated as shell code. `run.env` supplies literal values; cases can override
  them. A Bash entry point may itself evaluate its script normally.
- `run.cwd`: container working directory. `run.verify_imports` optionally maps
  Python module names to the expected source-directory roots; these are checked
  with the command's cwd/environment before the program starts.
- `outputs.directory`: local results root, outside uploaded code/input trees.
  Every job writes into its own directory there. `snapshot_seconds` defaults to
  30. `outputs.extra` can map names to additional container directories for
  programs that already write to fixed paths. The name `output` is reserved.
- `execution.timeout_seconds`: program execution limit. Submission adds 30
  minutes for storage setup and code/input copying. Queue/controller waiting
  and archive waiting have separate `wait_seconds` and `archive_seconds`.
- `cases`: optional list of unique names and environment overrides. Each case is
  one independent job and storage location; this version does not interpret a matrix.
  `max_in_flight`, up to 50, is a limit per campaign. Transfers and control
  commands also have smaller independent concurrency limits. Actual scheduling
  depends on cluster admission.

Every program receives `RUN_NAME`, `RUN_ID`, `OUTPUT_DIR` and
`INPUT_<NAME>_DIR` for each declared input. Programs should write their results
under `OUTPUT_DIR`. The wrapper captures stdout, stderr and exit status even
when the program fails. It preserves executable file modes when copying code.

A source snapshot is prepared once per campaign; local per-job copies use hard
links to that frozen snapshot, then each run storage location gets its own upload.
Do not edit files inside saved execution directories. Source edits after
preparation do not change a queued job. Hashes are checked before upload and
again before running. No Git commit or push is performed by this workflow.

## Runtime images

With `image_build`, each new job builds and publishes into its own run-specific
image path. Code or dependency changes are included by rebuilding; a digest is
used only to pin that job's actual image. Without `image_build`, a description
can still use a supplied immutable `runtime.image`; such images are externally
managed and are never automatically deleted by this workflow. Existing saved
runs keep their original image and cleanup behavior. Fresh TPU jobs still need
model loading and JIT compilation even when Docker reuses build layers.

The small generic wrapper is embedded in each generated recipe. It does not
require rebuilding the runtime image or CDK's directory-mapping feature.

## Storage and CDK

The sweep selects `storage.mode: cdk` with `storage.outputs_root` set to the
CDK job root. In this mode the profile needs only `cloud.project`, runtime and
hardware settings. The controller submits with CDK's output mount enabled,
resolves the confirmed job ID, then uploads the frozen inputs into that job's
unique workflow subfolder. The container waits for these inputs at
`$CDK_OUTPUT_DIR/workflow-<run-id>` before checking ownership, hashes and storage
read/write access. The controller checks the rendered image, command and
hardware before authorizing the workload. The observed service account is
recorded, with no manual identity configuration.

After verified collection, cleanup checks the owner marker and empties only
that run's subfolder using a scoped
[`gcloud storage rsync`](https://docs.cloud.google.com/sdk/gcloud/reference/storage/rsync)
from an empty local directory. Interrupted deletion can be retried. It does not
delete the shared bucket, other runs, or CDK's own logs and metadata. The shared
bucket's retention and soft-delete policies remain under CDK administration.
Local tests mock cloud commands; CPU-VM access to this prefix and the live CDK
mount still require verification on the first run.

The older dedicated-bucket mode remains supported for other descriptions and
saved runs. Its behavior is described below.


The generated recipe uses a Cloud Storage FUSE CSI volume pointing to the
new bucket at `/run-storage`. The default CDK output mount is disabled for this
recipe; there is no fallback to it. CDK's own job metadata and transported
console logs can still be stored in CDK-managed storage.

The Kubernetes service account is omitted from the submitted template because
CDK assigns it. After submission, the controller parses the rendered recipe
and verifies the account, image, command, hardware, and dedicated volume. Only
then does it publish a launch-authorization marker. Independently, the wrapper
requires an actual FUSE mount and verifies the run identity and bucket
read/write access before it starts the supplied program. Failed validation
leaves the job and diagnostics available; it does not silently change storage.
The initial check still requires a real cluster run: admission of the CSI
volume, workload identity, and GCS FUSE behavior cannot be established by local
mocked tests. If CDK strips the volume or the cluster disallows it, the workload
will not start. The controller does not change cluster-wide settings.

Bucket creation is explicit about project, region, Standard storage, uniform
bucket-level access, public access prevention, and disabled soft deletion.
Object versioning and retention locks are not enabled. Existing conflicting
buckets are not adopted. The saved ownership marker and labels are checked
before reuse and cleanup. An ambiguous create outcome requires inspection;
the controller will not guess ownership.

Each run storage location contains:

```text
owner.json
input/                  frozen source, inputs, and run configuration
control/                storage readiness and launch authorization
objects/                artifact snapshots addressed by their file hashes
manifest.json           current snapshot and execution status
```

Artifacts are copied to container-local scratch, then published periodically.
The manifest references closed objects and records hashes, sizes, image digest,
configuration identity and exit status. Repeated identical files reuse objects;
changed or growing files produce new objects until bucket cleanup.
Abrupt container loss may lose writes since the latest successful snapshot.
A final publication failure returns a nonzero exit code. Checksums detect an
incomplete upload rather than presenting it as a complete result.

The controller reports execution and CDK archival states separately. While the
job runs it retrieves recent stdout/stderr snapshots and their age; full result
artifacts are collected after completion. Large console logs can still make
these snapshots expensive. CPU-VM disconnects leave already published data in
GCS. Full collection also attempts CDK description and console-log retrieval,
including for jobs that never started their program.

## Resume, collect, and cleanup

For newly built images, `execution.cleanup: after_collection` opts each run into
automatic image and run-storage cleanup after verified collection. It overrides the
profile's manual-storage default for that run. The default for other descriptions
is `manual`. The same description can select `manual` to retain its owned image
and storage until explicit cleanup. Automatic cleanup requires a run-owned build.
Existing saved runs are not retroactively opted into deletion.

After automatic cleanup, `resume` returns the recorded outcome and `collect`
verifies the retained local bundle. To execute again, start a new named run.

Each new invocation with a description creates a new campaign. Saved execution
paths are explicit, so concurrent campaigns cannot accidentally resume each
other. Replace the example paths below with the path printed by the launcher:

```bash
bash workflow/run.sh status /path/to/saved-campaign
bash workflow/run.sh resume /path/to/saved-campaign
bash workflow/run.sh collect /path/to/saved-job
bash workflow/run.sh cleanup /path/to/saved-job
```

A resumed campaign preserves its source snapshots, profile, names, buckets and
job IDs. Completed executions are not resubmitted. Unknown submission outcomes
are reconciled by the saved unique tag; unresolved jobs retain their concurrency
slots. An interruption stops local monitoring, not the remote job. Command
stdout/stderr, exit codes and controller errors are retained per job. Job-local
and campaign-local locks prevent competing controllers; independent campaigns
can run simultaneously.

The explicit `cleanup` command requires one exact job directory and typed
`DELETE <run-id>` confirmation. It checks terminal/archive state, resource
ownership and local artifact hashes. For an owned-image run it deletes the
run's storage, the run's complete image package, and its local Docker tags. Older
and fixed-image runs keep the original bucket-only manual cleanup behavior.
`--discard-incomplete` explicitly permits manual cleanup without a verified
result bundle; it still checks ownership and terminal job state when submitted.

Automatic cleanup is selected only by `execution.cleanup: after_collection`;
it never discards incomplete results. If deletion fails, its error and progress
are retained and the controller returns nonzero. Resume continues collection
or cleanup from that same run record. The run image is removed through the
[Artifact Registry package deletion command](https://docs.cloud.google.com/sdk/gcloud/reference/artifacts/packages/delete),
scoped to its unique image path. No global image, bucket, or Docker-cache
pruning is performed. Local results and CDK-managed metadata/logs are preserved.

The launcher returns nonzero for failed workloads, missing/corrupt final
artifacts, unresolved submissions and archive failures. Available partial
results remain on disk. A verified artifact bundle does not prove that a
benchmark's numerical results or comparisons are valid; the supplied program
owns that validation.

## Local verification

```bash
python3 -m unittest discover -s workflow/tests -v
```

Use an interpreter with the pinned YAML dependency, or the workflow-local
interpreter after the first launcher invocation. Tests use local temporary
files, actual child processes, and mocked cloud commands. They do not create
buckets, submit jobs, or modify cluster state.

References:
- https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-ephemeral
- https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup
- https://docs.cloud.google.com/storage/docs/buckets
- https://docs.cloud.google.com/storage/docs/soft-delete
