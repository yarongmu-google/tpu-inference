# Described jobs

A description supplies code, inputs, arguments, output locations and independent
cases. An environment profile supplies the immutable runtime image, hardware,
cloud project, storage region and workload identity. Each submitted job receives
its own named bucket and saved execution record. No application-specific
completion rules are built into the controller.

This is a separate entry point from the existing scripts. It does not alter or
resume executions created by another launcher.

## Serving sweep through this workflow

From the repository root on the CPU VM:

```bash
bash workflow/run_sweep.sh
```

This selects the concrete `workflow/sweep.yml` description. It snapshots the
three existing sweep scripts, runs their configured points sequentially in one
TPU job, and sends client results, server logs and installed-source metadata to
the generic output collector. The generic controller has no knowledge of the
sweep's model, configurations or metrics. The first workload is the actual
sweep; the example below remains an optional small storage check.

On first use, the same logged command creates `workflow/local/environment.json`
interactively. It offers the published image digest and assigned Kubernetes
service-account name from the CPU VM's previous saved execution when available.
The registry's project/region are suggestions that require confirmation for
bucket storage. The workload's bucket-access IAM identity must be supplied or
come from an existing profile; it is never guessed from the account name.
Hardware settings are also confirmed once. A saved profile is reused without
rewriting it. Then the controller asks for a run name.

The published image supplies the installed packages and benchmark client. The
sweep description uses the image's existing source-verification bootstrap with
installation disabled. This launches through the new controller and dedicated
bucket path; it does not invoke the previous launcher. No image rebuild occurs.
The profile must identify an image compatible with these scripts and containing
the expected installed-source and benchmark-client paths.

`--dry-run --name check` can be passed to `run_sweep.sh` to prepare and inspect
its actual snapshots and recipe without creating cloud resources. Each normal
invocation starts a new campaign; use the generic `resume` command with its
printed saved path to reconnect. It does not import results or resume jobs
created by the previous launcher. The profile helper itself runs no cloud
commands and does not read credential files.

## First run

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
  one independent job and bucket; this version does not interpret a matrix.
  `max_in_flight`, up to 50, is a limit per campaign. Transfers and control
  commands also have smaller independent concurrency limits. Actual scheduling
  depends on cluster admission.

Every program receives `RUN_NAME`, `RUN_ID`, `OUTPUT_DIR` and
`INPUT_<NAME>_DIR` for each declared input. Programs should write their results
under `OUTPUT_DIR`. The wrapper captures stdout, stderr and exit status even
when the program fails. It preserves executable file modes when copying code.

A source snapshot is prepared once per campaign; local per-job copies use hard
links to that frozen snapshot, then each dedicated bucket gets its own upload.
Do not edit files inside saved execution directories. Source edits after
preparation do not change a queued job. Hashes are checked before upload and
again before running. No Git commit or push is performed by this workflow.

## Image reuse

The environment profile specifies a published image digest. The launcher never
builds or retags it. Standalone script changes are sent as code snapshots. For
Python package changes, make the intended source importable and use
`verify_imports` to catch an unexpected import path. The runner does not run
package installers or rebuild native extensions. Changes to dependencies,
package metadata/entry points, compiled modules or ABI requirements need an
updated compatible image or separately prepared artifact. A fresh TPU job may
still need model loading and JIT compilation when the image is reused.

The small generic wrapper is embedded in each generated recipe. It does not
require rebuilding the runtime image or CDK's directory-mapping feature.

## Dedicated storage and CDK

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

Each bucket contains:

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
changed or growing files produce new objects until manual bucket cleanup.
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

Cleanup requires one exact job directory, confirms the remote job has stopped
and archival has finished (or failed), checks ownership, and re-verifies the
local downloaded files. It asks the user to type `DELETE <run-id>` before
removing the bucket and all of its objects. Local results, runtime images and
CDK-managed metadata/logs are preserved. `--discard-incomplete` explicitly
permits cleanup without a verified result bundle; it still requires ownership,
terminal job state and typed confirmation. Cleanup never automatically runs
when the workload finishes. If cloud policy prevents deletion, its error is
retained and the bucket is not reported as deleted.

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
