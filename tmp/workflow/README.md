# Described jobs

This directory carries the generic build, submit, poll, collect, archive and
cleanup workflow used by the baseline job. It is copied from the existing
workflow with support for bounded local scratch storage.

Use `bash tmp/baselines/run.sh --name <description>` for the two baseline cases.
The checked-in `environment.yml` supplies the existing project, image registry,
CDK storage and TPU configuration. Each launch gets a unique run ID and owns
its image and GCS output prefix.

Generic entry points:

```bash
bash tmp/workflow/run.sh <description.yml> --name <description>
bash tmp/workflow/run.sh <description.yml> --name <description> --dry-run
bash tmp/workflow/run.sh resume <saved-campaign-directory>
bash tmp/workflow/run.sh status <saved-run-directory>
bash tmp/workflow/run.sh collect <saved-run-directory>
bash tmp/workflow/run.sh cleanup <saved-run-directory>
```

A description specifies its profile, image build, code snapshot, inputs, run
command, outputs and execution limits. The generic examples remain under
`examples/`. `execution.scratch_gib` sets scratch capacity at `/run-scratch`.
With `execution.scratch_storage_class`, it provisions a generic ephemeral PVC
using that class and `ReadWriteOnce`. Without a class, it remains an emptyDir
size ceiling. Neither adds a container ephemeral-storage request or limit.
The verifier checks the volume, capacity, mount and cleanup deadlines after
CDK rendering. A provisioned-scratch JobSet expires 600 seconds after completion
and its child Job has a 900-second expiry, matching the supplied working recipe.
Pod deletion garbage-collects its PVC; the storage class must use `Delete` to
reclaim the backing disk. Disk reclamation is asynchronous and is not certified
by the image/GCS cleanup flag. Results are published to GCS before the workload
exits and remain collectible after Pod deletion. Active jobs retain their disk.
Actual provisioning, capacity and disk reclamation require live validation.

Result archives belong to the description's output directory. `local/` contains
ignored controller/build state and compressed launcher logs. A nonzero exit is
printed and retained even if image preparation fails before job submission.
After verified final collection the cleanup policy can delete the run-owned
image and storage prefix. Unverified final output retains those resources.
