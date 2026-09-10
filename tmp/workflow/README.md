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
`examples/`. `execution.scratch_gib` optionally requests and mounts local disk
at `/run-scratch`. The recipe verifier checks that CDK preserves the resources
and storage mounts before submission.

Result archives belong to the description's output directory. `local/` contains
ignored controller/build state and compressed launcher logs. A nonzero exit is
printed and retained even if image preparation fails before job submission.
After verified final collection the cleanup policy can delete the run-owned
image and storage prefix. Unverified final output retains those resources.
