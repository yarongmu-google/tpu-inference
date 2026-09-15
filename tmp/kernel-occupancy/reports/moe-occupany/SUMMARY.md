# moe-occupany

0 candidate records; partial results; winners provisional.
Times are microseconds. Speedup = original / modified; >1 means modified is faster.
Wall and device rankings are separate. Failed accuracy cannot win; profiler failure only prevents device ranking.

**No accuracy-qualified winner is available. Measured timings and failure reasons follow.**

## Uniform routing

| T; be/bg/C/d1/d2/cT | Old wall | New wall | Speedup | Old TC | New TC | TC speedup | Accuracy old/new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Sparse routing

| T; be/bg/C/d1/d2/cT | Old wall | New wall | Speedup | Old TC | New TC | TC speedup | Accuracy old/new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Failures

```text
Traceback (most recent call last):
  File "/home/ymu_google_com/tpu-inference-topk/tmp/workflow/controller.py", line 773, in job_action
    return job.run()
           ^^^^^^^^^
  File "/home/ymu_google_com/tpu-inference-topk/tmp/workflow/controller.py", line 597, in run
    self.register()
  File "/home/ymu_google_com/tpu-inference-topk/tmp/workflow/controller.py", line 327, in register
    _, text = self.cdk(args=['recipe', 'list', '-o', 'json'])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ymu_google_com/tpu-inference-topk/tmp/workflow/controller.py", line 239, in cdk
    raise ValueError('CDK instructions changed; review the saved command output')
ValueError: CDK instructions changed; review the saved command output

```

```text
last_error: CDK instructions changed; review the saved command output
```

```text
No candidate outcomes were collected. This is a diagnostic report, not a successful tuning result.
```

```text
Full artifact collection is unverified in saved state; original data is retained.
```

