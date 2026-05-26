# Threshold Report - Lab 7.3

## Purpose
This report summarizes the out-of-scope (OOS) threshold sweep used to calibrate rejection behavior in the tuned retrieval pipeline.

## What Was Tested
The threshold sweep was run on the best tuned retrieval setup:
- Chunk size: 400
- top_k: 4
- Same QA set as the tuned evaluation

Why chunk 400 and top_k 4 were fixed for threshold testing:
- Chunk 400 and top_k 4 were the best retrieval settings from the chunk/top_k sweep.
- Threshold calibration should isolate one variable at a time.
- Fixing chunk and top_k ensures performance changes come from threshold changes only.
- This prevents mixing retrieval tuning effects with OOS cutoff tuning effects.

Thresholds tested:
- 0.60
- 0.65
- 0.70
- 0.72
- 0.74
- 0.76
- 0.78
- 0.80
- 0.82
- 0.84
- 0.86
- 0.88
- 0.90

## Threshold Sweep Results

| Threshold | Overall Hit Rate | Overall MRR | OOS Accuracy |
| --- | ---: | ---: | ---: |
| 0.60 | 0.552 | 0.491 | 0.000 |
| 0.65 | 0.552 | 0.491 | 0.000 |
| 0.70 | 0.552 | 0.491 | 0.000 |
| 0.72 | 0.552 | 0.491 | 0.000 |
| 0.74 | 0.621 | 0.560 | 0.182 |
| 0.76 | 0.690 | 0.629 | 0.364 |
| 0.78 | 0.828 | 0.767 | 0.727 |
| 0.80 | 0.897 | 0.836 | 0.909 |
| 0.82 | 0.931 | 0.871 | 1.000 |
| 0.84 | 0.931 | 0.871 | 1.000 |
| 0.86 | 0.931 | 0.871 | 1.000 |
| 0.88 | 0.931 | 0.871 | 1.000 |
| 0.90 | 0.931 | 0.871 | 1.000 |

## Best Threshold
Best threshold from the sweep:
- `oos_threshold = 0.82`

At this threshold:
- Overall Hit Rate: 0.931
- Overall MRR: 0.871
- OOS Accuracy: 1.000

## Interpretation
- Thresholds below 0.74 were too strict and failed to reject OOS questions correctly.
- Performance improved sharply once the threshold reached 0.78 and above.
- The best OOS rejection behavior was achieved at 0.82.
- The OOS cutoff was not learned automatically; it was calibrated by sweep against the existing tuned index.

## Conclusion
The threshold sweep showed that the tuned index required a much higher OOS cutoff than the initial heuristic of 0.28. A threshold of 0.82 produced the best rejection performance while preserving strong retrieval quality on the policy-focused questions.
