# Lab 7.3 - Retrieval Tuning Report

## Objective
Improve retrieval performance through chunking, metadata strategy, and retrieval parameter updates while keeping the original pipeline unchanged.

## What Was Tuned
A separate tuned pipeline was created so the original files stayed intact.

Tuned files:
- [src/ingest_tuned.py](src/ingest_tuned.py)
- [src/query_tuned.py](src/query_tuned.py)
- [src/eval_tuned.py](src/eval_tuned.py)
- [src/eval_tuned_best.py](src/eval_tuned_best.py)

## Experiment Setup
The sweep tested:
- Chunk sizes: 300, 400, 512
- Retrieval depth: top_k = 4, 6, 8, 10
- Overlap: 80
- Metadata tags: source, topic, type

The topic metadata included:
- ferpa
- residency
- registration
- academic_standing
- equity
- general

## Results Summary

### Policy-only QA Results
These scores ignore clearly non-policy questions and focus on policy-focused evaluation items.

| Chunk Size | top_k | Policy Hit Rate | Policy MRR |
| --- | ---: | ---: | ---: |
| 300 | 4 | 0.889 | 0.755 |
| 300 | 6 | 0.889 | 0.755 |
| 300 | 8 | 0.889 | 0.755 |
| 300 | 10 | 0.889 | 0.755 |
| 400 | 4 | 0.889 | 0.792 |
| 400 | 6 | 0.889 | 0.792 |
| 400 | 8 | 0.889 | 0.792 |
| 400 | 10 | 0.889 | 0.792 |
| 512 | 4 | 0.778 | 0.694 |
| 512 | 6 | 0.833 | 0.704 |
| 512 | 8 | 0.833 | 0.704 |
| 512 | 10 | 0.889 | 0.709 |

### Best-Case OOS-Calibrated Results
Using the calibrated out-of-scope threshold sweep, the best threshold was 0.82.

| Chunk Size | top_k | Threshold | Overall Hit Rate | Overall MRR | OOS Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 400 | 4 | 0.82 | 0.931 | 0.871 | 1.000 |

## Best Configuration
Best overall configuration from the sweep:
- Chunk size: 400
- top_k: 4
- OOS threshold: 0.82

Why this configuration won:
- Chunk size 400 preserved enough context without adding too much retrieval noise.
- Increasing top_k beyond 4 did not improve metrics in this corpus.
- The OOS threshold of 0.82 correctly rejected the out-of-scope examples in the tested set.

## Tradeoff Observations
- Smaller chunks (300) increased granularity but sometimes reduced contextual completeness.
- Larger chunks (512) preserved more context but occasionally added retrieval noise.
- Chunk size 400 gave the best balance between semantic completeness and retrieval precision.
- Retrieval depth plateaued after the first few nodes, so top_k values above 4 gave little improvement.

## Technical Limitations
- OOS detection was initially weak with threshold-only rejection.
- Some unrelated questions still matched semantically similar policy chunks.
- Metadata improved organization and reproducibility, but did not fully solve OOS classification.

## Future Improvements
Potential next steps:
- Hybrid retrieval with vector + BM25
- Embedding model comparison
- Cross-encoder reranking
- Semantic query rewriting
- Confidence-based OOS classification
- Metadata-aware reranking
- Dynamic chunk sizing

## Engineering Notes
- Original files were preserved.
- Tuned experiments were isolated from the baseline pipeline.
- Each configuration used separately persisted indexes.
- The same QA set and scoring logic were reused across experiments for consistency.

## Conclusion
Lab 7.3 tuning improved retrieval quality primarily through chunk optimization. The best measured configuration was chunk size 400 with top_k 4. Using the calibrated OOS threshold of 0.82 produced the strongest overall best-case evaluation.
