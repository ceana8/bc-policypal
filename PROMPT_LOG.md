# Prompt Log - Lab 7.3 Retrieval Tuning

Date: 2026-05-25
Project: bc-policypal 
Scope: Lab 7.3 tuning workflow

## 1. Goal Prompt
Improve retrieval performance through chunking, metadata strategy, and retrieval parameter updates while keeping original pipeline files unchanged.

## 2. Pipeline Setup Prompt
Create a separate tuned pipeline and keep original files intact.

Expected outputs:
- src/ingest_tuned.py
- src/query_tuned.py
- src/eval_tuned.py

## 3. Ingestion Tuning Prompt
Run chunk-size experiments and rebuild index for:
- chunk_size = 300
- chunk_size = 400
- chunk_size = 512

Use overlap = 80 and persist each tuned index separately.

Why overlap = 80:
- Keeps about 20% overlap with chunk_size 400, which is a practical middle ground.
- Preserves sentence and policy-rule context across chunk boundaries.
- Improves retrieval recall without creating too much duplicate chunk noise.
- Worked well with this corpus during tuning, where chunk 400 gave the best MRR.

## 4. Retrieval Parameter Prompt
For each chunked index, evaluate with:
- top_k = 4
- top_k = 6
- top_k = 8
- top_k = 10

Compare metrics using the same QA set and scorer.

## 5. Metadata Strategy Prompt
Add metadata fields during ingestion:
- source (pdf file)
- topic (ferpa, residency, registration, academic_standing, equity, general)
- type = policy

Use metadata in tuned track without modifying original track.

## 6. Evaluation Prompt
Run tuned evaluation and report:
- Overall Hit Rate
- Overall MRR
- Policy Hit Rate
- Policy MRR
- OOS Accuracy

## 7. Best Config Selection Prompt
Select best config by Policy MRR first, then Overall MRR.

Chosen config from experiment:
- chunk_size = 400
- top_k = 4

## 8. Final Reporting Prompt
Generate lab-ready summary:
- what was tuned
- experiment table
- best configuration
- key findings
- known limitations (OOS detection)

## 9. Notes
- Original files were preserved.
- Tuned files were created separately for reproducibility.
- OOS accuracy remained low with current threshold-only rejection logic.

## 10. Tradeoff Observations

- Smaller chunks (300) increased retrieval granularity but sometimes reduced contextual completeness for policy explanations.
- Larger chunks (512) preserved more context but occasionally introduced retrieval noise due to broader semantic coverage.
- Chunk size 400 provided the best balance between semantic completeness and retrieval precision, producing the strongest MRR in our experiments.

## 11. Retrieval Depth Observations

- Increasing top_k beyond 4 produced minimal improvement in retrieval metrics for the best chunk configuration.
- Retrieved results remained highly similar across top_k values, indicating relevant chunks were already ranked near the top.
- This suggests retrieval performance plateaued after the first few retrieved nodes.

## 12. Technical Limitations

- Current out-of-scope (OOS) detection relies on threshold-based rejection logic.
- Some unrelated questions were still matched to semantically similar policy chunks.
- OOS accuracy remained low (0.000 in the tuning sweep), so OOS handling remains a major gap.
- Metadata enrichment improved organization and experiment reproducibility, but did not fully resolve OOS classification.

## 13. Future Improvements

Potential future tuning directions include:

- Hybrid retrieval (vector + BM25)
- Embedding model comparison
- Cross-encoder reranking
- Semantic query rewriting
- Confidence-based OOS classification
- Metadata-aware reranking
- Dynamic chunk sizing strategies

## 14. Engineering Notes

- All tuned experiments were isolated from the baseline pipeline to preserve reproducibility.
- Each configuration used separately persisted indexes to ensure fair evaluation.
- The same QA set and scoring logic were reused across all experiments for consistency.
