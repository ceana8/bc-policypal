from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage

from eval_tuned import TEST_QUESTIONS, keyword_hit


def run_threshold_sweep(
    persist_dir="storage_tuned_400",
    top_k=4,
    thresholds=None,
):
    if thresholds is None:
        thresholds = [0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]

    load_dotenv()

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    rows = []
    for item in TEST_QUESTIONS:
        question = item["question"]
        expected_answer = item["expected_answer"]

        response = query_engine.query(question)
        source_nodes = response.source_nodes
        source_texts = [node.node.text for node in source_nodes]
        top_score = source_nodes[0].score if source_nodes else 0.0

        if expected_answer == "NOT_IN_POLICY":
            rows.append({"kind": "oos", "score": top_score})
        else:
            hit, rank = keyword_hit(source_texts, expected_answer)
            rows.append({"kind": "inscope", "hit": hit, "rank": rank})

    print(f"\nThreshold sweep for persist_dir={persist_dir}, top_k={top_k}")
    print("threshold overall_hit overall_mrr oos_acc")

    best = None
    for threshold in thresholds:
        hits, ranks, oos_hits = [], [], []

        for row in rows:
            if row["kind"] == "oos":
                hit = row["score"] < threshold
                rank = 1 if hit else None
                oos_hits.append(hit)
            else:
                hit = row["hit"]
                rank = row["rank"]

            hits.append(hit)
            ranks.append(rank)

        hit_rate = sum(1 for hit in hits if hit) / len(hits)
        mrr = sum((1 / rank) if rank else 0 for rank in ranks) / len(ranks)
        oos_acc = sum(1 for hit in oos_hits if hit) / len(oos_hits)

        print(f"{threshold:.2f} {hit_rate:.3f} {mrr:.3f} {oos_acc:.3f}")

        key = (oos_acc, hit_rate, mrr)
        if best is None or key > best[0]:
            best = (key, threshold, hit_rate, mrr, oos_acc)

    print("\nBEST_BY_OOS")
    print(
        "threshold={:.2f} overall_hit={:.3f} overall_mrr={:.3f} oos_acc={:.3f}".format(
            best[1], best[2], best[3], best[4]
        )
    )


if __name__ == "__main__":
    run_threshold_sweep()
