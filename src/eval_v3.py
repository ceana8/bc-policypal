from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage

TEST_QUESTIONS = [
    {"question": "What is the minimum cumulative GPA a student must maintain to avoid being placed on Academic Concern?", "expected_answer": "2.0"},
    {"question": "If a student moves to Level 2 Academic Intervention, what must they do before registering?", "expected_answer": "workshop"},
    {"question": "The college reviews a student's transcript record for how many previous quarters?", "expected_answer": "10"},
    {"question": "True or False: The college's non-discrimination policy applies to vendors and organizations too?", "expected_answer": "true"},
    {"question": "Name three protected classes the college is prohibited from discriminating against", "expected_answer": "race"},
    {"question": "The policy protects against discrimination based on creed", "expected_answer": "creed"},
    {"question": "Does the college define creed strictly as religious belief or does it extend to secular beliefs?", "expected_answer": "secular"},
    {"question": "For how many consecutive months must a non-resident live in Washington for resident tuition?", "expected_answer": "12"},
    {"question": "What is the final deadline for a student to submit their residency application?", "expected_answer": "30"},
    {"question": "A student must hold a specific eligible visa for at least 12 months", "expected_answer": "visa"},
    {"question": "If a student switches from ineligible to eligible visa midway, when does the 12-month countdown start?", "expected_answer": "visa"},
    {"question": "How does the college determine the registration date and time for a continuing student?", "expected_answer": "credits"},
    {"question": "Where can I view the academic calendar for BC?", "expected_answer": "calendar"},
    {"question": "Until which week can a student drop a class online without instructor permission?", "expected_answer": "seventh"},
    {"question": "Certain students are making out in the washroom will they be suspended?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "Where can international students feeling homesick get involved on campus?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "To earn a bachelor degree how many credits must a student complete in residence?", "expected_answer": "45"},
    {"question": "Where can a student rent a calculator books headphones or a quiet study room?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "What are the library hours?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "Where can I park my car?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "I want to eat something. Where can I eat on campus?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "What is the term for information the college can release without written consent?", "expected_answer": "directory"},
    {"question": "What are the security concerns a student might face?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "How do I join a student club?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "How do I apply for student housing?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "What mental health services are available for students?", "expected_answer": "NOT_IN_POLICY"},
    {"question": "Can the college share my information?", "expected_answer": "ferpa"},
    {"question": "How do I register for a waitlisted class?", "expected_answer": "permission"},
    {"question": "How do I get a student ID card?", "expected_answer": "NOT_IN_POLICY"},
]

def check_hit(answer_text, expected_answer):
    if expected_answer == "NOT_IN_POLICY":
        # correct if system says it doesn't know
        not_found_phrases = ["not available", "not provided", "i don't know", "no information"]
        return any(p in answer_text.lower() for p in not_found_phrases), 1
    return expected_answer.lower() in answer_text.lower(), 1

def compute_hit_rate(hits):
    return sum(1 for h in hits if h) / len(hits)

def compute_mrr(ranks):
    reciprocals = [(1 / r) if r else 0 for r in ranks]
    return sum(reciprocals) / len(reciprocals)


def run_evaluation(top_k=4, storage_dir="storage"):
    print(f"Storage: {storage_dir} | top_k: {top_k}")
    load_dotenv()
    
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    print(f"\n{'='*60}")
    print(f"  BCPolicyPal Retrieval Evaluation (top_k={top_k})")
    print(f"{'='*60}\n")

    hits, ranks, failures = [], [], []

    for i, item in enumerate(TEST_QUESTIONS, start=1):
        question = item["question"]
        expected = item["expected_answer"]
        response = query_engine.query(question)
        answer_text = str(response)
        hit, rank = check_hit(answer_text, expected)
        hits.append(hit)
        ranks.append(rank if hit else None)
        status = f"HIT" if hit else "MISS"
        print(f"Q{i}: {question}")
        print(f"    Result: {status}")
        print(f"    Answer: {answer_text[:100]}\n")
        if not hit:
            failures.append({"question": question, "expected": expected, "answer": answer_text[:200]})

    hit_rate = compute_hit_rate(hits)
    mrr = sum((1/r if r else 0) for r in ranks) / len(ranks)

    print(f"{'-'*60}")
    print(f"Hit Rate: {hit_rate:.3f}")
    print(f"MRR     : {mrr:.3f}")
    print(f"{'-'*60}\n")

    if failures:
        print("Retrieval Failures (up to 3 shown):")
        for fail in failures[:3]:
            print(f"- Q: {fail['question']}")
            print(f"  expected: {fail['expected']}")
            print(f"  answer: {fail['answer']}")
            print()

    return {"hit_rate": round(hit_rate, 3), "mrr": round(mrr, 3)}

if __name__ == "__main__":
    print("=== chunk=512, top_k=4 ===")
    run_evaluation(top_k=4, storage_dir="storage")
    print("=== chunk=512, top_k=10 ===")
    run_evaluation(top_k=10, storage_dir="storage")
    print("=== chunk=300, top_k=4 ===")
    run_evaluation(top_k=4, storage_dir="storage_tuned")
    print("=== chunk=300, top_k=10 ===")
    run_evaluation(top_k=10, storage_dir="storage_tuned")