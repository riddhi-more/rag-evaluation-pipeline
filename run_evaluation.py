# ── RUN EVALUATION ──────────────────────────────────────
# RAGAS version 0.1.21 — works with LangChain + Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from test_data import get_test_data
from evaluator import run_rag_pipeline

load_dotenv()

# ── CONFIGURATION ──────────────────────────────────────
PDF_PATH       = "company_policy.pdf"
OUTPUT_CSV     = "evaluation_results.csv"
PASS_THRESHOLD = 0.7

# ── LLM FOR RAGAS ──────────────────────────────────────
# Wrap ChatGroq in LangchainLLMWrapper for RAGAS 0.1.21
langchain_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)
ragas_llm = LangchainLLMWrapper(langchain_llm)

# ── EMBEDDINGS FOR RAGAS ───────────────────────────────
# RAGAS needs embeddings to score answer relevancy
# Use same model as our RAG system — no OpenAI needed
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# ── ASSIGN LLM TO METRICS ──────────────────────────────
# In RAGAS 0.1.21 metrics are module-level objects
# Assign LLM directly to each metric
faithfulness.llm      = ragas_llm
answer_relevancy.llm  = ragas_llm
context_precision.llm = ragas_llm
context_recall.llm    = ragas_llm

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]


# ── RUN EVALUATION ─────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("RAG EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1 — Validate test dataset
    print("\nStep 1 — Loading and validating test dataset...")
    dataset    = get_test_data()
    test_cases = dataset.test_cases
    print(f"Validated {len(test_cases)} test cases successfully")

    # Step 2 — Run RAG pipeline
    print("\nStep 2 — Running RAG pipeline...")
    results = run_rag_pipeline(test_cases, PDF_PATH)

    # Step 3 — Prepare data for RAGAS
    print("\nStep 3 — Preparing data for RAGAS...")
    ragas_data = {
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    ragas_dataset = Dataset.from_dict(ragas_data)
    print(f"Prepared {len(ragas_dataset)} samples for evaluation")

    # Step 4 — Run RAGAS scoring
    print("\nStep 4 — Running RAGAS scoring...")
    print("This may take 2-3 minutes...")
    scores = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    # Step 5 — Print report
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PDF:       {PDF_PATH}")
    print(f"Questions: {len(test_cases)}")
    print(f"Threshold: {PASS_THRESHOLD}")
    print("-" * 60)

    # Overall scores
    scores_df   = scores.to_pandas()
    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    print("\nOVERALL SCORES:")
    all_passed = True
    for metric in metric_cols:
        if metric in scores_df.columns:
            avg    = scores_df[metric].mean()
            status = "PASS" if avg >= PASS_THRESHOLD else "FAIL"
            if status == "FAIL":
                all_passed = False
            bar = "█" * int(avg * 20)
            print(f"  {metric:<22} {avg:.3f}  {bar:<20} [{status}]")

    print("-" * 60)
    overall = "ALL METRICS PASSED ✅" if all_passed else "SOME METRICS NEED ATTENTION ⚠️"
    print(f"\nOVERALL RESULT: {overall}")

    # Per question breakdown
    print("\nPER QUESTION BREAKDOWN:")
    print("-" * 60)
    for i, row in scores_df.iterrows():
        print(f"\n[{i+1}] {results[i]['question']}")
        print(f"     Answer: {results[i]['answer'][:80]}...")
        for metric in metric_cols:
            if metric in scores_df.columns:
                score  = row[metric]
                status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
                print(f"     {metric:<22} {score:.3f}  [{status}]")

    # Step 6 — Export CSV
    print(f"\nStep 5 — Exporting results to {OUTPUT_CSV}...")
    scores_df["question"]     = ragas_data["question"]
    scores_df["answer"]       = ragas_data["answer"]
    scores_df["ground_truth"] = ragas_data["ground_truth"]
    scores_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return scores


# ── ENTRY POINT ────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()