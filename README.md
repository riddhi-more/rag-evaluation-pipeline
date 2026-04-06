# RAG Evaluation Pipeline

An automated LLMOps evaluation pipeline that scores RAG system outputs across 4 RAGAS metrics with Pydantic data validation, configurable pass/fail thresholds, and CSV audit trail export. Built as part of a production-grade AI engineering portfolio.

## What It Does

Automatically evaluates any RAG system on 4 quality dimensions:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| Faithfulness | Did it hallucinate? | Hallucination = compliance risk |
| Answer Relevancy | Did it answer the question? | Off-topic answers erode trust |
| Context Precision | Did it retrieve the RIGHT chunks? | Bad retrieval = bad answers |
| Context Recall | Did it find ALL relevant info? | Incomplete retrieval = incomplete answers |

## Sample Output

```
============================================================
RAG EVALUATION PIPELINE
============================================================

OVERALL SCORES:
  faithfulness           0.917  ████████████████████  [PASS]
  answer_relevancy       0.987  ████████████████████  [PASS]
  context_precision      0.875  █████████████████     [PASS]
  context_recall         0.938  ██████████████████    [PASS]
------------------------------------------------------------
OVERALL RESULT: ALL METRICS PASSED ✅

Results saved to evaluation_results.csv
```

## Architecture

```
test_data.py          → Pydantic validated ground truth dataset
evaluator.py          → RAG pipeline (index → retrieve → answer)
run_evaluation.py     → RAGAS scoring + report + CSV export
company_policy.pdf    → test document
evaluation_results.csv → audit trail
```

## Why Pydantic Validation

Test datasets in production are maintained by multiple teams — AI engineers, business analysts, compliance. Without validation, one malformed entry breaks the entire evaluation run. Pydantic catches problems at load time:

```python
# Catches empty ground truth immediately
TestCase(question="What is the policy?", ground_truth="")
# ValidationError: ground_truth cannot be empty

# Catches missing question mark
TestCase(question="Tell me about shipping", ground_truth="...")
# ValidationError: Question must end with '?'
```

## Why EphemeralClient

Evaluation rebuilds the index fresh every run. EphemeralClient (in-memory) is faster than PersistentClient (disk) because there is no I/O overhead — perfect for evaluation workflows.

## NaN Handling

NaN scores indicate the metric could not be computed — not that the answer was wrong. Three causes: LLM parse failure, insufficient context, rate limiting. Production handling:

```python
# Never let NaN propagate to averages
avg = scores_df[metric].mean(skipna=True)

# Count and report NaN separately
nan_count = scores_df[metric].isna().sum()
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Evaluation | RAGAS 0.1.21 | RAG-specific metrics |
| Validation | Pydantic v1 | Fail-fast data validation |
| LLM | Groq Llama-3.3-70B | Fast, free inference |
| Embeddings | all-MiniLM-L6-v2 | No OpenAI dependency |
| Vector DB | ChromaDB EphemeralClient | Fast in-memory evaluation |
| LLM wrapper | LangchainLLMWrapper | Groq → RAGAS compatibility |

## Setup

```bash
git clone <your-repo-url>
cd rag-evaluation-pipeline

python3 -m venv venv
source venv/bin/activate

pip install ragas==0.1.21 langchain-core==0.1.52 \
    langchain-community==0.0.38 langchain-groq==0.1.3 \
    sentence-transformers==2.7.0 chromadb==0.4.24 \
    groq==0.9.0 python-dotenv==1.0.1 pypdf==4.2.0 \
    datasets==2.19.0 pydantic==1.10.21

echo "GROQ_API_KEY=your_key_here" > .env

python3 run_evaluation.py
```

## Production Extensions

- **CI/CD integration** — GitHub Actions workflow blocks PRs if metrics drop below threshold
- **Regression testing** — compare scores against baseline on every change
- **Larger test datasets** — 500+ questions covering edge cases and adversarial inputs
- **Additional metrics** — noise robustness, answer correctness, semantic similarity
- **Latency tracking** — quality metrics alongside performance metrics
- **Human evaluation layer** — automated scores as first filter, human reviewer for final validation

## Why This Matters For Production

Deploying a RAG system without evaluation is like deploying software without tests. RAGAS provides the equivalent of a test suite for AI systems — objective, reproducible evidence of quality that satisfies model risk validation requirements.

## Skills Demonstrated

- RAGAS evaluation framework — 4-metric RAG quality assessment
- Pydantic data validation — fail-fast validation at dataset boundary
- LLMOps pipeline design — automated quality gates before deployment
- NaN handling — distinguishing infrastructure issues from quality issues
- Adapter pattern — LangchainLLMWrapper, LangchainEmbeddingsWrapper
- Audit trail — CSV export for model risk review
- Version management — resolving complex multi-framework dependency conflicts
