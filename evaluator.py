# ── EVALUATOR ───────────────────────────────────────────
# Connects RAG system to RAGAS evaluation framework
# Runs each test question through RAG pipeline
# Collects: question, answer, retrieved chunks, ground truth
# Passes everything to RAGAS for scoring

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()

# ── MODELS ─────────────────────────────────────────────
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Models loaded successfully")


# ── INDEX PDF ──────────────────────────────────────────
def build_index(pdf_path: str) -> chromadb.Collection:
    """
    Load PDF, chunk it, embed chunks, store in ChromaDB.
    Returns the collection for querying.
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    texts = [chunk.page_content for chunk in chunks]
    print(f"Created {len(texts)} chunks")

    # Validate chunks not empty
    if not texts:
        raise ValueError(
            f"No text extracted from {pdf_path}. "
            "Ensure PDF contains actual text not scanned images."
        )

    # Embed chunks
    embeddings = embedding_model.encode(texts)
    print(f"Created {len(embeddings)} embeddings")

    # Store in ChromaDB
    chroma_client = chromadb.EphemeralClient()
    # EphemeralClient = in-memory only
    # No disk persistence needed for evaluation
    # Faster than PersistentClient for testing

    try:
        chroma_client.delete_collection("eval_chunks")
    except:
        pass

    collection = chroma_client.create_collection("eval_chunks")
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(texts))]
    )
    print(f"Stored {collection.count()} chunks in memory")
    return collection


# ── RETRIEVE CHUNKS ────────────────────────────────────
def retrieve_chunks(
    question: str,
    collection: chromadb.Collection,
    n_results: int = 3
) -> list[str]:
    """
    Embed question and retrieve top n relevant chunks.
    Returns list of relevant text chunks.
    """
    question_embedding = embedding_model.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )
    chunks = results['documents'][0]
    return chunks


# ── GET ANSWER ─────────────────────────────────────────
def get_answer(
    question: str,
    context_chunks: list[str]
) -> str:
    """
    Call Groq LLM with question and retrieved context.
    Returns the LLM answer.
    """
    context = "\n\n".join(context_chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the provided context. "
                    "If the answer is not in the context, "
                    "say I don't know. "
                    "Be concise and accurate."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )
    return response.choices[0].message.content


# ── RUN EVALUATION PIPELINE ────────────────────────────
def run_rag_pipeline(
    test_cases: list,
    pdf_path: str
) -> list[dict]:
    """
    Run every test case through the RAG pipeline.
    Returns list of results ready for RAGAS evaluation.

    Each result contains:
      question        → the original question
      answer          → what RAG answered
      contexts        → chunks retrieved from PDF
      ground_truth    → the correct answer
    """
    print(f"\nBuilding index from {pdf_path}...")
    collection = build_index(pdf_path)

    results = []
    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, case in enumerate(test_cases, 1):
        question     = case.question
        ground_truth = case.ground_truth

        print(f"[{i}/{len(test_cases)}] {question}")

        # Retrieve relevant chunks
        contexts = retrieve_chunks(question, collection)

        # Get answer from LLM
        answer = get_answer(question, contexts)

        print(f"  Answer: {answer[:80]}...")

        results.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth
        })

    print(f"\nPipeline complete — {len(results)} results ready for RAGAS")
    return results


# ── QUICK TEST ─────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with one question
    print("Quick evaluator test...")
    collection = build_index("company_policy.pdf")
    chunks = retrieve_chunks(
        "What is the refund policy?",
        collection
    )
    answer = get_answer("What is the refund policy?", chunks)
    print(f"\nTest question: What is the refund policy?")
    print(f"Retrieved chunks: {len(chunks)}")
    print(f"Answer: {answer}")