import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Official RBI & NPCI Dispute Circular Clauses (Knowledge Base)
RBI_NPCI_RULES = [
    {
        "clause_id": "RBI-2017-18/15-Sec-8",
        "title": "Zero Liability of Customer in Unauthorized Third-Party Breach",
        "text": "If an unauthorized electronic payment transaction occurs due to a third-party breach or mule account ring and the customer notifies the bank within 3 working days, customer liability is ZERO. The bank must credit the shadow reversal amount within 10 working days."
    },
    {
        "clause_id": "RBI-2017-18/15-Sec-9",
        "title": "Limited Liability due to Customer Negligence or Delayed Reporting",
        "text": "In cases where the loss is due to negligence by a customer (such as sharing UPI PIN, OTP, or clicking phishing collect links), the customer bears the loss until reporting it to the bank. If reported after 4 to 7 working days, maximum liability is capped between INR 5,000 and INR 25,000."
    },
    {
        "clause_id": "NPCI-UPI-OC-115-UDIR",
        "title": "UPI Dispute Redressal & Escrow Freeze for High-Velocity Mule Transfers",
        "text": "When real-time fraud scoring flags an account-draining transfer (drain ratio > 0.80) with high hourly velocity or ledger balance discrepancy, the beneficiary account must be placed on immediate ESCROW_HOLD or BLOCK_ACCOUNT pending automated chargeback code U010."
    },
    {
        "clause_id": "NPCI-UPI-OC-142-TAT",
        "title": "Auto-Reversal for Failed or Timed-Out Beneficiary Credits (T+1 SLA)",
        "text": "For normal low-risk UPI transfers where the remitter account is debited but the beneficiary account is not credited due to technical timeout, auto-reversal must be completed within T+1 calendar day or attract INR 100 per day compensation."
    }
]

# Load lightweight embedding model onto Apple M5 Metal GPU ("mps")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
CORPUS_TEXTS = [f"{r['clause_id']} {r['title']} {r['text']}" for r in RBI_NPCI_RULES]
CORPUS_VECTORS = EMBEDDER.encode(CORPUS_TEXTS, normalize_embeddings=True)
BM25_INDEX = BM25Okapi([doc.lower().split() for doc in CORPUS_TEXTS])

def retrieve_policy(query: str, top_k: int = 2) -> list[dict]:
    # 1. Sparse Keyword Score (BM25 - catches exact codes like 'U010' or 'T+1')
    bm25_scores = np.array(BM25_INDEX.get_scores(query.lower().split()))
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-6)

    # 2. Dense Semantic Score (Dot-product cosine similarity on normalized vectors)
    query_vec = EMBEDDER.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(CORPUS_VECTORS, query_vec)

    # 3. Hybrid Fusion (50% Keyword + 50% Semantic Meaning)
    hybrid_scores = 0.5 * bm25_norm + 0.5 * dense_scores
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return [
        {**RBI_NPCI_RULES[idx], "relevance_score": round(float(hybrid_scores[idx]), 4)}
        for idx in top_indices
    ]

if __name__ == "__main__":
    test_query = "High velocity account draining scam reported within 2 days"
    matches = retrieve_policy(test_query)
    for m in matches:
        print(f"[{m['clause_id']}] (Score: {m['relevance_score']}) -> {m['title']}")