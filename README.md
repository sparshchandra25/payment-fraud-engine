# Real-Time Payment Fraud & Autonomous Dispute Resolution Engine

<img width="1280" height="849" alt="PHOTO-2026-11-01-20-06-59" src="https://github.com/user-attachments/assets/a05f29d0-ca61-4f4a-89b5-9c50f26851d4" />

An end-to-end payment risk and dispute resolution pipeline combining **LightGBM + Isolation Forest**, **Hybrid RAG (BM25 + Dense Vector Search)**, and a **3-Node LangGraph Agent**, served through a **FastAPI** web console.

Built and benchmarked locally on Apple Silicon (MacBook Air M5 Metal GPU).

---

## System Architecture (5-File Pipeline)

Instead of sending every raw transaction straight to an LLM (which is slow and expensive for high-volume UPI traffic), this system uses a two-stage funnel: a fast tabular ML model scores transactions in milliseconds and triggers the RAG + LangGraph investigation agent only when a dispute or high-risk flag occurs.

1. **`predict_fraud.py` — Sub-30ms Tabular Fraud Gate**
   * Engineers domain-specific payment features:
     * **Account Drain Ratio:** $\text{Drain Ratio} = \frac{\text{Amount}}{\text{Old Balance} + 1.0}$
     * **Ledger Balance Error:** $\text{Balance Error} = |(\text{Old Balance} - \text{Amount}) - \text{New Balance}|$
   * Combines a supervised **LightGBM** classifier ($85\%$ weight) with an unsupervised **Isolation Forest** outlier detector ($15\%$ weight) to flag account-draining mule transfers.

2. **`policy_rag.py` — Hybrid Regulatory RAG Engine**
   * Indexes official **RBI Zero-Liability** (`RBI-2017-18/15`) and **NPCI UPI Chargeback** (`NPCI-UPI-OC-115-UDIR`) clauses.
   * Combines **BM25** sparse keyword scoring (`rank-bm25`) with dense vector embeddings (`all-MiniLM-L6-v2` running on Apple Metal `mps`) using a $50/50$ hybrid fusion score.

3. **`dispute_agent.py` — 3-Node LangGraph Workflow & SQLite Ledger**
   * **Node 1 (`investigate`):** Fetches transaction history from a thread-safe in-memory **SQLite** ledger and runs the LightGBM/Isolation Forest fraud scorer.
   * **Node 2 (`retrieve`):** Performs state-aware query enrichment based on fraud risk and reporting delay ($T+N$ days) to retrieve matching RBI/NPCI clauses.
   * **Node 3 (`verdict`):** Routes context to **Llama-3.3-70B** (via Groq API) or a deterministic local fallback engine, enforcing strict **Pydantic** JSON schema validation (`APPROVE_REFUND`, `ESCROW_HOLD`, or `BLOCK_ACCOUNT`).

4. **`app.py` — FastAPI Backend & Interactive Web Console**
   * Serves REST endpoints (`/api/adjudicate`, `/api/ledger`, `/api/benchmark`) and a 3-column FinTech operator dashboard with 1-click recruiter test presets.

5. **`eval_suite.py` — 50-Case Automated Evaluation & Caching Suite**
   * Runs 50 synthetic dispute scenarios against RBI policy rules and uses an $O(1)$ MD5 hash-table cache to eliminate redundant pipeline executions.

---

## Benchmark Telemetry (Apple M5 MPS)

| Metric | Measured Result |
| :--- | :--- |
| **Test Set Fraud PR-AUC** | `0.94+` (Synthetic UPI/Mobile-Money Stream) |
| **50-Case Policy Decision Accuracy** | `50 / 50` (`100%` Schema & Liability Compliance) |
| **Cache Hit Ratio (Burst Traffic)** | `40 / 50` (`80%` Hits) |
| **Mean Uncached Pipeline Latency** | `11.73 ms` |
| **Mean Cached Pipeline Latency** | `2.35 ms` (`80.0%` Latency Reduction) |

---

## Quickstart (Run Locally)

### 1. Clone & Install Dependencies (via `uv`)
```bash
git clone https://github.com/YOUR_USERNAME/payment-fraud-engine.git
cd payment-fraud-engine
uv sync
```

### 2. Run Individual Pipeline Modules
```bash
uv run predict_fraud.py
uv run policy_rag.py
uv run dispute_agent.py
uv run eval_suite.py
```

### 3. Launch the Live Web Dashboard
```bash
uv run uvicorn app:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser to test live transactions, inspect retrieved RBI circulars, and run the 50-case benchmark suite.
