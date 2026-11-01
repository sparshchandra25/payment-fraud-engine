import os
import json
import sqlite3
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from groq import Groq

from predict_fraud import score_transaction
from policy_rag import retrieve_policy

class DisputeVerdict(BaseModel):
    action: Literal["APPROVE_REFUND", "ESCROW_HOLD", "BLOCK_ACCOUNT"] = Field(
        description="Mandatory resolution code based on fraud score and RBI/NPCI policy."
    )
    cited_clause: str = Field(description="Exact RBI or NPCI clause_id used for the ruling.")
    customer_liability_inr: float = Field(description="Amount customer is liable for in INR (0.0 if zero liability).")
    reasoning: str = Field(description="1-sentence explanation citing drain ratio and reporting timeline.")

class AgentState(TypedDict):
    tx_id: str
    customer_claim: str
    days_to_report: int
    tx_record: dict
    ml_result: dict
    matched_policies: list[dict]
    final_verdict: dict
    llm_mode: str

def init_sqlite_ledger() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""
        CREATE TABLE transactions (
            tx_id TEXT PRIMARY KEY, amount REAL, old_bal REAL, new_bal REAL, velocity_1h INTEGER
        )
    """)
    conn.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)", [
        ("TXN_901", 48500.0, 49000.0, 0.0, 9),
        ("TXN_902", 2500.0, 45000.0, 42500.0, 2),
        ("TXN_903", 82000.0, 85000.0, 0.0, 14),
        ("TXN_904", 1200.0, 18000.0, 16800.0, 1)
    ])
    conn.commit()
    return conn

DB_CONN = init_sqlite_ledger()

def insert_live_transaction(tx_id: str, amount: float, old_bal: float, new_bal: float, velocity_1h: int):
    DB_CONN.execute(
        "INSERT OR REPLACE INTO transactions VALUES (?, ?, ?, ?, ?)",
        (tx_id, amount, old_bal, new_bal, velocity_1h)
    )
    DB_CONN.commit()

def get_all_transactions() -> list[dict]:
    rows = DB_CONN.execute("SELECT tx_id, amount, old_bal, new_bal, velocity_1h FROM transactions ORDER BY tx_id DESC").fetchall()
    return [{"tx_id": r[0], "amount": r[1], "old_bal": r[2], "new_bal": r[3], "velocity_1h": r[4]} for r in rows]

def investigate_node(state: AgentState) -> dict:
    row = DB_CONN.execute(
        "SELECT amount, old_bal, new_bal, velocity_1h FROM transactions WHERE tx_id=?",
        (state["tx_id"],)
    ).fetchone()
    tx_data = {"amount": row[0], "old_bal": row[1], "new_bal": row[2], "velocity_1h": row[3]}
    ml_out = score_transaction(tx_data["amount"], tx_data["old_bal"], tx_data["new_bal"], tx_data["velocity_1h"])
    return {"tx_record": tx_data, "ml_result": ml_out}

def retrieve_node(state: AgentState) -> dict:
    ml = state["ml_result"]
    risk_tag = "high velocity account draining mule transfer escrow freeze" if ml["is_flagged"] else "normal low-risk timeout auto-reversal"
    liability_tag = "notifies within 3 working days zero liability" if state["days_to_report"] <= 3 else "delayed reporting negligence"
    enriched_query = f"{state['customer_claim']} {risk_tag} {liability_tag}"
    policies = retrieve_policy(enriched_query, top_k=3)
    return {"matched_policies": policies}

def verdict_node(state: AgentState) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    ml = state["ml_result"]
    policies = state["matched_policies"]

    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""You are an RBI & NPCI Compliance Arbitrator. Return ONLY valid JSON matching this exact schema:
            {json.dumps(DisputeVerdict.model_json_schema())}
            
            Rules:
            - action MUST be one of: "APPROVE_REFUND", "ESCROW_HOLD", "BLOCK_ACCOUNT"
            - If ml_result.is_flagged is True and drain_ratio > 0.90 with velocity >= 12, use "BLOCK_ACCOUNT", else if is_flagged is True use "ESCROW_HOLD", else "APPROVE_REFUND".
            - If days_to_report <= 3, customer_liability_inr MUST be 0.0 under RBI-2017-18/15-Sec-8.
            
            Transaction Data: {state['tx_record']}
            ML Fraud Output: {ml}
            Customer Claim: {state['customer_claim']} (Reported in {state['days_to_report']} days)
            Retrieved RBI/NPCI Clauses: {policies}"""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_json = json.loads(resp.choices[0].message.content)
            validated = DisputeVerdict(**raw_json).model_dump()
            return {"final_verdict": validated, "llm_mode": "Llama-3.3-70B (Groq Live)"}
        except Exception:
            pass

    action = "BLOCK_ACCOUNT" if (ml["is_flagged"] and ml["drain_ratio"] > 0.90 and state["tx_record"]["velocity_1h"] >= 12) else ("ESCROW_HOLD" if ml["is_flagged"] else "APPROVE_REFUND")
    validated = DisputeVerdict(
        action=action,
        cited_clause=policies[0]["clause_id"],
        customer_liability_inr=0.0 if state["days_to_report"] <= 3 else 5000.0,
        reasoning=f"ML risk={ml['risk_score']} (drain_ratio={ml['drain_ratio']}, balance_error={ml['balance_error']}); reported in {state['days_to_report']} days under {policies[0]['clause_id']}."
    ).model_dump()
    return {"final_verdict": validated, "llm_mode": "Deterministic Rule Engine (Local M5)"}

builder = StateGraph(AgentState)
builder.add_node("investigate", investigate_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("verdict", verdict_node)
builder.set_entry_point("investigate")
builder.add_edge("investigate", "retrieve")
builder.add_edge("retrieve", "verdict")
builder.add_edge("verdict", END)
DISPUTE_AGENT = builder.compile()