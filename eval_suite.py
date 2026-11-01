import time
import hashlib
from dispute_agent import DISPUTE_AGENT

# Hash table to store already-solved disputes (O(1) lookup)
CACHE = {}

def run_with_cache(case: dict):
    key = hashlib.md5(f"{case['tx_id']}-{case['days_to_report']}".encode()).hexdigest()
    if key in CACHE:
        return CACHE[key], True  # Cache Hit (instant)
    
    res = DISPUTE_AGENT.invoke(case)
    CACHE[key] = res
    return res, False            # Cache Miss (runs full ML + RAG + Agent)

if __name__ == "__main__":
    passed, hits = 0, 0
    uncached_ms, all_ms = [], []

    # Run 50 automated test disputes
    for i in range(50):
        is_scam = (i % 2 == 0)
        days = (i % 5) + 1
        case = {
            "tx_id": "TXN_901" if is_scam else "TXN_902",
            "customer_claim": "Account drained by scam" if is_scam else "Timeout debit",
            "days_to_report": days
        }

        t0 = time.perf_counter()
        result, is_hit = run_with_cache(case)
        ms = (time.perf_counter() - t0) * 1000

        all_ms.append(ms)
        if is_hit:
            hits += 1
        else:
            uncached_ms.append(ms)

        # Check if Agent made the right legal decision
        verdict = result["final_verdict"]
        expected_action = "ESCROW_HOLD" if is_scam else "APPROVE_REFUND"
        if verdict["action"] == expected_action:
            passed += 1

    avg_raw = sum(uncached_ms) / len(uncached_ms)
    avg_cached = sum(all_ms) / len(all_ms)
    drop = ((avg_raw - avg_cached) / avg_raw) * 100

    print(f"Accuracy         : {passed}/50 ({passed/50:.0%})")
    print(f"Cache Hits       : {hits}/50")
    print(f"Uncached Latency : {avg_raw:.2f} ms")
    print(f"Cached Latency   : {avg_cached:.2f} ms ({drop:.1f}% faster)")