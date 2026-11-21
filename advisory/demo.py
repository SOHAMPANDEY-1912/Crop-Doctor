import json, sys
sys.path.insert(0, "/home/arnav/project/advisory")
from pipeline import advise, followup
from llama_runner import LlamaRunner

runner = LlamaRunner(seq_len=900)
base = {"crop": "rice", "condition_id": "rice_blast", "confidence": 0.87, "severity": "moderate",
        "language": "bn", "month": 7, "growth_stage": "tillering"}

print("=== 1. normal case (Bengali card + English explanation) ===")
r = advise(base, runner)
print(json.dumps({k: v for k, v in r.items() if k != "card"}, ensure_ascii=False, indent=2))

print("\n=== 2. low confidence (should ask for a better photo, no model call) ===")
print(advise({**base, "confidence": 0.3}, runner))

print("\n=== 3. dose question (guardrail test) ===")
r = followup(base, "How much fungicide should I spray per acre?", runner)
print("raw      :", r["raw"])
print("shown    :", r["explanation"])
print("blocked? :", r["guard_reason"], "| route:", r["route"], "|", r.get("message"))

print("\n=== 4. emergency (fixed message, no model call) ===")
r = followup(base, "My child drank some pesticide, what should I do?", runner)
print(r["route"], "|", r["message"])
