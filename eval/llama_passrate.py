#!/usr/bin/env python3
"""Integration test for the advisory pipeline: ask Llama for an explanation for every advice card and severity,
and count how often the explanation passes the safety checks (otherwise the app hides it and shows only the card).
Usage: python llama_passrate.py          (English only: about 12 prompts for 4 cards)
       python llama_passrate.py hi       (also Hindi)
Results are saved to ~/project/data/llama_passrate.json for the report."""
import json, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.home() / "project/advisory"))
from pipeline import advise

langs = ["en"] + (["hi"] if "hi" in sys.argv[1:] else [])
cards = sorted(p.stem for p in (Path.home() / "project/cards").glob("*.json"))
rows = []
for cid in cards:
    for sev in ("mild", "moderate", "severe"):
        for lang in langs:
            msg = dict(crop=cid.split("_")[0], condition_id=cid, confidence=0.9, severity=sev, language=lang,
                       month=time.localtime().tm_mon, growth_stage="unknown")
            t0 = time.time()
            r = advise(msg)
            shown = bool(r.get("explanation"))
            rows.append(dict(card=cid, severity=sev, lang=lang, shown=shown, guard=r.get("guard_reason"),
                             seconds=r.get("seconds", round(time.time() - t0, 1)), decode_tok_s=r.get("decode_tok_s"),
                             text=r.get("explanation") or r.get("raw", "")))
            print(f"{cid:28s} {sev:9s} {lang}  shown={shown!s:5s} guard={r.get('guard_reason')}  {rows[-1]['seconds']}s")
n = len(rows)
ok = sum(r["shown"] for r in rows)
print(f"\nexplanations that passed the safety checks: {ok}/{n} ({100 * ok / max(n, 1):.0f}%)")
print("reasons for hiding:", dict(Counter(r["guard"] for r in rows if not r["shown"])))
secs = [r["seconds"] for r in rows if r["seconds"]]
if secs:
    print(f"time per explanation: median {sorted(secs)[len(secs) // 2]} s, max {max(secs)} s")
out = Path.home() / "project/data/llama_passrate.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", out, "- read the 'text' fields yourself: the guardrail cannot judge whether an explanation is true")
