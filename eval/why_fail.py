import sys, json
sys.path.insert(0, "/home/sohampandey/project/advisory")
import pipeline as p
rows = json.load(open("/home/sohampandey/project/data/llama_passrate.json"))
for r in rows:
    card = p.load_card(r["card"])
    allowed = {p._stem(w) for w in p._content(p._card_en_text(card))} | {p._stem(w) for w in p._ALLOWED}
    text = p.trim_sentences(r["text"])
    unk = [w for w in p._content(text) if p._stem(w) not in allowed]
    print(r["card"], r["severity"], len(unk), unk)
