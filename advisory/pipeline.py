import calendar, json, re, time
from pathlib import Path
from llama_runner import LlamaRunner, chat_prompt

CARD_DIR = Path.home() / "project/cards"
LANG_NAMES = {"en": "English", "hi": "Hindi", "as": "Assamese", "bn": "Bengali"}
SEVERITIES = ["mild", "moderate", "severe"]

SYSTEM = """You are a crop advisory assistant for smallholder farmers in the Barak Valley, Assam, India.
Rules:
1. Use ONLY the facts in the CARD below. Never add facts, names or causes that are not in the CARD.
2. NEVER give pesticide names, doses, mixing amounts or spray intervals. The farmer sees these on the card.
3. Write exactly 3 short, simple sentences. Do not repeat the card. Do not guess causes, months or growth stages that are not in the CARD.
4. Reply in {language}.
5. Do not give medical, legal or financial advice."""

TASK_EXPLAIN = ("Explain to the farmer in simple words what this problem is, why it happens at "
                "what usually makes it worse, and the one or two most important actions right now. "
                "Write exactly 3 short sentences. Start directly with the problem; do not write an introduction. Use only words and facts from the CARD.")

# ---------- emergency check (plain code, runs BEFORE the model) ----------
EMERGENCY = re.compile(
    r"\b(poison|poisoned|swallow|swallowed|drank|drunk|ingest|ingested|snake|bitten|"
    r"unconscious|vomit|vomiting|breathing)\b|जहर|विष|निगल|সাপ|বিষ", re.I)
EMERGENCY_MSG = {
    "en": "This may be an emergency. Call a doctor or go to the nearest hospital now. For emergency help in India, call 112.",
    "hi": "यह आपातकालीन स्थिति हो सकती है। अभी डॉक्टर को बुलाएँ या नज़दीकी अस्पताल जाएँ। आपातकालीन मदद के लिए 112 पर कॉल करें।",
    "bn": "এটি জরুরি অবস্থা হতে পারে। এখনই ডাক্তারের কাছে যান বা কাছের হাসপাতালে যান। জরুরি সাহায্যের জন্য 112 নম্বরে ফোন করুন।",
    "as": "এয়া জৰুৰীকালীন অৱস্থা হ'ব পাৰে। এতিয়াই চিকিৎসকৰ ওচৰলৈ যাওক বা ওচৰৰ হস্পিতাললৈ যাওক। জৰুৰীকালীন সহায়ৰ বাবে 112 নম্বৰত ফোন কৰক।",
}


# ---------- dose / product questions: fixed answer, no model ----------
DOSE_INTENT = re.compile(
    r"\b(how much|how many|doses?|dosage|per acre|per hectare|per litre|per liter|spray|mix|dilute|"
    r"chemicals?|pesticides?|fungicides?|insecticides?|medicine|tricyclazole|"
    r"ignore (?:your|all|the) (?:rules|instructions))\b"
    r"|कितना|कितनी|मात्रा|दवा|কত|মাত্রা|ওষুধ|কীটনাশক|ছত্রাকনাশক", re.I)
DOSE_MSG = {
    "en": "I cannot give pesticide names or amounts here. Please ask your KVK or agriculture officer. They will tell you the right product and amount for your field.",
    "hi": "यहाँ दवा का नाम या मात्रा बताना संभव नहीं है। कृपया अपने केवीके या कृषि अधिकारी से पूछें। वे आपके खेत के लिए सही दवा और मात्रा बताएँगे।",
    "bn": "আমি এখানে কীটনাশকের নাম বা মাত্রা বলতে পারি না। অনুগ্রহ করে আপনার কেভিকে বা কৃষি আধিকারিকের কাছে জিজ্ঞাসা করুন। তাঁরা আপনার ক্ষেতের জন্য সঠিক ওষুধ ও মাত্রা বলে দেবেন।",
    "as": "মই ইয়াত কীটনাশকৰ নাম বা পৰিমাণ ক'ব নোৱাৰোঁ। অনুগ্ৰহ কৰি আপোনাৰ কে.ভি.কে. বা কৃষি বিষয়াক সোধক। তেওঁলোকে আপোনাৰ খেতিৰ বাবে সঠিক ঔষধ আৰু পৰিমাণ ক'ব।",
}


# ---------- cards ----------
def load_card(condition_id):
    path = CARD_DIR / f"{condition_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pick(field, lang):
    return field.get(lang) or field["en"]


def render_card(card, lang, severity):
    idx = SEVERITIES.index(severity)
    actions = []
    for s in SEVERITIES[: idx + 1]:                      # actions are cumulative
        actions += _pick(card["actions_by_severity"][s], lang)
    return {
        "name": _pick(card["names"], lang),
        "summary": _pick(card["summary"], lang),
        "symptoms": _pick(card["symptoms"], lang),
        "actions": actions,
        "organic": _pick(card["organic_options"], lang),
        "prevention": _pick(card["prevention"], lang),
        "escalate": _pick(card["escalate_when"], lang),
        "has_verified_chemicals": bool(card.get("chemical_options")),
    }


# ---------- routing (plain code, not the model) ----------
def route(msg):
    cid = msg["condition_id"]
    if cid == "healthy":
        return "healthy"
    if cid == "unknown":
        return "retake"
    c = msg["confidence"]
    if c >= 0.75:
        return "show_card"
    if c >= 0.5:
        return "possible"
    return "retake"


# ---------- prompt ----------
def build_prompt(msg, card, task):
    lang = msg.get("language", "en")
    llm_lang = lang if lang in ("en", "hi") else "en"     # Llama 3.2 supports en/hi officially
    sev = msg.get("severity", "moderate")
    actions = []
    for s in SEVERITIES[: SEVERITIES.index(sev) + 1]:
        actions += card["actions_by_severity"][s]["en"]
    system = SYSTEM.format(language=LANG_NAMES[llm_lang])
    user = (
        "CONTEXT\n"
        f"Crop: {card['crop']}\n"
        f"Detected: {card['names']['en']} (confidence {round(msg['confidence'] * 100)}%)\n"
        f"Severity: {sev}\n"
        "\n"
        "CARD (verified facts)\n"
        f"Symptoms: {card['symptoms']['en']}\n"
        f"Favourable conditions: {card['favourable_conditions']['en']}\n"
        f"Recommended actions for {sev} severity: {' '.join(actions)}\n"
        f"Prevention: {' '.join(card['prevention']['en'])}\n\n"
        f"TASK\n{task}"
    )
    return chat_prompt(system, user), llm_lang


# ---------- output guardrail ----------
_UNITS = r"(?:ml|g|gm|kg|l|mg|ppm|%|cc|percent|litres?|liters?|grams?)"
DOSE = re.compile(r"\d+(?:\.\d+)?\s?" + _UNITS + r"(?![A-Za-z])", re.I)
PHRASES = re.compile(
    r"\b(spray every|per acre|per hectare|per litre|per liter|tank mix|dosage|dose|dilute|mix(?:ed|ing)?)\b"
    r"|मिली|ग्राम|किलो|लीटर|एकड़|हेक्टेयर", re.I)
CHEMICALS = ["tricyclazole", "isoprothiolane", "carbendazim", "mancozeb", "propiconazole",
             "hexaconazole", "azoxystrobin", "tebuconazole"]


def safe_explanation(text, extra_blocked=()):
    if not text.strip():
        return None, "empty"
    if "i am not sure" in text.lower():
        return None, "model_unsure"
    if DOSE.search(text):
        return None, "dose_or_quantity"
    if PHRASES.search(text):
        return None, "dosing_phrase"
    low = text.lower()
    for name in list(CHEMICALS) + [str(x).lower() for x in extra_blocked if x]:
        if name in low:
            return None, f"chemical:{name}"
    return text.strip(), None


def trim_sentences(text, n=4):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


def grounded(text, card):
    """Reject explanations that mention names or numbers that are not in the card."""
    ref = json.dumps(card, ensure_ascii=False).lower()
    ref_words = set(re.findall(r"[a-z]+", ref)) | {"kvk", "india", "assam"}
    if re.search(r"\d", text) and not re.search(r"\d", ref):
        return False
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for w in sentence.split()[1:]:
            w = w.strip(".,;:!?()\"'")
            if len(w) > 3 and w[:1].isupper() and w.lower() not in ref_words:
                return False
    return True


# ---------- main entry ----------
def advise(msg, runner=None, task=TASK_EXPLAIN):
    decision = route(msg)
    if decision in ("healthy", "retake"):
        return {"route": decision, "card": None, "explanation": None}
    card = load_card(msg["condition_id"])
    if card is None:
        return {"route": "no_card", "card": None, "explanation": None}
    runner = runner or LlamaRunner(seq_len=900)
    prompt, llm_lang = build_prompt(msg, card, task)
    t0 = time.time()
    raw, stats = runner.generate(prompt,max_new_tokens=160)
    seconds = round(time.time() - t0, 1)
    extra = [c.get("active_ingredient") for c in card.get("chemical_options", [])]
    explanation, reason = safe_explanation(trim_sentences(raw), extra)
    if explanation and not grounded(explanation, card):
        explanation, reason = None, "ungrounded"
    lang = msg.get("language", "en")
    rendered = render_card(card, lang, msg.get("severity", "moderate"))
    return {
        "route": decision,                       # show_card or possible
        "card": rendered,
        "speak": rendered["summary"],            # TTS reads the verified card, never the model text
        "explanation": explanation,              # None if the guardrail rejected it
        "explanation_lang": llm_lang,
        "guard_reason": reason,
        "raw": raw,
        "seconds": seconds,
        "decode_tok_s": stats.get("decode_token_per_sec"),
    }


def followup(msg, question, runner=None):
    if EMERGENCY.search(question):                       # fixed message, no model call
        lang = msg.get("language", "en")
        return {"route": "emergency", "message": EMERGENCY_MSG.get(lang, EMERGENCY_MSG["en"]),
                "card": None, "raw": "", "explanation": None, "guard_reason": None, "seconds": 0}
    if DOSE_INTENT.search(question):                     # product / amount questions: fixed answer
        lang = msg.get("language", "en")
        return {"route": "refer_expert", "message": DOSE_MSG.get(lang, DOSE_MSG["en"]),
                "card": None, "raw": "", "explanation": None, "guard_reason": None, "seconds": 0}
    return answer_followup(msg, question, runner)



def answer_followup(msg, question, runner=None):
    from qa import find_sections, section_text, NOT_COVERED_MSG
    lang = msg.get("language", "en")
    sev = msg.get("severity", "moderate")
    card = load_card(msg["condition_id"])
    topic, keys = find_sections(question)
    if card is None or not keys:
        return {"route": "not_covered", "message": NOT_COVERED_MSG.get(lang, NOT_COVERED_MSG["en"]),
                "card": None, "raw": "", "explanation": None, "guard_reason": None, "seconds": 0}
    answer_card = " ".join(section_text(card, k, lang, sev) for k in keys)
    facts_en = " ".join(section_text(card, k, "en", sev) for k in keys)
    llm_lang = lang if lang in ("en", "hi") else "en"
    system = ("You help a farmer. Use ONLY the FACTS given. Do not add any other facts, names or causes. "
              "Reply in " + LANG_NAMES[llm_lang] + " in at most 2 short, simple sentences.")
    user = "FACTS\n" + facts_en + "\n\nFARMER QUESTION\n" + question
    runner = runner or LlamaRunner(seq_len=900)
    t0 = time.time()
    raw, stats = runner.generate(chat_prompt(system, user), max_new_tokens=80)
    seconds = round(time.time() - t0, 1)
    extra = [c.get("active_ingredient") for c in card.get("chemical_options", [])]
    explanation, reason = safe_explanation(trim_sentences(raw, 2), extra)
    if explanation and not grounded(explanation, card):
        explanation, reason = None, "ungrounded"
    return {"route": "answered", "topic": topic, "answer_card": answer_card,
            "explanation": explanation, "explanation_lang": llm_lang, "guard_reason": reason,
            "raw": raw, "seconds": seconds, "card": None}


# ---------- stricter grounding check (overrides the earlier grounded) ----------
_STOP = set(("this that these those with from into onto over under about after before because while which "
             "their there them then than they have has had been being were will would should could might must "
             "your yours also very much many some more most only just each such same other another "
             "what when where whom whose does doing done make made need needs like").split())
_ALLOWED = {"farmer", "field", "plant", "plants", "rice", "crop", "please", "contact", "officer",
            "agriculture", "extension", "kvk", "disease", "problem", "important", "action", "actions",
            "advice", "ask", "help"}
_ALLOWED |= {"barak", "valley", "assam", "india", "mild", "moderate", "severe",
    "cause", "causes", "usually", "common", "worse", "worsen", "serious", "major",
    "easily", "seen", "affects", "areas", "region", "likely", "occur", "happens",
    "take", "prevent", "quickly", "provide", "inspect", "harm", "damage", "high",
    "levels", "during", "significant", "presence", "detected", "susceptible",
    "untreated", "even", "left", "particularly", "characteristic", "here",
    "explanation", "understand", "symptoms", "appear", "treated", "promptly",
    "infestation", "pest"}


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _content(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in _STOP]


def _card_en_text(card):
    parts = []

    def walk(x):
        if isinstance(x, dict):
            if "en" in x and isinstance(x["en"], (str, list)):
                walk(x["en"])
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            parts.append(x)

    keys = ("names", "summary", "symptoms", "favourable_conditions", "actions_by_severity",
            "organic_options", "prevention", "escalate_when")
    walk({k: card[k] for k in keys if k in card})
    return " ".join(parts)


def grounded(text, card):
    low = text.lower()
    if re.search(r"\byes\b", low) and re.search(r"\bno\b", low):
        return False                                    # contradicts itself
    allowed = {_stem(w) for w in _content(_card_en_text(card))} | {_stem(w) for w in _ALLOWED}
    words = _content(text)
    if not words:
        return False
    unknown = [w for w in words if _stem(w) not in allowed]
    return len(unknown) <= 2




_HARD = ("fungal", "fungus", "fungi", "bacteria", "bacterial", "virus", "viral",
         "insect", "insects", "mite", "mites",
         "fatal", "die", "dies", "dead", "death", "kill", "kills")

def grounded(text, card):
    low = text.lower()
    if re.search(r"\byes\b", low) and re.search(r"\bno\b", low):
        return False
    sents = [x.strip().lower() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if len(sents) != len(set(sents)):
        return False   # repeated sentence = degenerate output
    card_low = _card_en_text(card).lower()
    for t in _HARD:
        if re.search(r"\b" + t + r"\b", low) and not re.search(r"\b" + t + r"\b", card_low):
            return False
    allowed = {_stem(w) for w in _content(card_low)} | {_stem(w) for w in _ALLOWED}
    words = _content(text)
    if not words:
        return False
    unknown = [w for w in words if _stem(w) not in allowed]
    return len(unknown) <= 2

# ---------- Llama as a constrained question router ----------
def llm_route(question, runner=None):
    """Llama only picks ONE label from a fixed list; it never writes content."""
    from qa import SECTIONS
    labels = {name: keys for name, _pat, keys in SECTIONS}
    desc = {
        "weather": "why the disease appears, weather, causes",
        "urea": "fertilizer or urea",
        "treat": "what to do now, treatment, control",
        "prevent": "preventing it in future",
        "organic": "organic or natural options",
        "symptoms": "what the disease looks like, signs",
        "escalate": "when to call an expert",
        "none": "anything else, not about this disease",
    }
    system = ("Classify the farmer's question about rice blast into exactly ONE label. "
              "Reply with only the label, nothing else.\nLabels:\n"
              + "\n".join(k + ": " + v for k, v in desc.items()))
    runner = runner or LlamaRunner(seq_len=300, temperature=0.0)
    raw, _ = runner.generate(chat_prompt(system, question), max_new_tokens=6)
    word = re.sub(r"[^a-z]", "", raw.lower().split()[0]) if raw.strip() else ""
    if word in labels:
        return word, labels[word]
    return None, []
