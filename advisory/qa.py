import re

OFFTOPIC = re.compile(
    r"\b(tomorrow|forecast|today|tonight|price|prices|cost|sell|selling|market|rate|loan|subsidy|"
    r"insurance|prime minister|capital|news|joke|cricket|election)\b", re.I)

SECTIONS = [
    ("urea",     r"\b(urea|fertili[sz]\w*|nitrogen|manure|potash|dap)\b", ["actions:1", "prevention:2"]),
    ("organic",  r"\b(organic|natural\w*|neem|bio[- ]?\w*|cow ?dung|home ?made)\b", ["organic"]),
    ("prevent",  r"\b(prevent\w*|avoid\w*|next (?:year|season|crop)|come back|future|protect\w*)\b", ["prevention"]),
    ("symptoms", r"\b(symptoms?|signs?|look(?:s)? like|spots?|lesions?|identify|brown|grey|gray|yellow\w*)\b", ["symptoms"]),
    ("weather",  r"\b(weather|rain\w*|cloud\w*|humid\w*|dew|damp|why (?:did|does|is|has))\b", ["favourable"]),
    ("escalate", r"\b(officer|office|kvk|expert|specialist|agronomist|contact|visit|call (?:someone|somebody)|serious|worse)\b", ["escalate"]),
    ("treat",    r"\b(what (?:should|can|do) i do|what to do|treat\w*|control\w*|manage|cure|remedy|stop (?:it|this|the)|burn|remove|pull|destroy|help|solution)\b", ["actions"]),
]

NOT_COVERED_MSG = {
    "en": "I do not have a verified answer for that. Please ask your KVK or agriculture officer.",
    "hi": "इसका सत्यापित उत्तर मेरे पास नहीं है। कृपया अपने केवीके या कृषि अधिकारी से पूछें।",
    "bn": "এর জন্য আমার কাছে যাচাই করা উত্তর নেই। অনুগ্রহ করে আপনার কেভিকে বা কৃষি আধিকারিকের কাছে জিজ্ঞাসা করুন।",
    "as": "ইয়াৰ বাবে মোৰ ওচৰত যাচাই কৰা উত্তৰ নাই। অনুগ্ৰহ কৰি আপোনাৰ কে.ভি.কে. বা কৃষি বিষয়াক সোধক।",
}

SEV = ["mild", "moderate", "severe"]


def _l(field, lang):
    return field.get(lang) or field["en"]


def _join(x):
    return " ".join(x) if isinstance(x, list) else x


def find_sections(question):
    if OFFTOPIC.search(question):
        return None, []
    for name, pat, keys in SECTIONS:
        if re.search(pat, question, re.I):
            return name, keys
    return None, []


def section_text(card, key, lang, severity):
    if key == "favourable":
        return _join(_l(card["favourable_conditions"], lang))
    if key == "symptoms":
        return _join(_l(card["symptoms"], lang))
    if key == "escalate":
        return _join(_l(card["escalate_when"], lang))
    if key == "organic":
        return _join(_l(card["organic_options"], lang))
    if key == "prevention":
        return _join(_l(card["prevention"], lang))
    if key == "prevention:2":
        return _l(card["prevention"], lang)[2]
    if key == "actions:1":
        return _l(card["actions_by_severity"]["mild"], lang)[1]
    if key == "actions":
        out = []
        for s in SEV[: SEV.index(severity) + 1]:
            out += _l(card["actions_by_severity"][s], lang)
        return " ".join(out)
    raise KeyError(key)
