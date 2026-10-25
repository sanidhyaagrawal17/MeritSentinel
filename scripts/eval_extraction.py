import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from resumeproof.core.llm import FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume

personas = ["sam_okafor", "alex_rivera", "priya_nair"]
client = FakeLLMClient()
STOPWORDS = {"and", "with", "for", "the", "a", "an", "in", "on", "at", "to", "of", "by", "from", "as", "is", "using", "built", "implemented"}

def match_claims(expected_list, extracted_claims):
    # We want 1-to-1 bipartite matching so TP + FP = extracted and TP + FN = expected
    # Calculate similarity score between each (exp_idx, claim_idx)
    pairs = []
    for e_idx, exp in enumerate(expected_list):
        e_text = exp["text"].lower().strip()
        e_words = set(re.findall(r"[a-zA-Z0-9]+", e_text)) - STOPWORDS
        for c_idx, c in enumerate(extracted_claims):
            c_text = c.text.lower().strip()
            c_words = set(re.findall(r"[a-zA-Z0-9]+", c_text)) - STOPWORDS
            
            score = 0.0
            if e_text == c_text:
                score = 1.0
            elif e_text in c_text or c_text in e_text:
                score = 0.9
            else:
                overlap = len(e_words.intersection(c_words))
                union = len(e_words.union(c_words))
                jaccard = overlap / union if union > 0 else 0
                if overlap >= max(1, len(e_words) * 0.5):
                    score = 0.5 + 0.4 * jaccard
            if score >= 0.5:
                pairs.append((score, e_idx, c_idx))
                
    # Greedy best match
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_e = set()
    matched_c = set()
    matches = []
    for score, e_idx, c_idx in pairs:
        if e_idx not in matched_e and c_idx not in matched_c:
            matched_e.add(e_idx)
            matched_c.add(c_idx)
            matches.append((e_idx, c_idx, score))
            
    tp = len(matches)
    fp = len(extracted_claims) - tp
    fn = len(expected_list) - tp
    
    fp_claims = [extracted_claims[i] for i in range(len(extracted_claims)) if i not in matched_c]
    fn_expected = [expected_list[i] for i in range(len(expected_list)) if i not in matched_e]
    
    return {
        "expected": len(expected_list),
        "extracted": len(extracted_claims),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_claims": [c.text for c in fp_claims],
        "fn_expected": [e["text"] for e in fn_expected],
        "matches": [(expected_list[e]["text"], extracted_claims[c].text) for e, c, _ in matches]
    }

for p in personas:
    pdf_path = Path(f"data/demo/{p}/resume.pdf")
    parsed = parse_resume(pdf_path)
    facts, claims = extract_from_resume(parsed, client, persona_id=p, user_id=f"user_{p}")
    key_file = Path(f"data/demo/{p}/expected_claims.json")
    ground_truth = json.loads(key_file.read_text(encoding="utf-8"))
    
    res = match_claims(ground_truth["expected_claims"], claims)
    print(f"=== Persona: {p} ===")
    print(f"Extracted: {res['extracted']}, Expected: {res['expected']}, TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}")
    print(f"Check: TP + FP = {res['tp'] + res['fp']} (== {res['extracted']})")
    print(f"Check: TP + FN = {res['tp'] + res['fn']} (== {res['expected']})")
    print(f"Precision: {res['tp'] / (res['tp'] + res['fp']):.4f}, Recall: {res['tp'] / (res['tp'] + res['fn']):.4f}")
    print("FP Claims:")
    for c in res['fp_claims']:
        print(f"  - '{c}'")
    print("FN Expected:")
    for e in res['fn_expected']:
        print(f"  - '{e}'")
    print()
