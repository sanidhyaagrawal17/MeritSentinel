"""Evaluation Script for Candidate Resume Claim Extraction Accuracy.

Audits candidate resume against an expected_claims.json file located in
data/local/, .data_uploads/, or data/.

Reports:
- Extracted count, Expected count
- True Positives (TP), False Positives (FP), False Negatives (FN)
- Precision & Recall
- Line-level error analysis (all FPs and FNs)
- Complete architectural breakdown of LLM-based vs. Rule-based pipeline steps.
"""

import json
import os
import re
import sys
from pathlib import Path

# Set stdout encoding to UTF-8
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from resumeproof.core.llm import OpenAICompatibleClient, FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.claims import ClaimType

STOPWORDS = {
    "and", "with", "for", "the", "a", "an", "in", "on", "at", "to", "of",
    "by", "from", "as", "is", "using", "built", "implemented", "developed",
    "engineered", "trained", "across", "under", "per", "via", "over"
}

def match_claims(expected_list: list[dict], extracted_claims: list) -> dict:
    """1-to-1 greedy bipartite matching ensuring TP + FP = extracted and TP + FN = expected."""
    pairs = []
    for e_idx, exp in enumerate(expected_list):
        e_text = exp["text"].lower().strip()
        e_type = str(exp.get("claim_type", "")).lower().strip()
        e_proj = str(exp.get("project_id", "")).lower().strip() if exp.get("project_id") else None
        e_words = set(re.findall(r"[a-zA-Z0-9]+", e_text)) - STOPWORDS

        for c_idx, c in enumerate(extracted_claims):
            c_text = c.text.lower().strip()
            c_type = getattr(c.claim_type, "value", str(c.claim_type)).lower().strip()
            c_proj = str(c.project_id).lower().strip() if c.project_id else None
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

            # Type and project alignment bonus
            if e_type and c_type and e_type == c_type:
                score += 0.05
            if e_proj and c_proj and e_proj == c_proj:
                score += 0.05

            if score >= 0.5:
                pairs.append((score, e_idx, c_idx))

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
        "fp_claims": fp_claims,
        "fn_expected": fn_expected,
        "matches": [(expected_list[e], extracted_claims[c], score) for e, c, score in matches],
    }

def find_candidate_files() -> tuple[Path, Path]:
    """Finds candidate resume PDF and expected claims JSON file."""
    # Find resume
    resume_path = None
    upload_dir = ROOT_DIR / ".data_uploads"
    if upload_dir.exists():
        for f in upload_dir.glob("*.pdf"):
            if "resume" in f.name.lower() or "sanidhya" in f.name.lower():
                resume_path = f
                break

    if not resume_path:
        # Fallback to data/
        for f in (ROOT_DIR / "data").glob("**/*.pdf"):
            if "resume" in f.name.lower() and "demo" not in str(f):
                resume_path = f
                break

    # Find expected claims
    search_dirs = [
        ROOT_DIR / "data" / "local",
        ROOT_DIR / ".data_uploads",
        ROOT_DIR / "data",
    ]
    expected_path = None
    for d in search_dirs:
        if d.exists():
            for f in d.glob("*expected*.json"):
                if "demo" not in str(f):
                    expected_path = f
                    break
            if expected_path:
                break

    return resume_path, expected_path

def run_evaluation(use_real_llm: bool = True):
    print("=" * 80)
    print("RESUMEPROOF CANDIDATE RESUME EXTRACTION EVALUATION AUDIT")
    print("=" * 80)

    resume_path, expected_path = find_candidate_files()

    if not resume_path or not resume_path.exists():
        print(f"ERROR: Candidate resume PDF not found in .data_uploads/ or data/.")
        sys.exit(1)

    print(f"Target Resume PDF: {resume_path}")

    # Parse resume
    parsed = parse_resume(resume_path)
    print(f"Total lines parsed: {len(parsed.lines)} across {parsed.total_pages} page(s).")

    # Client
    if use_real_llm:
        print("Model: Ollama qwen2.5:3b (OpenAI-compatible native schema)")
        client = OpenAICompatibleClient()
    else:
        print("Model: FakeLLMClient (rule-based fallback validation)")
        client = FakeLLMClient()

    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="candidate",
        user_id="candidate_user",
        model="qwen2.5:3b",
    )

    print(f"\n[Extraction Summary]")
    print(f"- Total Profile Facts: {len(facts)}")
    print(f"- Total Mastery Claims: {len(claims)}")

    tool_claims = [c for c in claims if c.claim_type == ClaimType.TOOL]
    skill_claims = [c for c in claims if c.claim_type == ClaimType.SKILL]
    metric_claims = [c for c in claims if c.claim_type == ClaimType.PROJECT_METRIC]
    exp_claims = [c for c in claims if c.claim_type == ClaimType.EXPERIENCE_CLAIM]
    achieve_claims = [c for c in claims if c.claim_type == ClaimType.ACHIEVEMENT]

    print(f"  * Stack-Line Tools: {len(tool_claims)}")
    print(f"  * Skills: {len(skill_claims)}")
    print(f"  * Project Metrics: {len(metric_claims)}")
    print(f"  * Experience Claims: {len(exp_claims)}")
    print(f"  * Achievements: {len(achieve_claims)}")

    if not expected_path or not expected_path.exists():
        print(f"\n[NOTE] Expected claims ground-truth JSON not found yet.")
        print(f"Checked paths: data/local/*expected*.json, .data_uploads/*expected*.json")
        print(f"To run accuracy scoring, place expected_claims.json in data/local/ or .data_uploads/.")
        print(f"Currently extracted {len(claims)} verified claims ready for evaluation.")
        return

    print(f"\nGround Truth Key File: {expected_path}")
    raw_gt = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_list = raw_gt.get("expected_claims", raw_gt) if isinstance(raw_gt, dict) else raw_gt

    res = match_claims(expected_list, claims)

    precision = res["tp"] / (res["tp"] + res["fp"]) if (res["tp"] + res["fp"]) > 0 else 0.0
    recall = res["tp"] / (res["tp"] + res["fn"]) if (res["tp"] + res["fn"]) > 0 else 0.0

    print("\n" + "=" * 80)
    print("EXTRACTION ACCURACY REPORT")
    print("=" * 80)
    print(f"Extracted: {res['extracted']}")
    print(f"Expected:  {res['expected']}")
    print(f"TP:        {res['tp']}")
    print(f"FP:        {res['fp']}")
    print(f"FN:        {res['fn']}")
    print(f"Check: TP + FP = {res['tp'] + res['fp']} (== {res['extracted']})")
    print(f"Check: TP + FN = {res['tp'] + res['fn']} (== {res['expected']})")
    print(f"Precision: {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.1f}%)")

    if res["fp_claims"]:
        print("\n--- False Positive Claims (Extracted but not in ground truth) ---")
        for c in res["fp_claims"]:
            c_type = getattr(c.claim_type, "value", str(c.claim_type))
            print(f"  [{c_type:15s}] (Proj: {str(c.project_id):16s} Line: {str(c.source_line):2s}) {c.text}")

    if res["fn_expected"]:
        print("\n--- False Negative Claims (In ground truth but missed by extractor) ---")
        for e in res["fn_expected"]:
            e_type = e.get("claim_type", "claim")
            e_proj = e.get("project_id", "None")
            print(f"  [{e_type:15s}] (Proj: {str(e_proj):16s}) {e.get('text', '')}")

    print("\n" + "=" * 80)
    print("PIPELINE ARCHITECTURE: RULE-BASED VS. LLM-BASED STEPS")
    print("=" * 80)
    print("""
1. PDF Parsing & Text Extraction (RULE-BASED):
   - Engine: pdfplumber layout extraction with pypdf fallback.
   - Normalization: Ligature decoding (ff, fi, fl, ffi), unicode quotes, dash collapse.
   - Unwrapping: Multi-line wrapped text reconstruction preserving metrics and punctuation.

2. Structural Resume Mapping (RULE-BASED):
   - detect_section_header(): Maps headers across case styles (UPPERCASE and Title Case)
     to canonical sections: PROJECTS, SKILLS, EXPERIENCE, ACHIEVEMENTS, EDUCATION.
   - _map_resume_lines(): Associates line numbers with active sections and active project_id.

3. Profile Fact Extraction (HYBRID: LLM + RULE-BASED):
   - LLM: qwen2.5:3b extracts structured candidate metadata (Name, Email, CGPA, Degree, etc.).
   - Rule Guard: Non-experience resumes strictly block employer/employment dates.

4. Mastery Claim Extraction (HYBRID: LLM + RULE-BASED):
   - LLM: qwen2.5:3b extracts core claims, action sentences, and technical metrics.
   - Rule Pass 1 (Tech-Stack Tools): Automatically parses project stack lines, creating
     ClaimType.TOOL claims linked to their parent project (PyTorch, ONNX Runtime, etc.).
   - Rule Pass 2 (Skill Splitting): Splits comma-separated skill lists into distinct skill claims.
   - Rule Pass 3 (Bullet Fallback): Captures any valid bullet points missed by LLM.
   - Rule Pass 4 (Deduplication): Deterministic deduplication keying on (claim_type, project_id, text).
""")

if __name__ == "__main__":
    use_real = "--fake-llm" not in sys.argv
    run_evaluation(use_real_llm=use_real)
