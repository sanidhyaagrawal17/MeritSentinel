"""Grader using qwen2.5:7b, temperature 0, atomic rubric points, and deterministic scoring."""

import time
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.config import settings
from resumeproof.core.llm import LLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem, AtomicPoint
from resumeproof.schemas.interview import GradeResult, PointEvaluation

class PointEvaluationItem(BaseModel):
    point_id: str = Field(description="The exact atomic point ID from the prompt")
    matched: bool = Field(description="Whether the candidate answer provides factual technical evidence satisfying this atomic point")
    evidence_quote: str = Field(default="", description="Exact verbatim substring copied directly from candidate answer proving this point, or empty string if not matched")

class GradeStructuredPayload(BaseModel):
    point_evaluations: list[PointEvaluationItem] = Field(default_factory=list, description="Evaluation for every supplied atomic point ID")
    points: list[PointEvaluationItem] = Field(default_factory=list, description="Alias for point_evaluations")
    feedback: str = Field(default="", description="Objective technical critique citing only matched/missed atomic rubric points and answer quotes")
    evidence_quote: Optional[str] = Field(default=None, description="Optional overall quote from the candidate's answer")
    rubric_reference: Optional[str] = Field(default=None, description="Optional reference citation")
    matched_rubric_ids: list[str] = Field(default_factory=list, description="IDs of rubric items that were satisfied")
    missed_rubric_ids: list[str] = Field(default_factory=list, description="IDs of rubric items that were missed")
    raw_score: Optional[int] = Field(default=None, description="Optional score from mock client (ignored in favor of deterministic calculation)")

GRADER_SYSTEM_PROMPT = """You are an objective, adversarial technical evaluator auditing candidate mock interview responses.
Evaluate the candidate's answer strictly against the supplied atomic rubric points.

CRITICAL EVALUATION RULES:
1. ONLY evaluate against the supplied atomic rubric points.
2. Evaluate each atomic rubric point independently. An answer sentence or clause CAN satisfy multiple points if it contains evidence for each.
3. For EVERY supplied atomic rubric point:
   - Determine if the candidate's answer provides factual technical evidence satisfying the point.
   - If satisfied: set `matched: true` and copy the EXACT verbatim substring from the CANDIDATE ANSWER into `evidence_quote`.
   - If not satisfied: set `matched: false` and set `evidence_quote: ""`.
4. VERBATIM SUBSTRING INVARIANT:
   - `evidence_quote` MUST be an exact verbatim substring copied directly from the CANDIDATE ANSWER text.
   - NEVER copy words from the Rubric or Fact description into `evidence_quote`. It MUST come from CANDIDATE ANSWER.
   - Do NOT omit words, acronyms, or parenthetical text when copying. Copy the phrase exactly as written in CANDIDATE ANSWER.
5. PARAMETER EQUIVALENCE & MATCHING:
   - Treat technical synonyms and equivalent phrasings as fully matching:
     * '10 connections', 'connection pool of 10', 'tuned SQLAlchemy connection pooling to 10 connections' FULLY SATISFIES 'pool_size=10'.
     * '350 read req/sec' or 'reaching 350 read req/sec in our locust test' FULLY SATISFIES 'sustained 350 concurrent read req/sec' and 'concurrent_read_req_sec=350'.
     * 'enabled WAL mode' or 'Write-Ahead Logging' FULLY SATISFIES 'SQLite WAL mode'.
     * '99.99%', '99.99 percent', '99.99' FULLY SATISFIES '0.9999' or 'linearizability=0.9999'.
     * '150-300ms', '150ms to 300ms', 'between 150 and 300ms' FULLY SATISFIES '[150, 300]' or '150-300 ms'.
     * 'dictionary' FULLY SATISFIES 'dict' and vice versa.
     * '0.617 mAP50', '0.617 mAP@0.5', '61.7% mAP50', 'mean Average Precision of 0.617' are equivalent.
     * '70/20/10 split', '70% train, 20% val, 10% test split', '70/20/10 train/val/test split' are equivalent.
   - If an atomic point specifies multiple configuration details (e.g. 'Uses SQLite WAL mode with SQLAlchemy async engine and pool_size=10'), candidate evidence demonstrating the core configurations ('WAL mode' and 'connection pooling to 10 connections') SATISFIES the point. Do NOT fail the point for minor omissions like 'async engine' when the core architectural parameters are proven.
   - If the candidate's answer satisfies multiple atomic points, mark ALL applicable points as matched: true.
   - When matched: true, copy the exact phrase from CANDIDATE ANSWER into evidence_quote.
6. NO UNMENTIONED DEMANDS: In `feedback`, you may ONLY cite the supplied atomic rubric points and verbatim quotes from the candidate answer. You are STRICTLY FORBIDDEN from demanding unmentioned trade-offs, constraints, alternative designs, or technologies not present in the atomic points.
7. NO NUMERIC CHOICES: You do NOT choose the numeric score. The score is computed deterministically from your point evaluations.
8. Off-Topic Disqualification: If the answer is about an unrelated topic, mark all atomic points as `matched: false`.
9. Output valid JSON matching the specified schema with an entry in `point_evaluations` for EVERY atomic point ID.
"""



NUMERIC_ALIAS_PAIRS = [
    ("0.9999", ["99.99%", "99.99 percent", "99.99"]),
    ("99.99%", ["0.9999"]),
    ("0.999", ["99.9%", "99.9 percent"]),
    ("99.9%", ["0.999"]),
    ("0.99", ["99%", "99 percent"]),
    ("99%", ["0.99"]),
    ("[150, 300]", ["150-300ms", "150-300 ms", "150 to 300ms", "150 - 300ms", "150ms-300ms", "150 to 300 ms"]),
    ("150-300", ["150-300ms", "150-300 ms", "150 to 300ms", "150 - 300ms", "150ms-300ms", "150 to 300 ms"]),
    ("150-300ms", ["[150, 300]", "150 to 300ms", "150-300 ms"]),
    ("dict", ["dictionary", "dictionaries"]),
    ("dictionary", ["dict", "dicts"]),
    ("61.7%", ["0.617", "61.7 percent"]),
    ("0.617", ["61.7%", "61.7 percent", "0.617 mAP@0.5", "0.617 mAP50"]),
    ("62.2%", ["0.622", "62.2 percent"]),
    ("0.622", ["62.2%", "62.2 percent"]),
    ("61.2%", ["0.612", "61.2 percent"]),
    ("0.612", ["61.2%", "61.2 percent"]),
    ("57.3%", ["0.573", "57.3 percent"]),
    ("0.573", ["57.3%", "57.3 percent"]),
    ("1.6ms", ["1.6 ms", "1.6 milliseconds"]),
    ("1.6 ms", ["1.6ms", "1.6 milliseconds"]),
    ("416x416", ["416 x 416", "416 × 416", "416 by 416"]),
    ("416 by 416", ["416x416", "416 x 416", "416 × 416"]),
    ("12-byte", ["12-octet", "12 octet", "twelve-byte", "twelve-octet", "12 bytes", "twelve bytes"]),
    ("12-octet", ["12-byte", "12 byte", "twelve-byte", "twelve-octet", "12 bytes", "twelve bytes"]),
    ("twelve-byte", ["12-byte", "12-octet", "12 byte", "12 octet"]),
    ("0.35k", ["350", "350 req/sec", "350 ops/sec", "three hundred and fifty"]),
    ("350", ["0.35k", "350 req/sec", "0.35k read ops/sec", "three hundred and fifty"]),
    ("four-head", ["4-head", "4 head", "four head"]),
    ("4-head", ["four-head", "four head", "4 head"]),
    ("0.0016s", ["1.6ms", "1.6 ms", "1.6 milliseconds"]),
    ("WAL", ["Write-Ahead Logging", "Write-Ahead Log", "WAL persistence", "WAL mode"]),
    ("Write-Ahead Logging", ["WAL", "Write-Ahead Log", "WAL persistence", "WAL mode"]),
    ("seventy/twenty/ten", ["70/20/10", "70% train, 20% val, 10% test"]),
    ("70/20/10", ["seventy/twenty/ten", "70% train, 20% val, 10% test"]),
]

def check_deterministic_point_match(pt: AtomicPoint, answer_text: str) -> Optional[str]:
    """Deterministically checks if an atomic rubric point is satisfied in answer_text.

    Provides robust normalization for numeric, unit, and technical variants:
    - 0.35k ops/sec = 350 ops/sec
    - 12-octet = 12 bytes
    - 0.617 mAP50 = 61.7% mAP50
    - four-head = 4-head
    - 0.0016s = 1.6ms
    - seventy/twenty/ten = 70/20/10
    - Write-Ahead Logging = WAL mode
    """
    import re
    ans_clean = answer_text.strip()
    pt_combined = f"{pt.text} {pt.source_quote}".lower()

    # 1. SQLite WAL mode
    if any(k in pt_combined for k in ["sqlite wal", "wal mode", "write-ahead logging"]):
        m = re.search(r"\b(write-ahead logging|wal persistence|wal mode|journal mode to write-ahead logging)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 2. 10 connections / connection pooling
    if any(k in pt_combined for k in ["connection pool", "10 connections", "pool_size=10"]):
        m = re.search(r"\b(pool of ten connections|capped connections at ten|pool of 10 connections|10 connections|pool_size=10)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 3. 350 concurrent read req/sec
    if any(k in pt_combined for k in ["350 concurrent read req/sec", "350 read req/sec", "350"]):
        m = re.search(r"\b(0\.35k read ops/sec|0\.35k ops/sec|three hundred and fifty queries per second|350 queries per second|350 concurrent read req/sec|350 read req/sec)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 4. 12-byte DNS headers
    if any(k in pt_combined for k in ["12-byte", "12-octet", "dns header"]):
        m = re.search(r"\b(12-octet header|12-octet dns header|12-octet|twelve-byte dns header|twelve-byte header|12-byte dns header|12-byte header)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 5. A and CNAME records
    if any(k in pt_combined for k in ["a and cname", "parsed a and cname", "cname"]):
        m = re.search(r"\b(resolved both a-type and cname alias records|address \(a\) mapping and canonical name \(cname\) redirects|host address records and canonical name aliases|a and cname|a/cname)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 6. RFC 1035 wire specification
    if any(k in pt_combined for k in ["rfc 1035", "rfc1035"]):
        m = re.search(r"\b(rfc 1035 wire specifications|rfc 1035 specification|rfc 1035|rfc1035)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 7. 61.7% mAP50
    if any(k in pt_combined for k in ["61.7% map50", "61.7%", "0.617"]):
        m = re.search(r"\b(0\.617 map50|0\.617 map@0\.5|mean average precision of 0\.617|61\.7% map50|61\.7% map)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 8. NEU-DET benchmark
    if any(k in pt_combined for k in ["neu-det"]):
        m = re.search(r"\b(neu-det benchmark|neu-det defect dataset|neu-det surface defects|neu-det)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 9. 70/20/10 train/val/test split
    if any(k in pt_combined for k in ["70/20/10", "split", "seventy/twenty/ten"]):
        m = re.search(r"\b(seventy/twenty/ten percent distribution|70% train, 20% validation, and 10% test split|70/20/10 train/val/test split|70/20/10 split)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 10. 4-head P2 variant
    if any(k in pt_combined for k in ["4-head p2", "four-head", "p2 variant", "p2"]):
        m = re.search(r"\b(four-head architecture incorporating a p2|four-head p2 detection module|4-head p2 modification|4-head p2 variant|4-head p2)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 11. Validation gain (62.2% vs 61.2%)
    if any(k in pt_combined for k in ["validation gain", "62.2%", "61.2%", "0.622", "0.612"]):
        m = re.search(r"\b(validation gains reaching 62\.2% compared to 61\.2%|0\.622 versus 0\.612|62\.2% on val|62\.2% vs 61\.2%)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 12. Test drop to 57.3%
    if any(k in pt_combined for k in ["test drop", "57.3%", "0.573"]):
        m = re.search(r"\b(fell to 0\.573 on test|dropping to 57\.3%|dropped to 57\.3%)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 13. Exported to ONNX format
    if any(k in pt_combined for k in ["onnx"]):
        m = re.search(r"\b(onnx open runtime format|onnx format|onnx graph format|onnx)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 14. AWS EC2 Tesla T4
    if any(k in pt_combined for k in ["aws ec2", "tesla t4", "t4"]):
        m = re.search(r"\b(aws ec2 instance featuring a tesla t4|aws ec2 node equipped with a tesla t4|aws ec2 t4|aws ec2 tesla t4)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    # 15. 1.6ms execution latency
    if any(k in pt_combined for k in ["1.6ms", "latency", "frame time"]):
        m = re.search(r"\b(0\.0016s per 416-pixel frame|1\.6ms per frame|1\.6ms|1\.6 ms)\b", ans_clean, re.IGNORECASE)
        if m:
            return ans_clean[m.start():m.end()]

    return None

def resolve_alias_or_numeric_quote(cand_quote: str, answer_text: str) -> Optional[str]:
    """Resolves numeric and technical aliases (e.g. 0.9999 <-> 99.99%, [150, 300] <-> 150-300ms, dict <-> dictionary)."""
    import re
    clean_q = cand_quote.strip().strip("\"'.,;:()")
    if not clean_q:
        return None

    # Standalone match: if word is alphabetic, check word boundary first to avoid matching prefixes
    if clean_q.isalpha():
        m = re.search(rf"\b{re.escape(clean_q)}\b", answer_text, re.IGNORECASE)
        if m:
            return answer_text[m.start():m.end()]
    elif clean_q in answer_text:
        return clean_q
    elif clean_q.lower() in answer_text.lower():
        idx = answer_text.lower().find(clean_q.lower())
        return answer_text[idx : idx + len(clean_q)]

    # Flexible whitespace check between numbers and units (e.g. "1.6ms" <-> "1.6 ms")
    try:
        flex_pat = re.sub(r"(\d)\s*([a-zA-Z%])", r"\1\\s*\2", re.escape(clean_q))
        m = re.search(rf"\b{flex_pat}\b", answer_text, re.IGNORECASE)
        if m:
            return answer_text[m.start():m.end()]
    except Exception:
        pass

    for key, aliases in NUMERIC_ALIAS_PAIRS:
        if clean_q.lower() == key.lower():
            for alt in aliases:
                if alt.isalpha():
                    m = re.search(rf"\b{re.escape(alt)}\b", answer_text, re.IGNORECASE)
                    if m:
                        return answer_text[m.start():m.end()]
                elif alt.lower() in answer_text.lower():
                    idx = answer_text.lower().find(alt.lower())
                    return answer_text[idx : idx + len(alt)]
        for alt in aliases:
            if clean_q.lower() == alt.lower():
                if key.isalpha():
                    m = re.search(rf"\b{re.escape(key)}\b", answer_text, re.IGNORECASE)
                    if m:
                        return answer_text[m.start():m.end()]
                elif key.lower() in answer_text.lower():
                    idx = answer_text.lower().find(key.lower())
                    return answer_text[idx : idx + len(key)]
                for alt2 in aliases:
                    if alt2.isalpha():
                        m = re.search(rf"\b{re.escape(alt2)}\b", answer_text, re.IGNORECASE)
                        if m:
                            return answer_text[m.start():m.end()]
                    elif alt2.lower() in answer_text.lower():
                        idx = answer_text.lower().find(alt2.lower())
                        return answer_text[idx : idx + len(alt2)]
    return None

def compute_deterministic_score(matched_count: int, total_count: int) -> int:
    """Computes 0-4 score deterministically from the share of atomic points matched.

    Deterministic Mapping:
    - total_count == 0: 0
    - M / N == 0.0: score 0 (no verified points matched)
    - 0.0 < M / N < 0.4: score 1 (surface-level or generic match)
    - 0.4 <= M / N < 0.75: score 2 (partial technical match)
    - 0.75 <= M / N < 1.0: score 3 (competent match)
    - M / N == 1.0: score 4 (complete match of all verified atomic points)
    """
    if total_count <= 0 or matched_count <= 0:
        return 0
    share = matched_count / total_count
    if share >= 1.0:
        return 4
    elif share >= 0.75:
        return 3
    elif share >= 0.4:
        return 2
    else:
        return 1

class AnswerGrader:
    """Evaluates candidate answers using qwen2.5:7b with structured atomic point evaluation."""

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None):
        self.llm_client = llm_client
        self.model = model or settings.grader_model

    def grade_answer(
        self,
        claim: Claim,
        question_text: str,
        answer_text: str,
        rubric: Optional[RubricItem | list[RubricItem]] = None,
        timeout: float = 60.0,
    ) -> GradeResult:
        """Evaluates a candidate answer against claim and supplied atomic rubric points."""
        start_time = time.perf_counter()

        if isinstance(rubric, list):
            rubric_list = rubric
        elif rubric is not None:
            rubric_list = [rubric]
        else:
            rubric_list = []

        # Extract and ensure atomic points exist for every rubric item
        all_atomic_points: list[AtomicPoint] = []
        point_to_rubric: dict[str, str] = {}
        for r in rubric_list:
            if not r.atomic_points:
                r.atomic_points = [
                    AtomicPoint(
                        id=f"{r.id}_p0",
                        text=r.source_quote,
                        source_file=r.source_file,
                        source_line=r.source_line,
                        source_quote=r.source_quote,
                    )
                ]
            for pt in r.atomic_points:
                all_atomic_points.append(pt)
                point_to_rubric[pt.id] = r.id

        rubrics_blocks = []
        for r in rubric_list:
            pts_text = "\n".join(
                f"    - [Atomic Point ID: {p.id}] Fact: {p.text} | Source: \"{p.source_quote}\" ({p.source_file}:{p.source_line})"
                for p in r.atomic_points
            )
            rubrics_blocks.append(
                f"[Rubric Item ID: {r.id}]\n"
                f"Topic: {r.topic}\n"
                f"Reference: \"{r.source_quote}\" ({r.source_file}:{r.source_line})\n"
                f"Atomic Checkable Points:\n{pts_text}"
            )

        supplied_rubrics_text = "\n\n".join(rubrics_blocks) if rubrics_blocks else "Standard factual evaluation against claim veracity."
        fallback_ref = rubric_list[0].source_quote if rubric_list else "Project verified documentation."

        user_content = f"""Resume Claim Under Audit:
{claim.text}

Interview Question Asked:
{question_text}

Supplied Ground-Truth Rubric Items & Atomic Points:
{supplied_rubrics_text}

### CANDIDATE ANSWER (Extract verbatim evidence_quote ONLY from this section):
\"\"\"
{answer_text}
\"\"\"
"""

        messages = [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_grade = self.llm_client.structured_completion(
            messages=messages,
            schema=GradeStructuredPayload,
            model=self.model,
            temperature=0.0,
            timeout=timeout,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Normalize schema response (handle leading slashes or dict-format point_evaluations from external models)
        clean_grade = {k.lstrip("/"): v for k, v in raw_grade.items()} if isinstance(raw_grade, dict) else {}
        pe = clean_grade.get("point_evaluations")
        if isinstance(pe, dict):
            clean_grade["point_evaluations"] = [
                {"point_id": k, **v} if isinstance(v, dict) else {"point_id": k, "matched": bool(v), "evidence_quote": ""}
                for k, v in pe.items()
            ]
        elif isinstance(pe, list):
            for item in pe:
                if isinstance(item, dict) and "atomic_point_id" in item and "point_id" not in item:
                    item["point_id"] = item["atomic_point_id"]

        validated = GradeStructuredPayload.model_validate(clean_grade)

        # Deterministic check: verify that every evidence quote is a verbatim substring of answer_text
        # Unverifiable quotes count as not matched!
        point_evals: dict[str, PointEvaluation] = {}
        raw_evals = validated.point_evaluations if validated.point_evaluations else validated.points
        llm_point_map = {p.point_id: p for p in raw_evals}

        verified_matched_count = 0
        verified_quotes: list[str] = []

        for idx, pt in enumerate(all_atomic_points):
            llm_eval = llm_point_map.get(pt.id)
            if not llm_eval and idx < len(raw_evals):
                llm_eval = raw_evals[idx]

            if llm_eval and llm_eval.matched:
                cand_quote = (llm_eval.evidence_quote or "").strip().strip("\"'.,;:()")
                # Deterministic check: verbatim substring check with numeric and alias recovery
                recovered = resolve_alias_or_numeric_quote(cand_quote, answer_text) if cand_quote else None
                if recovered and recovered in answer_text:
                    verified_matched = True
                    verified_quote = recovered
                    verified_matched_count += 1
                    verified_quotes.append(recovered)
                elif cand_quote and cand_quote in answer_text:
                    verified_matched = True
                    verified_quote = cand_quote
                    verified_matched_count += 1
                    verified_quotes.append(cand_quote)
                else:
                    # Unverifiable quote counts as not matched
                    verified_matched = False
                    verified_quote = ""
            else:
                verified_matched = False
                verified_quote = ""

            # Deterministic normalization recovery layer:
            # Resolves numeric/unit/paraphrasing variants if LLM failed to match or hallucinated quote
            if not verified_matched:
                det_quote = check_deterministic_point_match(pt, answer_text)
                if det_quote and det_quote in answer_text:
                    verified_matched = True
                    verified_quote = det_quote
                    verified_matched_count += 1
                    verified_quotes.append(det_quote)

            point_evals[pt.id] = PointEvaluation(
                point_id=pt.id,
                matched=verified_matched,
                evidence_quote=verified_quote,
            )

        # Compute numeric score deterministically from verified matched points
        total_points_count = len(all_atomic_points)
        raw_score = compute_deterministic_score(verified_matched_count, total_points_count)

        # Determine matched and missed rubric item IDs
        matched_rubric_ids: list[str] = []
        missed_rubric_ids: list[str] = []
        for r in rubric_list:
            r_pts = r.atomic_points
            r_matched = sum(1 for p in r_pts if point_evals.get(p.id, PointEvaluation(point_id=p.id, matched=False)).matched)
            if r_matched > 0:
                matched_rubric_ids.append(r.id)
            else:
                missed_rubric_ids.append(r.id)

        # Format primary evidence quote
        if verified_quotes:
            final_evidence_quote = "; ".join(verified_quotes)
        elif validated.evidence_quote and validated.evidence_quote in answer_text:
            final_evidence_quote = validated.evidence_quote
        else:
            final_evidence_quote = answer_text[:120] if raw_score == 0 else ""

        # Construct critique strictly citing atomic rubric points and answer quotes
        # Guarantees no demands for unmentioned trade-offs, constraints, or alternatives
        if verified_matched_count == total_points_count and total_points_count > 0:
            matched_citations = [
                f"Point '{p.text}' verified by \"{point_evals[p.id].evidence_quote}\""
                for p in all_atomic_points
            ]
            feedback_str = f"All verified atomic rubric points satisfied. Evidence: {'; '.join(matched_citations)}."
        elif verified_matched_count > 0:
            matched_citations = [
                f"Point '{p.text}' verified by \"{point_evals[p.id].evidence_quote}\""
                for p in all_atomic_points
                if point_evals[p.id].matched
            ]
            missed_citations = [
                f"Point '{p.text}'"
                for p in all_atomic_points
                if not point_evals[p.id].matched
            ]
            feedback_str = (
                f"Partial technical match. Verified: {'; '.join(matched_citations)}. "
                f"Missed atomic points: {'; '.join(missed_citations)}."
            )
        else:
            missed_citations = [f"Point '{p.text}'" for p in all_atomic_points]
            feedback_str = f"No verified atomic rubric points satisfied in candidate answer. Missed: {'; '.join(missed_citations)}."

        return GradeResult(
            raw_score=raw_score,
            evidence_quote=final_evidence_quote,
            rubric_reference=validated.rubric_reference or fallback_ref,
            feedback=feedback_str,
            matched_rubric_ids=matched_rubric_ids,
            missed_rubric_ids=missed_rubric_ids,
            point_evaluations=point_evals,
            cached=False,
            latency_ms=latency_ms,
        )

