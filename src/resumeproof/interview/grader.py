"""Grader using qwen2.5:7b, temperature 0, atomic rubric points, and deterministic scoring."""

import time
import logging
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.config import settings
from resumeproof.core.llm import LLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem, AtomicPoint

logger = logging.getLogger("resumeproof.interview.grader")
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



GENERAL_TECHNICAL_ALIASES: list[tuple[str, list[str]]] = [
    ("dict", ["dictionary", "dictionaries"]),
    ("dictionary", ["dict", "dicts"]),
    ("wal", ["write-ahead logging", "write-ahead log", "wal mode", "wal persistence"]),
    ("write-ahead logging", ["wal", "write-ahead log", "wal mode", "wal persistence"]),
    ("byte", ["octet", "bytes", "octets"]),
    ("octet", ["byte", "bytes", "octets"]),
]


def _search_token_boundary(pattern: str, text: str) -> Optional[str]:
    import re
    m = re.search(rf"(?<![a-zA-Z0-9_]){re.escape(pattern)}(?![a-zA-Z0-9_])", text, re.IGNORECASE)
    return text[m.start():m.end()] if m else None


def resolve_alias_or_numeric_quote(cand_quote: str, answer_text: str) -> Optional[str]:
    """Resolves numeric, unit, range, and general technical aliases dynamically against answer_text."""
    import re
    from resumeproof.verifier.injection import normalize_words_to_numbers

    clean_q = cand_quote.strip().strip("\"'.,;:()")
    if not clean_q:
        return None

    # 1. Direct verbatim / case-insensitive match
    if clean_q.isalpha():
        found = _search_token_boundary(clean_q, answer_text)
        if found:
            return found
    elif clean_q in answer_text:
        return clean_q
    elif clean_q.lower() in answer_text.lower():
        idx = answer_text.lower().find(clean_q.lower())
        return answer_text[idx : idx + len(clean_q)]

    # 2. Flexible whitespace between numbers and units (e.g. "1.6ms" <-> "1.6 ms")
    try:
        flex_pat = re.sub(r"(\d)\s*([a-zA-Z%])", r"\1\\s*\2", re.escape(clean_q))
        m = re.search(rf"(?<![a-zA-Z0-9_]){flex_pat}(?![a-zA-Z0-9_])", answer_text, re.IGNORECASE)
        if m:
            return answer_text[m.start():m.end()]
    except Exception:
        pass

    # 3. Decimal <-> Percentage & raw <-> k-suffix conversion
    try:
        val_float = float(clean_q)
        if 0.0 < val_float < 1.0:
            pct = round(val_float * 100.0, 6)
            for ps in (f"{pct:g}%", f"{pct:g} percent", f"{pct:g}"):
                found = _search_token_boundary(ps, answer_text)
                if found:
                    return found
        elif val_float >= 100.0:
            k_val = round(val_float / 1000.0, 6)
            found = _search_token_boundary(f"{k_val:g}k", answer_text)
            if found:
                return found
    except ValueError:
        pass

    if clean_q.endswith("%"):
        try:
            dec = round(float(clean_q[:-1]) / 100.0, 6)
            found = _search_token_boundary(f"{dec:g}", answer_text)
            if found:
                return found
        except ValueError:
            pass

    # 4. Milliseconds <-> Seconds conversion
    ms_match = re.match(r"^(\d+(?:\.\d+)?)\s*ms$", clean_q, re.IGNORECASE)
    if ms_match:
        sec = round(float(ms_match.group(1)) / 1000.0, 9)
        for ss in (f"{sec:g}s", f"{sec:g} s"):
            found = _search_token_boundary(ss, answer_text)
            if found:
                return found

    s_match = re.match(r"^(\d+(?:\.\d+)?)\s*s$", clean_q, re.IGNORECASE)
    if s_match:
        ms_val = round(float(s_match.group(1)) * 1000.0, 6)
        for mss in (f"{ms_val:g}ms", f"{ms_val:g} ms"):
            found = _search_token_boundary(mss, answer_text)
            if found:
                return found

    # 5. k-suffix -> raw integer/float
    k_match = re.match(r"^(\d+(?:\.\d+)?)\s*k$", clean_q, re.IGNORECASE)
    if k_match:
        raw_val = round(float(k_match.group(1)) * 1000.0, 6)
        raw_str = f"{int(raw_val)}" if raw_val.is_integer() else f"{raw_val:g}"
        found = _search_token_boundary(raw_str, answer_text)
        if found:
            return found

    # 6. Bracket range [a, b] <-> a-b / a to b
    bracket_match = re.match(r"^\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]$", clean_q)
    if bracket_match:
        a, b = bracket_match.groups()
        for ra in (f"{a}-{b}ms", f"{a}-{b} ms", f"{a}-{b}", f"{a} to {b}ms", f"{a} to {b} ms", f"{a} to {b}", f"{a} - {b}ms", f"{a} - {b}", f"{a}ms-{b}ms"):
            found = _search_token_boundary(ra, answer_text)
            if found:
                return found

    # 7. General technical alias table
    for key, aliases in GENERAL_TECHNICAL_ALIASES:
        if clean_q.lower() == key.lower():
            for alt in aliases:
                found = _search_token_boundary(alt, answer_text)
                if found:
                    return found
        for alt in aliases:
            if clean_q.lower() == alt.lower():
                found = _search_token_boundary(key, answer_text)
                if found:
                    return found
                for alt2 in aliases:
                    found = _search_token_boundary(alt2, answer_text)
                    if found:
                        return found

    # 8. Written English numbers fallback
    try:
        norm_ans = normalize_words_to_numbers(answer_text)
        if norm_ans != answer_text:
            found_norm = resolve_alias_or_numeric_quote(clean_q, norm_ans)
            if found_norm and found_norm in answer_text:
                return found_norm
    except Exception:
        pass

    return None


def check_deterministic_point_match(pt: AtomicPoint, answer_text: str) -> Optional[str]:
    """Algorithmically checks if numeric, unit, range, or alias tokens from pt match answer_text."""
    import re
    ans_clean = answer_text.strip()
    if not ans_clean:
        return None
    pt_combined = f"{pt.text} {pt.source_quote}"

    candidates: list[str] = []
    candidates.extend(re.findall(r"\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]", pt_combined))
    candidates.extend(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|ms|s|k)\b", pt_combined, re.IGNORECASE))
    candidates.extend(re.findall(r"\b(?:wal|write-ahead logging)\b", pt_combined, re.IGNORECASE))

    seen: set[str] = set()
    for tok in candidates:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        matched = resolve_alias_or_numeric_quote(tok, ans_clean)
        if matched and matched in answer_text:
            return matched

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

        try:
            raw_grade = self.llm_client.structured_completion(
                messages=messages,
                schema=GradeStructuredPayload,
                model=self.model,
                temperature=0.0,
                timeout=timeout,
            )
        except Exception as exc:
            if self.model != settings.interviewer_model and ("404" in str(exc) or "not found" in str(exc).lower()):
                logger.warning(f"Grader model '{self.model}' unavailable ({exc}); falling back to '{settings.interviewer_model}'.")
                raw_grade = self.llm_client.structured_completion(
                    messages=messages,
                    schema=GradeStructuredPayload,
                    model=settings.interviewer_model,
                    temperature=0.0,
                    timeout=timeout,
                )
            else:
                raise

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
            final_evidence_quote = ""

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

