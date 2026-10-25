"""Strict Deterministic Verifier: Zero LLM in the blocking path."""

from dataclasses import dataclass, field
import re
from typing import Optional

from resumeproof.schemas.claims import Claim
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.verifier.injection import (
    normalize_unicode_homoglyphs,
    normalize_words_to_numbers,
)

# Known technical tools and languages for word-boundary matching
KNOWN_TOOLS = [
    "C++", "C#", "C", "Java", "JavaScript", "TypeScript", "Python", "Go", "Golang",
    "Rust", "Kotlin", "Swift", "Scala", "PHP", "Ruby", "Perl", "Lua", "SQL",
    "FastAPI", "Flask", "Django", "Spring", "Node.js", "Express", "React",
    "PostgreSQL", "MySQL", "SQLite", "DuckDB", "Redis", "MongoDB", "Cassandra",
    "Kafka", "RabbitMQ", "Raft", "Paxos", "Docker", "Kubernetes", "Istio",
    "AWS", "GCP", "Azure", "Linux", "POSIX", "gRPC", "Protobuf", "ONNX",
    "LightGBM", "PyTorch", "TensorFlow", "Scikit-Learn", "Jepsen", "Locust",
]

# Regex for word boundary matching, handling special characters like ++ and #
def build_tool_pattern(tool: str) -> re.Pattern:
    escaped = re.escape(tool)
    if tool in ("C++", "C#"):
        return re.compile(rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_\+#])", re.IGNORECASE)
    elif tool == "C":
        return re.compile(rf"(?<![a-zA-Z0-9_\+#])C(?![a-zA-Z0-9_\+#])")
    else:
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

TOOL_REGEX_MAP = {tool: build_tool_pattern(tool) for tool in KNOWN_TOOLS}


class VerificationBlockedError(Exception):
    """Raised when an unverified or blocked resume draft attempts export."""
    pass


@dataclass
class SentenceVerificationResult:
    sentence_idx: int
    sentence_text: str
    status: str  # "VERIFIED" | "BLOCKED"
    supporting_claim_id: Optional[str] = None
    extracted_entities: dict = field(default_factory=dict)
    blocking_reasons: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    passed: bool
    total_sentences: int
    verified_count: int
    blocked_count: int
    sentence_results: list[SentenceVerificationResult]
    blocking_summary: list[str] = field(default_factory=list)

    @property
    def pass_ratio(self) -> float:
        if self.total_sentences == 0:
            return 1.0
        return self.verified_count / self.total_sentences


def normalize_number_string(val: str) -> str:
    """Normalizes numeric representations (e.g. '80,000' -> '80000', '45 ms' -> '45ms')."""
    s = val.replace(",", "").strip().lower()
    s = re.sub(r"\s*ms\b", "ms", s)
    s = re.sub(r"\s*(?:req/sec|reqs/sec|requests/sec|rps|qps)\b", " req/sec", s)
    s = re.sub(r"\s*%", "%", s)
    return s.strip()



def extract_numbers_and_metrics(text: str) -> list[str]:
    """Extracts numbers, percentages, latencies, and metrics from text."""
    # Find numbers with units like ms, req/sec, %, or standalone numbers
    patterns = [
        r"\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:req/sec|reqs/sec|requests/sec|rps|qps)\b",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s*(?:ms|seconds|minutes)\b",
        r"\b\d+(?:,\d+)*(?:\.\d+)?\b",
    ]
    matches = []
    seen_spans = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            span = m.span()
            if not any(span[0] >= s[0] and span[1] <= s[1] for s in seen_spans):
                seen_spans.add(span)
                matches.append(normalize_number_string(m.group(0)))
    return matches



def extract_tools_from_text(text: str) -> list[str]:
    """Extracts known technical tools using strict word-boundary matching."""
    tools = []
    for tool_name, reg in TOOL_REGEX_MAP.items():
        if reg.search(text):
            # Special check: ensure Java is not matched in JavaScript
            if tool_name.lower() == "java" and re.search(r"\bjavascript\b", text, re.IGNORECASE):
                # Check if java appears outside javascript
                cleaned = re.sub(r"\bjavascript\b", "", text, flags=re.IGNORECASE)
                if not reg.search(cleaned):
                    continue
            tools.append(tool_name)
    return tools


class StrictVerifier:
    """Deterministic Verifier with zero LLM in the blocking path."""

    def __init__(self, allow_unverified_profile_headers: bool = True):
        self.allow_headers = allow_unverified_profile_headers

    def split_into_sentences(self, text: str) -> list[str]:
        """Splits resume text or markdown into logical bullet points and sentences."""
        lines = text.strip().splitlines()
        sentences = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                sentences.append(stripped)
                continue
            # Split bullets or sentences
            if stripped.startswith(("-", "*", "•")):
                clean_bullet = re.sub(r"^[-*•]\s*", "", stripped).strip()
                # If bullet has multiple sentences, split by period followed by space
                subs = re.split(r"(?<=[.!?])\s+", clean_bullet)
                for s in subs:
                    if s.strip():
                        sentences.append(s.strip())
            else:
                subs = re.split(r"(?<=[.!?])\s+", stripped)
                for s in subs:
                    if s.strip():
                        sentences.append(s.strip())
        return sentences

    def verify_sentence(
        self,
        sentence: str,
        eligible_claims: list[Claim],
        confirmed_facts: list[ProfileFact],
        sentence_idx: int = 1,
    ) -> SentenceVerificationResult:
        """Deterministically verifies a single sentence against backing claims and facts."""
        reasons: list[str] = []
        normalized_sentence = normalize_words_to_numbers(sentence)

        # 1. Check if sentence is a structural markdown heading
        if sentence.startswith("#"):
            h_text = sentence.lstrip("#").strip().upper()
            if h_text in ("EDUCATION", "EXPERIENCE", "WORK EXPERIENCE", "PROJECTS", "VERIFIED PROJECTS", "SKILLS", "TECHNICAL SKILLS", "VERIFIED TECHNICAL SKILLS"):
                return SentenceVerificationResult(
                    sentence_idx=sentence_idx,
                    sentence_text=sentence,
                    status="VERIFIED",
                    supporting_claim_id="structural:heading",
                )
            # Check name in heading
            name_fact = next((f for f in confirmed_facts if f.fact_type == ProfileFactType.NAME), None)
            if name_fact and name_fact.value.lower() in sentence.lower():
                return SentenceVerificationResult(
                    sentence_idx=sentence_idx,
                    sentence_text=sentence,
                    status="VERIFIED",
                    supporting_claim_id="fact:name",
                )
            # Check project heading matching an eligible claim's project_id
            clean_proj_h = sentence.lstrip("#").strip().lower()
            for c in eligible_claims:
                if c.project_id:
                    norm_pid = c.project_id.replace("-", " ").lower()
                    if norm_pid == clean_proj_h or norm_pid in clean_proj_h or clean_proj_h in norm_pid:
                        return SentenceVerificationResult(
                            sentence_idx=sentence_idx,
                            sentence_text=sentence,
                            status="VERIFIED",
                            supporting_claim_id=f"project:{c.project_id}",
                        )

        # 2. Check if sentence matches confirmed profile facts (e.g. contact, degree, GPA)
        for f in confirmed_facts:
            f_norm = normalize_number_string(f.value)
            if f.value.lower() in sentence.lower() or f_norm in normalized_sentence.lower():
                return SentenceVerificationResult(
                    sentence_idx=sentence_idx,
                    sentence_text=sentence,
                    status="VERIFIED",
                    supporting_claim_id=f"fact:{f.fact_type.value}",
                    extracted_entities={"fact_type": f.fact_type.value, "fact_value": f.value},
                )

        # 3. Extract entities from candidate sentence
        extracted_tools = extract_tools_from_text(normalized_sentence)
        extracted_numbers = extract_numbers_and_metrics(normalized_sentence)

        # 4. Check against eligible claims
        best_match_claim: Optional[Claim] = None
        max_overlap_score = 0.0

        clean_sent_words = set(re.findall(r"[a-zA-Z0-9]+", normalized_sentence.lower()))

        for claim in eligible_claims:
            claim_norm = normalize_words_to_numbers(f"{claim.text} {' '.join(claim.tools)} {' '.join(claim.metric_values)}")
            claim_words = set(re.findall(r"[a-zA-Z0-9]+", claim_norm.lower()))

            overlap = len(clean_sent_words.intersection(claim_words))
            score = overlap / max(1, len(clean_sent_words))

            # Exact substring match gives 1.0
            if claim.text.strip().lower() in sentence.strip().lower() or sentence.strip().lower() in claim.text.strip().lower():
                score = 1.0

            if score > max_overlap_score:
                max_overlap_score = score
                best_match_claim = claim

        if not best_match_claim or max_overlap_score < 0.4:
            # Check if this sentence is a list of verified skills (e.g. "FastAPI, Python, SQLite, Sockets")
            if ("," in sentence or len(extracted_tools) >= 2) and extracted_tools:
                all_eligible_tools = set()
                for c in eligible_claims:
                    all_eligible_tools.update(t.lower() for t in c.tools)
                    for t_name in extract_tools_from_text(c.text):
                        all_eligible_tools.add(t_name.lower())
                    if getattr(c, "claim_type", None) and getattr(c.claim_type, "value", str(c.claim_type)) == "skill":
                        all_eligible_tools.add(c.text.strip().lower())

                unverified_tools = [t for t in extracted_tools if t.lower() not in all_eligible_tools]
                if not unverified_tools:
                    return SentenceVerificationResult(
                        sentence_idx=sentence_idx,
                        sentence_text=sentence,
                        status="VERIFIED",
                        supporting_claim_id="skills:aggregate",
                        extracted_entities={"tools": extracted_tools},
                    )
                else:
                    for ut in unverified_tools:
                        reasons.append(f"Unbacked tool: '{ut}' is not present in supporting claim or verified skill set.")
                    return SentenceVerificationResult(
                        sentence_idx=sentence_idx,
                        sentence_text=sentence,
                        status="BLOCKED",
                        supporting_claim_id=None,
                        extracted_entities={"tools": extracted_tools, "numbers": extracted_numbers},
                        blocking_reasons=reasons,
                    )

            reasons.append("No supporting eligible claim or confirmed profile fact found for sentence.")
            return SentenceVerificationResult(
                sentence_idx=sentence_idx,
                sentence_text=sentence,
                status="BLOCKED",
                supporting_claim_id=None,
                extracted_entities={"tools": extracted_tools, "numbers": extracted_numbers},
                blocking_reasons=reasons,
            )

        # Grounding checks against best_match_claim:
        claim_norm_text = normalize_words_to_numbers(f"{best_match_claim.text} {' '.join(best_match_claim.tools)} {' '.join(best_match_claim.metric_values)}")
        claim_tools = {t.lower() for t in best_match_claim.tools}
        claim_numbers = [normalize_number_string(n) for n in extract_numbers_and_metrics(claim_norm_text)]

        # Check Tools: Every tool mentioned in the sentence must be present in the claim's tools or text
        for t in extracted_tools:
            t_lower = t.lower()
            if t_lower not in claim_tools and not TOOL_REGEX_MAP[t].search(best_match_claim.text):
                # Check if tool is confirmed in another claim
                in_other_eligible = any(t_lower in (ct.lower() for ct in c.tools) for c in eligible_claims)
                if not in_other_eligible:
                    reasons.append(f"Unbacked tool: '{t}' is not present in supporting claim or verified skill set.")

        # Check Numbers: Derived or rounded numbers are strictly BLOCKED
        # Every number in the sentence must exist in the supporting claim
        for num in extracted_numbers:
            # Check if number appears in claim numbers or text
            num_clean = re.sub(r"[^0-9\.]", "", num)
            found_num = False
            for c_num in claim_numbers:
                c_clean = re.sub(r"[^0-9\.]", "", c_num)
                if num == c_num or (num_clean and num_clean == c_clean):
                    found_num = True
                    break
            if not found_num and num_clean and num_clean not in re.sub(r"[^0-9\.]", " ", claim_norm_text).split():
                reasons.append(f"Unbacked or derived/rounded number: '{num}' does not match verified claim numbers {claim_numbers}.")

        status = "BLOCKED" if reasons else "VERIFIED"
        return SentenceVerificationResult(
            sentence_idx=sentence_idx,
            sentence_text=sentence,
            status=status,
            supporting_claim_id=best_match_claim.id if status == "VERIFIED" else best_match_claim.id,
            extracted_entities={"tools": extracted_tools, "numbers": extracted_numbers},
            blocking_reasons=reasons,
        )

    def verify(
        self,
        resume_text: str,
        eligible_claims: list[Claim],
        confirmed_facts: list[ProfileFact],
    ) -> VerificationReport:
        """Verifies full resume text sentence by sentence. Zero LLM calls."""
        sentences = self.split_into_sentences(resume_text)
        results: list[SentenceVerificationResult] = []
        verified_count = 0
        blocked_count = 0
        blocking_summary: list[str] = []

        for idx, s in enumerate(sentences, start=1):
            res = self.verify_sentence(
                sentence=s,
                eligible_claims=eligible_claims,
                confirmed_facts=confirmed_facts,
                sentence_idx=idx,
            )
            results.append(res)
            if res.status == "VERIFIED":
                verified_count += 1
            else:
                blocked_count += 1
                for r in res.blocking_reasons:
                    blocking_summary.append(f"Sentence {idx}: {r}")

        passed = (blocked_count == 0)
        return VerificationReport(
            passed=passed,
            total_sentences=len(sentences),
            verified_count=verified_count,
            blocked_count=blocked_count,
            sentence_results=results,
            blocking_summary=blocking_summary,
        )

    def verify_or_raise(
        self,
        resume_text: str,
        eligible_claims: list[Claim],
        confirmed_facts: list[ProfileFact],
    ) -> VerificationReport:
        """Verifies full resume text and raises VerificationBlockedError if any sentence is blocked."""
        report = self.verify(resume_text, eligible_claims, confirmed_facts)
        if not report.passed:
            details = "\n  - " + "\n  - ".join(report.blocking_summary)
            raise VerificationBlockedError(
                f"Export blocked: {report.blocked_count} sentence(s) failed strict verification.\n"
                f"Violations:{details}"
            )
        return report

    def evaluate_test_set(
        self,
        test_samples: list[dict],
        eligible_claims: list[Claim],
        confirmed_facts: list[ProfileFact],
    ) -> dict:
        """Evaluates verifier on labeled sentences containing planted unsupported claims.

        test_samples: list of dicts with keys:
            - "sentence": str
            - "is_supported": bool (True = should verify, False = planted unsupported claim)
        """
        tp = 0  # correctly blocked unsupported claims
        fp = 0  # supported claim falsely blocked
        tn = 0  # supported claim correctly verified
        fn = 0  # unsupported claim erroneously passed as verified

        details = []

        for item in test_samples:
            s_text = item["sentence"]
            is_supported = item["is_supported"]
            res = self.verify_sentence(s_text, eligible_claims, confirmed_facts)
            is_blocked = (res.status == "BLOCKED")

            if not is_supported and is_blocked:
                tp += 1
            elif not is_supported and not is_blocked:
                fn += 1
            elif is_supported and is_blocked:
                fp += 1
            elif is_supported and not is_blocked:
                tn += 1

            details.append({
                "sentence": s_text,
                "is_supported": is_supported,
                "is_blocked": is_blocked,
                "passed_verification": res.status == "VERIFIED",
                "reasons": res.blocking_reasons,
            })

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)

        return {
            "total_samples": len(test_samples),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "details": details,
        }
