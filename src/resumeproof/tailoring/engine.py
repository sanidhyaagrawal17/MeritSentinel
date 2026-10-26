"""Tailoring Engine: Selects defensible mastery-eligible claims and confirmed profile facts."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional

from resumeproof.schemas.claims import Claim, ClaimStatus
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.mastery import MasteryState
from resumeproof.tailoring.matcher import HybridMatcher, MatchScore
from resumeproof.verifier.injection import AdversarialDetector


@dataclass
class RequirementGap:
    requirement_text: str
    status: str  # "matched_and_defensible" | "needs_interview_practice" | "unsupported_gap"
    supporting_claim_id: Optional[str] = None
    supporting_claim_text: Optional[str] = None
    effective_score: Optional[float] = None
    confidence: Optional[float] = None
    match_score: Optional[MatchScore] = None
    reason: str = ""


@dataclass
class GapReport:
    matched_and_defensible: list[RequirementGap] = field(default_factory=list)
    needs_interview_practice: list[RequirementGap] = field(default_factory=list)
    unsupported_gaps: list[RequirementGap] = field(default_factory=list)

    @property
    def total_requirements(self) -> int:
        return len(self.matched_and_defensible) + len(self.needs_interview_practice) + len(self.unsupported_gaps)

    @property
    def defensibility_ratio(self) -> float:
        if self.total_requirements == 0:
            return 1.0
        return len(self.matched_and_defensible) / self.total_requirements


@dataclass
class TailoringResult:
    tailored_markdown: str
    gap_report: GapReport
    included_claims: list[Claim]
    included_facts: list[ProfileFact]
    attribution_trace: dict[str, str]  # sentence -> claim_id or fact_type


class TailoringEngine:
    """Selects only mastery-eligible claims (effective_score >= 2.5, confidence >= 0.5)

    and confirmed profile facts to produce a tailored, defensible resume.
    """

    def __init__(
        self,
        matcher: Optional[HybridMatcher] = None,
        score_threshold: float = 2.5,
        confidence_threshold: float = 0.5,
        match_cutoff: float = 0.35,
        adversarial_detector: Optional[AdversarialDetector] = None,
    ):
        self.matcher = matcher or HybridMatcher()
        self.score_threshold = score_threshold
        self.confidence_threshold = confidence_threshold
        self.match_cutoff = match_cutoff
        self.adversarial_detector = adversarial_detector or AdversarialDetector()

    def is_claim_eligible(
        self,
        claim: Claim,
        mastery: Optional[MasteryState],
        all_claims: Optional[list[Claim]] = None,
        mastery_map: Optional[dict[str, MasteryState]] = None,
    ) -> bool:
        """Determines if a claim is strictly eligible for inclusion in tailored resume.

        Project stack-line tools (ClaimType.TOOL with project_id) are eligible if:
        1. Confirmed by candidate in review (claim.status == ClaimStatus.CONFIRMED), OR
        2. Their parent project has at least one mastery-eligible claim.

        All other claims must be confirmed AND meet mastery thresholds.
        """
        ctype_val = getattr(claim.claim_type, "value", str(claim.claim_type)).lower()
        if ctype_val == "tool" and claim.project_id:
            if claim.status == ClaimStatus.CONFIRMED:
                return True
            if all_claims and mastery_map:
                for other in all_claims:
                    if other.project_id == claim.project_id and other.id != claim.id:
                        other_m = mastery_map.get(other.id)
                        if (
                            other.status == ClaimStatus.CONFIRMED
                            and other_m is not None
                            and other_m.effective_score >= self.score_threshold
                            and other_m.confidence >= self.confidence_threshold
                        ):
                            return True
            return False

        # Standard claims (metrics, experience, skills)
        if claim.status != ClaimStatus.CONFIRMED:
            return False
        if mastery is None:
            return False
        return (
            mastery.effective_score >= self.score_threshold
            and mastery.confidence >= self.confidence_threshold
        )

    def parse_job_description(self, jd_text: str) -> list[str]:
        """Extracts individual requirement lines/bullets from a Job Description."""
        lines = jd_text.strip().splitlines()
        requirements = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            # Strip bullet prefixes
            clean = re.sub(r"^[-*•\d\.\)]\s*", "", stripped).strip()
            if len(clean) >= 12:
                requirements.append(clean)
        return requirements

    def analyze_gaps(
        self,
        jd_requirements: list[str],
        claims: list[Claim],
        mastery_map: dict[str, MasteryState],
    ) -> GapReport:
        """Categorizes JD requirements into matched & defensible, needs practice, and unsupported."""
        gap_report = GapReport()

        if not claims:
            for req in jd_requirements:
                gap_report.unsupported_gaps.append(RequirementGap(
                    requirement_text=req,
                    status="unsupported_gap",
                    reason="No claims available in candidate profile.",
                ))
            return gap_report

        claim_texts = [f"{c.text} Tools: {', '.join(c.tools)} Metrics: {', '.join(c.metric_values)}" for c in claims]

        for req in jd_requirements:
            is_adv, adv_reasons = self.adversarial_detector.inspect(req)
            if is_adv:
                gap_report.unsupported_gaps.append(RequirementGap(
                    requirement_text=req,
                    status="unsupported_gap",
                    reason=f"Adversarial instruction detected: {'; '.join(adv_reasons)}. Excluded from tailoring.",
                ))
                continue

            ranked = self.matcher.rank_candidates(req, claim_texts)
            best_idx, best_score = ranked[0] if ranked else (-1, MatchScore(0, 0, 0, 0))

            if best_idx >= 0 and best_score.total >= self.match_cutoff:
                best_claim = claims[best_idx]
                m_state = mastery_map.get(best_claim.id)
                is_eligible = self.is_claim_eligible(best_claim, m_state, all_claims=claims, mastery_map=mastery_map)

                if is_eligible:
                    gap_report.matched_and_defensible.append(RequirementGap(
                        requirement_text=req,
                        status="matched_and_defensible",
                        supporting_claim_id=best_claim.id,
                        supporting_claim_text=best_claim.text,
                        effective_score=m_state.effective_score if m_state else None,
                        confidence=m_state.confidence if m_state else None,
                        match_score=best_score,
                        reason=f"Verified defensible with effective score {m_state.effective_score:.2f} >= {self.score_threshold}.",
                    ))
                else:
                    curr_score = m_state.effective_score if m_state else 0.0
                    curr_conf = m_state.confidence if m_state else 0.0
                    gap_report.needs_interview_practice.append(RequirementGap(
                        requirement_text=req,
                        status="needs_interview_practice",
                        supporting_claim_id=best_claim.id,
                        supporting_claim_text=best_claim.text,
                        effective_score=curr_score,
                        confidence=curr_conf,
                        match_score=best_score,
                        reason=f"Claim matches requirement (score {best_score.total:.2f}) but mastery is below threshold: score {curr_score:.2f} < {self.score_threshold} or conf {curr_conf:.2f} < {self.confidence_threshold}.",
                    ))
            else:
                gap_report.unsupported_gaps.append(RequirementGap(
                    requirement_text=req,
                    status="unsupported_gap",
                    match_score=best_score if best_idx >= 0 else None,
                    reason=f"No matching verified claim found in candidate profile (best match score {best_score.total:.2f} < {self.match_cutoff}).",
                ))

        return gap_report

    def tailor(
        self,
        jd_text: str,
        facts: list[ProfileFact],
        claims: list[Claim],
        mastery_map: dict[str, MasteryState],
    ) -> TailoringResult:
        """Assembles a tailored resume draft using ONLY mastery-eligible claims and confirmed facts."""
        jd_requirements = self.parse_job_description(jd_text)
        gap_report = self.analyze_gaps(jd_requirements, claims, mastery_map)

        # 1. Filter confirmed facts
        confirmed_facts = [f for f in facts if f.is_confirmed]
        fact_dict: dict[ProfileFactType, ProfileFact] = {f.fact_type: f for f in confirmed_facts}

        # 2. Filter eligible claims
        eligible_claims = [
            c for c in claims
            if self.is_claim_eligible(c, mastery_map.get(c.id), all_claims=claims, mastery_map=mastery_map)
        ]

        # Rank eligible claims by relevance to JD requirements
        ranked_claims: list[tuple[Claim, float]] = []
        if jd_requirements and eligible_claims:
            for c in eligible_claims:
                c_text = f"{c.text} {' '.join(c.tools)} {' '.join(c.metric_values)}"
                # Max score against any requirement
                max_score = max(
                    (self.matcher.match(req, c_text).total for req in jd_requirements),
                    default=0.0
                )
                ranked_claims.append((c, max_score))
            ranked_claims.sort(key=lambda x: x[1], reverse=True)
            chosen_claims = [c for c, _ in ranked_claims]
        else:
            chosen_claims = eligible_claims

        # 3. Build Markdown Resume
        md_lines = []
        attribution_trace: dict[str, str] = {}

        # Header / Name
        name_fact = fact_dict.get(ProfileFactType.NAME)
        name_val = name_fact.value if name_fact else "Candidate"
        md_lines.append(f"# {name_val}")
        attribution_trace[f"# {name_val}"] = "fact:name"

        # Contact line
        contact_parts = []
        if ProfileFactType.LOCATION in fact_dict:
            contact_parts.append(fact_dict[ProfileFactType.LOCATION].value)
        if ProfileFactType.EMAIL in fact_dict:
            contact_parts.append(fact_dict[ProfileFactType.EMAIL].value)
        if ProfileFactType.PHONE in fact_dict:
            contact_parts.append(fact_dict[ProfileFactType.PHONE].value)

        if contact_parts:
            contact_line = " | ".join(contact_parts)
            md_lines.append(contact_line)
            attribution_trace[contact_line] = "fact:contact"

        # Education Section
        if ProfileFactType.EDUCATION_INSTITUTION in fact_dict:
            md_lines.append("\n## EDUCATION")
            inst = fact_dict[ProfileFactType.EDUCATION_INSTITUTION].value
            deg = fact_dict.get(ProfileFactType.DEGREE).value if ProfileFactType.DEGREE in fact_dict else ""
            grad = fact_dict.get(ProfileFactType.GRADUATION_DATE).value if ProfileFactType.GRADUATION_DATE in fact_dict else ""
            cgpa = fact_dict.get(ProfileFactType.CGPA).value if ProfileFactType.CGPA in fact_dict else ""

            edu_line1 = f"**{inst}**"
            if grad:
                edu_line1 += f" - {grad}"
            md_lines.append(edu_line1)
            attribution_trace[edu_line1] = "fact:education"

            edu_line2 = deg
            if cgpa:
                edu_line2 += f" (CGPA: {cgpa})" if deg else f"CGPA: {cgpa}"
            if edu_line2:
                md_lines.append(edu_line2)
                attribution_trace[edu_line2] = "fact:degree"

        # Experience Section (only if employer confirmed)
        if ProfileFactType.EMPLOYER_NAME in fact_dict:
            md_lines.append("\n## EXPERIENCE")
            emp = fact_dict[ProfileFactType.EMPLOYER_NAME].value
            dates = fact_dict.get(ProfileFactType.EMPLOYMENT_DATES).value if ProfileFactType.EMPLOYMENT_DATES in fact_dict else ""
            exp_header = f"**{emp}**" + (f" ({dates})" if dates else "")
            md_lines.append(exp_header)
            attribution_trace[exp_header] = "fact:employer"

            # Add eligible experience claims
            exp_claims = [c for c in chosen_claims if c.project_id is None and c.claim_type != "skill"]
            for c in exp_claims:
                bullet = f"- {c.text}"
                md_lines.append(bullet)
                attribution_trace[bullet] = c.id

        # Projects Section
        project_claims: dict[str, list[Claim]] = {}
        for c in chosen_claims:
            if c.project_id:
                project_claims.setdefault(c.project_id, []).append(c)

        if project_claims:
            md_lines.append("\n## VERIFIED PROJECTS")
            for pid, pclaims in project_claims.items():
                pname = pid.replace("-", " ").title()
                pheader = f"### {pname}"
                md_lines.append(pheader)
                attribution_trace[pheader] = f"project:{pid}"

                tools_in_proj = [c for c in pclaims if getattr(c.claim_type, "value", str(c.claim_type)).lower() == "tool"]
                bullets_in_proj = [c for c in pclaims if getattr(c.claim_type, "value", str(c.claim_type)).lower() != "tool"]

                if tools_in_proj:
                    tech_str = ", ".join(sorted(set(t.text for t in tools_in_proj)))
                    stack_line = f"Technologies: {tech_str}"
                    md_lines.append(stack_line)
                    attribution_trace[stack_line] = f"project_tools:{pid}"

                for c in bullets_in_proj:
                    bullet = f"- {c.text}"
                    md_lines.append(bullet)
                    attribution_trace[bullet] = c.id

        # Skills Section
        skill_claims = [c for c in chosen_claims if c.claim_type == "skill" or c.project_id is None]
        all_tools = set()
        for c in chosen_claims:
            all_tools.update(c.tools)

        if all_tools or skill_claims:
            md_lines.append("\n## VERIFIED TECHNICAL SKILLS")
            unique_skills = sorted(list(all_tools.union(set(c.text for c in skill_claims if len(c.text.split()) <= 3))))
            skills_line = "- " + ", ".join(unique_skills)
            md_lines.append(skills_line)
            attribution_trace[skills_line] = "skills:aggregate"

        tailored_md = "\n".join(md_lines)

        return TailoringResult(
            tailored_markdown=tailored_md,
            gap_report=gap_report,
            included_claims=chosen_claims,
            included_facts=confirmed_facts,
            attribution_trace=attribution_trace,
        )
