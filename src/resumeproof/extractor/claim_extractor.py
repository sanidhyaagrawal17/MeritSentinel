"""Structured claim and profile fact extractor using LLM structured outputs."""

import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field

from resumeproof.core.llm import LLMClient
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus, ClaimAudit
from resumeproof.extractor.pdf_parser import ParseResult

ProfileFactTypeLiteral = Literal[
    "name",
    "email",
    "phone",
    "location",
    "education_institution",
    "degree",
    "cgpa",
    "graduation_date",
    "employer_name",
    "employment_dates",
]

ClaimTypeLiteral = Literal[
    "skill",
    "tool",
    "project_metric",
    "experience_claim",
    "achievement",
]

class RawProfileFactExtraction(BaseModel):
    fact_type: ProfileFactTypeLiteral = Field(
        description="One of: name, email, phone, location, education_institution, degree, cgpa, graduation_date, employer_name, employment_dates"
    )
    value: str = Field(description="The exact extracted fact value")
    source_line: Optional[int] = Field(default=None, description="Line number where this fact appears")

class RawClaimExtraction(BaseModel):
    claim_type: ClaimTypeLiteral = Field(
        description="One of: skill, tool, project_metric, experience_claim, achievement"
    )
    project_id: Optional[str] = Field(default=None, description="Identifier of the associated project, if applicable")
    text: str = Field(description="The full extracted claim statement verbatim")
    tools: list[str] = Field(default_factory=list, description="Specific tools, libraries, or protocols mentioned")
    metric_values: list[str] = Field(default_factory=list, description="Numeric metrics or benchmarks mentioned (e.g., '45,000 req/sec', '1.13ms')")
    source_line: Optional[int] = Field(default=None, description="Line number where this claim appears")

class FullExtractionPayload(BaseModel):
    profile_facts: list[RawProfileFactExtraction] = Field(default_factory=list)
    claims: list[RawClaimExtraction] = Field(default_factory=list)

import re

EXTRACTION_SYSTEM_PROMPT = """You are an adversarial resume auditor.
Extract all claims and facts from the provided numbered resume lines into two distinct categories:

CATEGORY 1: profile_facts (ONLY the candidate's personal details and credentials)
- name, email, phone, location, education_institution, degree, cgpa, graduation_date, employer_name, employment_dates.
- CRITICAL INVARIANT: Profile facts must NEVER be inferred or guessed.
- employer_name and employment_dates ONLY exist if the resume explicitly contains an Experience, Employment, or Work Experience section with verified corporate positions. If there is NO Experience section, do NOT output employer_name or employment_dates.
- NEVER put skills, projects, or metrics in profile_facts.
- Use exact lowercase enum keys: 'name', 'email', 'phone', 'location', 'education_institution', 'degree', 'cgpa', 'graduation_date', 'employer_name', 'employment_dates'.

CATEGORY 2: claims (ALL technical bullet points and skills)
- Every single bullet point under Projects or Experience MUST be extracted as a claim with its FULL sentence text verbatim. Never truncate into short fragments.
- Bullet points under Projects or Experience MUST NEVER be typed as 'skill' or 'tool'. They must always be typed as 'project_metric' (if containing numbers, benchmarks, or throughput) or 'experience_claim'.
- For claims listed under a specific project, assign the project_id matching the project title heading above the bullet.
- In the Skills section, extract EACH individual skill, tool, language, or technology as its OWN individual claim with claim_type='skill' (e.g. split 'Languages: Python, SQL' into one claim for 'Python' and one claim for 'SQL'; split 'Tools: Git, Linux, Docker' into individual claims for 'Git', 'Linux', and 'Docker').
- Use exact lowercase enum keys: 'skill', 'tool', 'project_metric', 'experience_claim', 'achievement'.

Always map each fact and claim to its exact line number in the source text.
"""

def normalize_fact_type(raw_type: str, value: str) -> ProfileFactType:
    """Robustly normalizes raw LLM fact type string to ProfileFactType enum."""
    clean = raw_type.lower().strip().replace(" ", "_").replace("-", "_")
    try:
        return ProfileFactType(clean)
    except ValueError:
        pass

    # Heuristic fallback based on string content and value patterns
    if "email" in clean or "@" in value:
        return ProfileFactType.EMAIL
    if "phone" in clean or (any(c.isdigit() for c in value) and any(p in value for p in ["(", ")", "-", "+"])):
        return ProfileFactType.PHONE
    if "location" in clean or "city" in clean or "address" in clean or ("," in value and any(st in value for st in ["IL", "CA", "NY", "TX", "WA", "CO"])):
        return ProfileFactType.LOCATION
    if "institution" in clean or "school" in clean or "university" in clean or "college" in clean or "tech" in value.lower():
        return ProfileFactType.EDUCATION_INSTITUTION
    if "degree" in clean or "b.tech" in value.lower() or "b.s." in value.lower() or "m.s." in value.lower():
        return ProfileFactType.DEGREE
    if "cgpa" in clean or "gpa" in clean or "/ 10" in value or "/ 4" in value:
        return ProfileFactType.CGPA
    if "date" in clean or "graduation" in clean or any(m in value.lower() for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
        return ProfileFactType.GRADUATION_DATE
    if "employer" in clean or "company" in clean:
        return ProfileFactType.EMPLOYER_NAME
    if "employment" in clean:
        return ProfileFactType.EMPLOYMENT_DATES

    return ProfileFactType.NAME

def normalize_claim_type(raw_type: str) -> ClaimType:
    """Robustly normalizes raw LLM claim type string to ClaimType enum."""
    clean = raw_type.lower().strip().replace(" ", "_").replace("-", "_")
    try:
        return ClaimType(clean)
    except ValueError:
        pass

    if "metric" in clean or "benchmark" in clean or "number" in clean:
        return ClaimType.PROJECT_METRIC
    if "skill" in clean:
        return ClaimType.SKILL
    if "tool" in clean:
        return ClaimType.TOOL
    if "experience" in clean or "project" in clean:
        return ClaimType.EXPERIENCE_CLAIM
    if "achieve" in clean:
        return ClaimType.ACHIEVEMENT

    return ClaimType.PROJECT_METRIC

def _split_skill_text(raw_text: str) -> list[str]:
    """Splits a skill text (e.g. 'Languages: Python, SQL' or 'FastAPI, SQLite, Sockets') into individual skill names."""
    txt_heading = raw_text.lstrip("#").strip().upper().rstrip(":")
    if txt_heading in ("SKILLS", "TECHNICAL SKILLS", "LANGUAGES", "TOOLS", "NETWORKING & BACKEND", "DISTRIBUTED SYSTEMS", "ML & DATA"):
        return []

    if ":" in raw_text:
        raw_text = raw_text.split(":", 1)[1]
    raw_skills = re.split(r"[,;•|]+", raw_text)
    clean_skills = []
    for s in raw_skills:
        clean = s.strip()
        clean = re.sub(r"^[-*•#\s]+", "", clean).strip()
        clean = re.sub(r"^(and\s+)", "", clean, flags=re.IGNORECASE).strip()
        clean = clean.rstrip(".,;")
        if clean.upper() in ("SKILLS", "TECHNICAL SKILLS", "LANGUAGES", "TOOLS", "NETWORKING & BACKEND", "DISTRIBUTED SYSTEMS", "ML & DATA"):
            continue
        if clean and len(clean) >= 2 and not clean.lower().startswith("claim:"):
            clean_skills.append(clean)
    return clean_skills

def _map_resume_lines(parse_result: ParseResult) -> tuple[dict[int, str], dict[int, str], bool]:
    """Maps line numbers to active sections and project IDs, and detects if experience section exists."""
    line_sections: dict[int, str] = {}
    line_projects: dict[int, str] = {}
    current_section = "GENERAL"
    current_project = None
    has_experience_section = False

    for line in parse_result.lines:
        txt = line.text.strip()
        txt_upper = txt.lstrip("#").strip().upper().rstrip(":")

        if txt_upper in ("EDUCATION", "PROJECTS", "SKILLS", "TECHNICAL SKILLS", "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "PROFESSIONAL EXPERIENCE"):
            current_section = txt_upper
            current_project = None
            line_sections[line.line_number] = current_section
            if current_section in ("EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "PROFESSIONAL EXPERIENCE"):
                has_experience_section = True
            continue

        if current_section == "PROJECTS":
            # Identify project name heading lines like "Task-Tracker API (Python, FastAPI, SQLite):"
            if "(" in txt and (txt.endswith(":") or not txt.startswith("-")):
                proj_name = txt.split("(")[0].strip()
                current_project = re.sub(r"[^a-zA-Z0-9]+", "-", proj_name.lower()).strip("-")
            elif txt.endswith(":") and not txt.startswith("-"):
                current_project = re.sub(r"[^a-zA-Z0-9]+", "-", txt[:-1].lower()).strip("-")

        line_sections[line.line_number] = current_section
        if current_project:
            line_projects[line.line_number] = current_project

    return line_sections, line_projects, has_experience_section

def deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """Deduplicates claims having identical text or duplicate tools plus metrics."""
    unique: list[Claim] = []
    seen_texts: set[str] = set()
    seen_tools_metrics: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for c in claims:
        clean_text = c.text.strip().lower()
        if clean_text in seen_texts:
            continue

        # If tools and metrics are both non-empty and identical to a previous claim, consider duplicate
        tools_tuple = tuple(sorted(t.strip().lower() for t in c.tools))
        metrics_tuple = tuple(sorted(m.strip().lower() for m in c.metric_values))
        if tools_tuple and metrics_tuple:
            tm_key = (tools_tuple, metrics_tuple)
            if tm_key in seen_tools_metrics:
                continue
            seen_tools_metrics.add(tm_key)

        seen_texts.add(clean_text)
        unique.append(c)

    return unique

def extract_from_resume(
    parse_result: ParseResult,
    llm_client: LLMClient,
    persona_id: Optional[str] = None,
    user_id: str = "default_user",
    model: str = "qwen2.5:3b",
) -> tuple[list[ProfileFact], list[Claim]]:
    """Extracts Profile Facts and Mastery Claims from parsed resume lines."""
    line_sections, line_projects, has_experience_section = _map_resume_lines(parse_result)
    numbered_text = "\n".join(f"[{line.line_number}] {line.text}" for line in parse_result.lines)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume Text with Line Numbers:\n\n{numbered_text}"}
    ]

    raw_data = llm_client.structured_completion(
        messages=messages,
        schema=FullExtractionPayload,
        model=model,
        temperature=0.0
    )

    extracted_payload = FullExtractionPayload.model_validate(raw_data)

    CLAIM_TYPES_SET = {"skill", "tool", "project_metric", "experience_claim", "achievement", "metric"}
    raw_claims_list = list(extracted_payload.claims)

    facts: list[ProfileFact] = []
    for item in extracted_payload.profile_facts:
        clean_ft = str(item.fact_type).lower().strip().replace(" ", "_").replace("-", "_")
        if clean_ft in CLAIM_TYPES_SET or any(ct in clean_ft for ct in ["metric", "skill", "achievement", "project"]):
            # Move to claims!
            raw_claims_list.append(RawClaimExtraction(
                claim_type=clean_ft,
                text=item.value,
                source_line=item.source_line,
            ))
            continue

        ftype = normalize_fact_type(str(item.fact_type), item.value)
        # Invariant: employer_name and employment_dates ONLY exist if the resume has an experience section
        if not has_experience_section and ftype in (ProfileFactType.EMPLOYER_NAME, ProfileFactType.EMPLOYMENT_DATES):
            continue

        # Ignore placeholder values like "not provided"
        if item.value.lower().strip() in ("not provided", "none", "n/a", "null", ""):
            continue

        facts.append(ProfileFact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            persona_id=persona_id,
            fact_type=ftype,
            value=item.value,
            source_line=item.source_line,
            source_page=1,
            is_confirmed=False
        ))

    claims: list[Claim] = []

    for item in raw_claims_list:
        ctype = normalize_claim_type(str(item.claim_type))

        # Check section context
        in_skills_section = item.source_line is not None and line_sections.get(item.source_line) in ("SKILLS", "TECHNICAL SKILLS")
        in_project_or_exp_section = item.source_line is not None and line_sections.get(item.source_line) in (
            "PROJECTS", "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "PROFESSIONAL EXPERIENCE"
        )

        # Associate project_id and exact source_line from parse_result
        project_id = item.project_id
        if project_id and project_id.startswith("name_"):
            project_id = project_id.replace("name_", "")
        if project_id:
            project_id = re.sub(r"[^a-zA-Z0-9]+", "-", project_id.lower()).strip("-")

        claim_words = set(re.findall(r"[a-zA-Z0-9]+", item.text.lower()))
        best_line_num = None
        best_overlap = 0
        for line in parse_result.lines:
            lw = set(re.findall(r"[a-zA-Z0-9]+", line.text.lower()))
            overlap = len(claim_words.intersection(lw))
            if overlap > best_overlap:
                best_overlap = overlap
                best_line_num = line.line_number

        if best_line_num is not None and best_overlap >= 3:
            item.source_line = best_line_num
            if best_line_num in line_projects:
                project_id = line_projects[best_line_num]
        elif item.source_line is not None and item.source_line in line_projects:
            project_id = line_projects[item.source_line]

        # CRITICAL INVARIANT: experience or project claims must NEVER be typed as skill or tool
        if in_project_or_exp_section or project_id:
            if ctype in (ClaimType.SKILL, ClaimType.TOOL):
                if item.metric_values or any(char.isdigit() for char in item.text):
                    ctype = ClaimType.PROJECT_METRIC
                else:
                    ctype = ClaimType.EXPERIENCE_CLAIM

        # CRITICAL INVARIANT: split skill lines into one skill claim per skill
        if in_skills_section or (ctype == ClaimType.SKILL and not project_id and not in_project_or_exp_section):
            # Split skill line into individual skills
            skills_list = _split_skill_text(item.text)
            if not skills_list and item.tools:
                skills_list = [t.strip() for t in item.tools if t.strip()]
            if not skills_list:
                skills_list = [item.text.strip()]

            for skill_name in skills_list:
                claim_id = str(uuid.uuid4())
                audit = ClaimAudit(
                    extractor_text=skill_name,
                    extractor_tools=[skill_name],
                    extractor_metrics=[],
                    action_taken=ClaimStatus.EXTRACTED
                )
                claims.append(Claim(
                    id=claim_id,
                    user_id=user_id,
                    persona_id=persona_id,
                    project_id=None,
                    claim_type=ClaimType.SKILL,
                    text=skill_name,
                    tools=[skill_name],
                    metric_values=[],
                    source_line=item.source_line,
                    source_page=1,
                    status=ClaimStatus.EXTRACTED,
                    audit=audit
                ))
            continue

        # Regular non-skill claim
        claim_id = str(uuid.uuid4())
        audit = ClaimAudit(
            extractor_text=item.text,
            extractor_tools=item.tools,
            extractor_metrics=item.metric_values,
            action_taken=ClaimStatus.EXTRACTED
        )

        claims.append(Claim(
            id=claim_id,
            user_id=user_id,
            persona_id=persona_id,
            project_id=project_id,
            claim_type=ctype,
            text=item.text,
            tools=item.tools,
            metric_values=item.metric_values,
            source_line=item.source_line,
            source_page=1,
            status=ClaimStatus.EXTRACTED,
            audit=audit
        ))

    # Fallback: ensure all skills listed in resume Skills sections are parsed as individual skill claims
    for line in parse_result.lines:
        txt_heading = line.text.lstrip("#").strip().upper().rstrip(":")
        if txt_heading in ("SKILLS", "TECHNICAL SKILLS", "LANGUAGES", "TOOLS", "NETWORKING & BACKEND", "DISTRIBUTED SYSTEMS", "ML & DATA"):
            continue
        if line_sections.get(line.line_number) in ("SKILLS", "TECHNICAL SKILLS"):
            split_skills = _split_skill_text(line.text)
            for skill_name in split_skills:
                if skill_name.upper() in ("SKILLS", "TECHNICAL SKILLS", "LANGUAGES", "TOOLS"):
                    continue
                if not any(c.claim_type == ClaimType.SKILL and c.text.lower() == skill_name.lower() for c in claims):
                    cid = str(uuid.uuid4())
                    claims.append(Claim(
                        id=cid,
                        user_id=user_id,
                        persona_id=persona_id,
                        project_id=None,
                        claim_type=ClaimType.SKILL,
                        text=skill_name,
                        tools=[skill_name],
                        metric_values=[],
                        source_line=line.line_number,
                        source_page=1,
                        status=ClaimStatus.EXTRACTED,
                        audit=ClaimAudit(
                            extractor_text=skill_name,
                            extractor_tools=[skill_name],
                            extractor_metrics=[],
                            action_taken=ClaimStatus.EXTRACTED
                        )
                    ))

    # Bullet fallback: ensure all bullets in project/experience sections are captured
    for line in parse_result.lines:
        sec = line_sections.get(line.line_number)
        if sec in ("PROJECTS", "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "PROFESSIONAL EXPERIENCE"):
            txt = line.text.strip()
            if txt.startswith(("-", "*", "•")) or txt.lower().startswith("claim:"):
                clean_bullet = re.sub(r"^[-*•]\s*", "", txt).strip()
                clean_bullet = re.sub(r"^claim:\s*", "", clean_bullet, flags=re.IGNORECASE).strip()
                if len(clean_bullet) >= 10:
                    bullet_words = set(re.findall(r"[a-zA-Z0-9]+", clean_bullet.lower()))
                    already_captured = any(
                        len(bullet_words.intersection(set(re.findall(r"[a-zA-Z0-9]+", c.text.lower())))) >= max(3, len(bullet_words) * 3 // 4)
                        for c in claims
                    )
                    if not already_captured:
                        proj_id = line_projects.get(line.line_number)
                        ctype = ClaimType.PROJECT_METRIC if any(ch.isdigit() for ch in clean_bullet) else ClaimType.EXPERIENCE_CLAIM
                        claims.append(Claim(
                            id=str(uuid.uuid4()),
                            user_id=user_id,
                            persona_id=persona_id,
                            project_id=proj_id,
                            claim_type=ctype,
                            text=clean_bullet,
                            source_line=line.line_number,
                            source_page=1,
                            status=ClaimStatus.EXTRACTED,
                            audit=ClaimAudit(
                                extractor_text=clean_bullet,
                                extractor_tools=[],
                                extractor_metrics=[],
                                action_taken=ClaimStatus.EXTRACTED
                            )
                        ))

    # Apply strict deduplication (same source line or same tools + metrics, preserving individual skills)
    deduped_claims = deduplicate_claims(claims)

    return facts, deduped_claims

