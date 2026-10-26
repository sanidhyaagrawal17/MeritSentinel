"""Strict Deterministic Verifier: Zero LLM in the blocking path."""

from dataclasses import dataclass, field
import re
from typing import Optional

from resumeproof.schemas.claims import Claim
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.verifier.injection import (
    normalize_unicode_homoglyphs,
    normalize_words_to_numbers,
    AdversarialDetector,
)

# Known technical tools and languages for word-boundary matching
KNOWN_TOOLS = [
    "C++", "C#", "C", "Java", "JavaScript", "TypeScript", "Python", "Go", "Golang",
    "Rust", "Kotlin", "Swift", "Scala", "PHP", "Ruby", "Perl", "Lua", "SQL", "Haskell",
    "FastAPI", "Flask", "Django", "Spring", "Node.js", "Express", "React",
    "PostgreSQL", "MySQL", "SQLite", "DuckDB", "Redis", "MongoDB",
    "Kafka", "Raft", "Paxos", "Docker", "Kubernetes", "Istio",
    "AWS", "AWS EC2", "GCP", "Azure", "Linux", "POSIX", "gRPC", "Protobuf", "ONNX",
    "LightGBM", "PyTorch", "TensorFlow", "Scikit-Learn", "Jepsen", "Locust",
    "ONNX Runtime", "Ultralytics YOLO11", "YOLO11", "NetworkX", "Cisco Meraki API",
    "JWT", "JSON-RPC 2.0", "JSON-RPC", "Model Context Protocol", "MCP",
    "ChromaDB", "Next.js", "Next.js 14", "Pydantic", "Postman", "Git", "GitHub",
    "Apache Spark", "Spark", "Databricks", "Snowflake", "BigQuery",
    "Jenkins", "Milvus", "TensorRT", "BGP", "CUDA", "Google Antigravity (Gemini)", "Google Antigravity", "Gemini",
]

TOOL_ALIASES: dict[str, list[str]] = {
    "mcp": ["model context protocol"],
    "model context protocol": ["mcp"],
    "yolo11": ["ultralytics yolo11"],
    "ultralytics yolo11": ["yolo11"],
    "next.js": ["next.js 14"],
    "next.js 14": ["next.js"],
    "json-rpc": ["json-rpc 2.0"],
    "json-rpc 2.0": ["json-rpc"],
    "rest": ["restful apis", "rest api", "rest apis"],
    "restful apis": ["rest", "rest api", "rest apis"],
    "rag": ["retrieval-augmented generation", "retrieval-augmented generation (rag)"],
    "retrieval-augmented generation (rag)": ["rag"],
    "google antigravity": ["google antigravity (gemini)"],
    "google antigravity (gemini)": ["google antigravity"],
}

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
    # Mask numbers that are part of known tool names (e.g. JSON-RPC 2.0, Next.js 14, YOLO11)
    masked_text = text
    for tool_with_digit in ["JSON-RPC 2.0", "Next.js 14", "Ultralytics YOLO11", "YOLO11", "HTTP/2", "HTTP/3", "OAuth 2.0"]:
        masked_text = re.sub(re.escape(tool_with_digit), " " * len(tool_with_digit), masked_text, flags=re.IGNORECASE)

    patterns = [
        r"\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:req/sec|reqs/sec|requests/sec|rps|qps)\b",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s*(?:ms|seconds|minutes)\b",
        r"\b\d+(?:,\d+)*(?:\.\d+)?\b",
    ]
    matches = []
    seen_spans = set()
    for p in patterns:
        for m in re.finditer(p, masked_text, re.IGNORECASE):
            span = m.span()
            if not any(span[0] >= s[0] and span[1] <= s[1] for s in seen_spans):
                seen_spans.add(span)
                matches.append(normalize_number_string(m.group(0)))
    return matches



def extract_tools_from_text(text: str) -> list[str]:
    """Extracts known technical tools using longest-match span subsumption to prevent root subtoken leakage."""
    extracted = []
    occupied_spans: list[tuple[int, int]] = []

    # Sort tools by length descending so longer composite names claim spans first
    sorted_tools = sorted(KNOWN_TOOLS, key=len, reverse=True)

    for tool_name in sorted_tools:
        reg = TOOL_REGEX_MAP[tool_name]
        for m in reg.finditer(text):
            span = m.span()
            # If this match is subsumed by an already matched longer tool span, skip
            if any(span[0] >= occ[0] and span[1] <= occ[1] for occ in occupied_spans):
                continue
            occupied_spans.append(span)
            if tool_name not in extracted:
                extracted.append(tool_name)

    return extracted


COMMON_RESUME_WORDS = {
    # Pronouns, Prepositions, Conjunctions
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "from", "by", "of", "as",
    "into", "through", "across", "under", "over", "between", "after", "before", "while", "during", "using",
    "via", "we", "our", "i", "my", "this", "that", "these", "those", "both", "all", "each", "every",
    # Resume Action Verbs
    "built", "engineered", "designed", "trained", "benchmarked", "exported", "configured", "architected",
    "deployed", "developed", "implemented", "achieved", "optimized", "evaluated", "parsed", "resolved",
    "maintained", "conducted", "applied", "generated", "integrated", "executed", "sustained", "profiled",
    "scaled", "partitioned", "monitored", "delivered", "authored", "created", "spearheaded", "collaborated",
    "tested", "verified", "automated", "retained", "attained", "demonstrated", "served", "handled", "reached",
    "clocked", "dropped", "gained", "tuned", "reverted", "switched", "introduced", "structured", "streamlined",
    "processed", "dispatched", "scraped", "persisted", "modeled", "queried", "decoupled", "constructed",
    "managed", "secured", "indexed", "isolated", "adhering", "unpacking", "decoding", "following",
    # Resume Section & Fact Nouns
    "bachelor", "master", "doctor", "science", "engineering", "technology", "university", "college",
    "school", "institute", "india", "expected", "june", "july", "august", "september", "october",
    "november", "december", "january", "february", "march", "april", "may", "cgpa", "gpa", "degree",
    "education", "experience", "projects", "skills", "summary", "profile", "overview", "work", "history",
    # Standard Technical Concepts / English Words (NOT specific unverified tool/framework packages)
    "computer", "technical", "systems", "system", "service", "services", "backend", "frontend", "fullstack",
    "pipeline", "pipelines", "dataset", "datasets", "benchmark", "benchmarks", "surface", "defect", "defects",
    "detection", "model", "models", "architecture", "architectures", "variant", "variants", "baseline",
    "baselines", "feature", "features", "layer", "layers", "network", "networks", "protocol", "protocols",
    "format", "formats", "specification", "specifications", "spec", "specs", "mode", "modes", "database",
    "databases", "connection", "connections", "pool", "pools", "pooling", "query", "queries", "header",
    "headers", "record", "records", "traffic", "packet", "packets", "socket", "sockets", "rule", "rules",
    "state", "states", "status", "index", "indices", "indexing", "search", "tool", "tools", "metric",
    "metrics", "score", "scores", "split", "splits", "train", "training", "validation", "test", "testing",
    "single", "multi", "high", "low", "real", "time", "level", "cloud", "web", "full", "core", "code",
    "data", "line", "lines", "bullet", "bullets", "node", "nodes", "cluster", "clusters", "instance",
    "instances", "client", "clients", "server", "servers", "api", "apis", "rest", "restful", "endpoint",
    "endpoints", "app", "apps", "application", "applications", "module", "modules", "script", "scripts",
    "throughput", "latency", "inference", "execution", "rate", "limit", "limiter", "limiting", "window",
    "sliding", "fixed", "token", "bucket", "leaky", "algorithm", "algorithms", "structure", "structures",
    "binary", "hex", "byte", "bytes", "octet", "octets", "bitwise", "memory", "storage", "disk", "file",
    "files", "directory", "directories", "log", "logs", "logging", "journal", "journaling", "lock", "locks",
    "locking", "concurrency", "concurrent", "thread", "threads", "process", "processes", "async", "sync",
    "synchronous", "asynchronous", "event", "events", "loop", "queue", "queues", "message", "messages",
    "worker", "workers", "task", "tasks", "tracker", "tracking", "flow", "workflow", "workflows", "job",
    "jobs", "batch", "stream", "streaming", "load", "balancer", "balancing", "proxy", "reverse", "gateway",
    "route", "routes", "routing", "router", "routers", "sniffer", "sniffing", "capture", "capturing",
    "filter", "filtering", "nat", "vlan", "subnet", "subnetting", "ip", "ipv4", "ipv6", "tcp", "udp",
    "icmp", "arp", "dns", "dhcp", "http", "https", "ssl", "tls", "raw", "checksum", "crc",
    "payload", "segment", "datagram", "frame", "frames", "bit", "bits", "flag", "flags", "ack", "syn",
    "fin", "rst", "psh", "urg", "ttl", "tos", "mac", "address", "addresses", "resolution",
    "resolver", "recursive", "iterative", "authoritative", "root", "tld", "domain", "nameserver", "lookup",
    "lookups", "forward", "zone", "zones", "rr", "cname", "txt", "mx", "soa", "ptr", "srv", "ns",
    "wire", "unpacking", "packing", "struct", "decode", "decoder", "decoding", "encode", "encoder",
    "encoding", "parser", "parsers", "parsing", "ast", "tree", "graph", "graphs", "entity", "relationships",
    "relational", "columnar", "distributed", "serverless", "reproducible", "fleet", "broker", "brokers",
    "exchanges", "queuing", "depth", "canary", "continuous", "delivery", "gitops", "declarative",
    "technologies", "rendered", "validated", "provisioned", "orchestrated", "wi", "fi", "acid", "compliant",
    "retrieval", "augmented", "generation", "embeddings", "semantic",
}

def extract_unbacked_technical_terms(
    sentence: str,
    eligible_claims: list[Claim],
    confirmed_facts: list[ProfileFact],
) -> list[str]:
    """Detects unbacked technical tools, packages, and frameworks outside KNOWN_TOOLS.
    Removes the closed-world assumption by identifying capitalized technical tokens,
    camelCase/PascalCase words, or specific CLI/package names that do not match
    standard resume English vocabulary and are not grounded in any eligible claim or confirmed fact.
    """
    import re
    allowed_tokens = set()
    for c in eligible_claims:
        allowed_tokens.update(re.findall(r"[a-zA-Z0-9]+", c.text.lower()))
        for t in c.tools:
            allowed_tokens.update(re.findall(r"[a-zA-Z0-9]+", t.lower()))
        if c.project_id:
            allowed_tokens.update(re.findall(r"[a-zA-Z0-9]+", c.project_id.lower()))
    for f in confirmed_facts:
        allowed_tokens.update(re.findall(r"[a-zA-Z0-9]+", f.value.lower()))

    # Known candidate tools outside KNOWN_TOOLS (e.g. Terraform, Ansible, Celery, Airflow, dbt, Flink, Cassandra, Neo4j, etc.)
    CANDIDATE_OUTSIDE_TOOLS = [
        "terraform", "ansible", "celery", "airflow", "dbt", "flink", "cassandra", "neo4j",
        "rabbitmq", "prometheus", "grafana", "consul", "vault", "elasticsearch", "opensearch",
        "keda", "spinnaker", "argocd", "supabase", "clickhouse",
    ]

    unbacked = []
    
    # 1. Match specific outside tools
    for tool_token in CANDIDATE_OUTSIDE_TOOLS:
        m = re.search(rf"\b{re.escape(tool_token)}\b", sentence, re.IGNORECASE)
        if m:
            matched_text = sentence[m.start():m.end()]
            if tool_token.lower() not in allowed_tokens and matched_text not in unbacked:
                unbacked.append(matched_text)

    # 2. Match general PascalCase/CamelCase/Capitalized technical tokens
    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", sentence)
    for tok in tokens:
        tl = tok.lower()
        if len(tok) < 3:
            continue
        if tl in allowed_tokens or tl in COMMON_RESUME_WORDS:
            continue
        # If token contains hyphens, check if all subparts are allowed
        if "-" in tok:
            subparts = [p.lower() for p in tok.split("-") if p]
            if all(sp in allowed_tokens or sp in COMMON_RESUME_WORDS for sp in subparts):
                continue
        # Capitalized token or PascalCase
        if tok[0].isupper() and (any(c.isupper() for c in tok[1:]) or tl not in COMMON_RESUME_WORDS):
            if tok not in unbacked:
                unbacked.append(tok)

    return unbacked


class StrictVerifier:
    """Deterministic Verifier with zero LLM in the blocking path."""

    def __init__(self, allow_unverified_profile_headers: bool = True):
        self.allow_headers = allow_unverified_profile_headers
        self.adversarial_detector = AdversarialDetector()

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
        # 0. Check for adversarial injection or homoglyph evasion
        is_adv, adv_reasons = self.adversarial_detector.inspect(sentence)
        if is_adv:
            return SentenceVerificationResult(
                sentence_idx=sentence_idx,
                sentence_text=sentence,
                status="BLOCKED",
                supporting_claim_id=None,
                extracted_entities={},
                blocking_reasons=adv_reasons,
            )

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
        matching_fact: Optional[ProfileFact] = None
        for f in confirmed_facts:
            f_norm = normalize_number_string(f.value)
            if f.value.lower() in sentence.lower() or f_norm in normalized_sentence.lower():
                matching_fact = f
                break

        if matching_fact:
            # When matching a profile fact, verify that the sentence does not introduce
            # conflicting numbers (e.g. fabricated CGPA, inflated dates) or ungrounded metrics
            sent_numbers = extract_numbers_and_metrics(normalized_sentence)
            if sent_numbers:
                valid_numbers = set()
                for cf in confirmed_facts:
                    for n in extract_numbers_and_metrics(normalize_words_to_numbers(cf.value)):
                        valid_numbers.add(n)
                    valid_numbers.add(normalize_number_string(cf.value))
                for c in eligible_claims:
                    for n in extract_numbers_and_metrics(normalize_words_to_numbers(f"{c.text} {' '.join(c.metric_values)}")):
                        valid_numbers.add(n)

                # Check specifically for contradictory CGPA
                cgpa_match = re.search(r"\b(?:cgpa|gpa)\s*[:=]?\s*(\d+(?:\.\d+)?)\b", sentence, re.IGNORECASE)
                if cgpa_match:
                    cgpa_val = cgpa_match.group(1)
                    confirmed_cgpa = next((cf.value for cf in confirmed_facts if cf.fact_type == ProfileFactType.CGPA), None)
                    if confirmed_cgpa and cgpa_val not in confirmed_cgpa:
                        reasons.append(f"Unbacked or conflicting CGPA: '{cgpa_val}' does not match confirmed CGPA '{confirmed_cgpa}'.")

                for num in sent_numbers:
                    if num not in valid_numbers and not any(num in vn for vn in valid_numbers):
                        reasons.append(f"Unbacked or derived/rounded number: '{num}' does not match verified profile facts or claims.")

            if not reasons:
                return SentenceVerificationResult(
                    sentence_idx=sentence_idx,
                    sentence_text=sentence,
                    status="VERIFIED",
                    supporting_claim_id=f"fact:{matching_fact.fact_type.value}",
                    extracted_entities={"fact_type": matching_fact.fact_type.value, "fact_value": matching_fact.value},
                )
            else:
                return SentenceVerificationResult(
                    sentence_idx=sentence_idx,
                    sentence_text=sentence,
                    status="BLOCKED",
                    supporting_claim_id=None,
                    extracted_entities={"numbers": sent_numbers},
                    blocking_reasons=reasons,
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
            union = len(clean_sent_words.union(claim_words))
            jaccard = overlap / union if union > 0 else 0.0

            # Exact full sentence or clean bullet match gives 2.0
            clean_s = re.sub(r"^[-*•]\s*", "", sentence).strip().lower()
            clean_c = claim.text.strip().lower()
            if clean_s == clean_c:
                score = 2.0
            elif len(clean_c) > 15 and (clean_c in clean_s or clean_s in clean_c):
                score = 1.0 + jaccard
            else:
                score = jaccard

            if score > max_overlap_score:
                max_overlap_score = score
                best_match_claim = claim

        if not best_match_claim or max_overlap_score < 0.30:
            # Check if this sentence is a list of verified skills or project tools (e.g. "FastAPI, Python, SQLite" or "Technologies: Go, Redis")
            # Invariant: Skill/tool listings cannot contain metric numbers (e.g. 5,000 req/sec)
            is_skill_listing = (
                not extracted_numbers
                and ("," in sentence or len(extracted_tools) >= 2 or sentence.lower().startswith("technologies:") or sentence.lower().startswith("tools:"))
                and extracted_tools
            )
            if is_skill_listing:
                all_eligible_tools = set()
                for c in eligible_claims:
                    all_eligible_tools.update(t.lower() for t in c.tools)
                    for t_name in extract_tools_from_text(c.text):
                        all_eligible_tools.add(t_name.lower())
                    ctype_val = getattr(c, "claim_type", None)
                    if ctype_val and getattr(ctype_val, "value", str(ctype_val)).lower() in ("skill", "tool"):
                        all_eligible_tools.add(c.text.strip().lower())

                unverified_tools = []
                for t in extracted_tools:
                    tl = t.lower()
                    aliases = TOOL_ALIASES.get(tl, [])
                    if tl not in all_eligible_tools and not any(a in all_eligible_tools for a in aliases):
                        unverified_tools.append(t)

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

            unbacked_outside = extract_unbacked_technical_terms(sentence, eligible_claims, confirmed_facts)
            for ut in unbacked_outside:
                reasons.append(f"Unbacked technical term or package token: '{ut}' is not present in supporting claim or verified skill set.")
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
            aliases = TOOL_ALIASES.get(t_lower, [])
            valid_names = {t_lower} | set(aliases)
            in_claim = any(vn in claim_tools or any(vn in ct.lower() for ct in claim_tools) or TOOL_REGEX_MAP[t].search(best_match_claim.text) for vn in valid_names)
            if not in_claim:
                # Check if tool is confirmed in another claim
                in_other_eligible = any(any(vn in (ct.lower() for ct in c.tools) or vn in c.text.lower() for vn in valid_names) for c in eligible_claims)
                if not in_other_eligible:
                    reasons.append(f"Unbacked tool: '{t}' is not present in supporting claim or verified skill set.")

        # Open-world check: Check for unbacked technical terms, packages, or frameworks outside KNOWN_TOOLS
        unbacked_outside = extract_unbacked_technical_terms(sentence, eligible_claims, confirmed_facts)
        for ut in unbacked_outside:
            # Avoid duplicate reporting if already flagged
            if not any(ut.lower() in r.lower() for r in reasons):
                reasons.append(f"Unbacked technical term or package token: '{ut}' is not present in supporting claim or verified skill set.")

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
