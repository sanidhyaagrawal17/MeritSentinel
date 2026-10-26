"""Script to evaluate Real Ollama 7B Grader Discrimination on 5 Rubric Items x 4 Answer Tiers.

Tests:
- Strong (Tier 4)
- Terse (Tier 2-3)
- Generic (Tier 1)
- Wrong-Topic / Hallucinated (Tier 0)

Measures:
- Strict monotonic ordering: Strong > Terse > Generic > Wrong-topic
- Weak monotonic ordering: Strong >= Terse >= Generic >= Wrong-topic
- Wrong-topic rejection rate: % scoring exactly 0.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from resumeproof.core.llm import OpenAICompatibleClient, FakeLLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem, AtomicPoint
from resumeproof.interview.grader import AnswerGrader

CASES = [
    # Case 1: SQLite WAL & Connection Pooling (Task-Tracker API)
    {
        "id": "case_1_sqlite_wal",
        "claim": Claim(id="c1", user_id="u", text="FastAPI REST API with SQLite connection pooling sustaining 350 concurrent read req/sec"),
        "rubric": RubricItem(
            id="rubric_sqlite_1",
            project_id="task-tracker-api",
            topic="sqlite_wal",
            source_file="README.md",
            source_line=5,
            source_quote="Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.",
            criteria={
                0: "Irrelevant, wrong topic, or fabricated claims.",
                1: "Generic explanation of database optimization.",
                2: "Mentions connection pooling without WAL mode details.",
                3: "Accurately articulates WAL mode and 350 req/sec benchmark.",
                4: "Comprehensive architecture covering reader concurrency and pooling parameters.",
            },
            atomic_points=[
                AtomicPoint(id="c1_p1", text="Configured SQLite WAL mode", source_file="README.md", source_line=5, source_quote="SQLite WAL mode"),
                AtomicPoint(id="c1_p2", text="SQLAlchemy connection pool with 10 connections", source_file="README.md", source_line=5, source_quote="connection pooling"),
                AtomicPoint(id="c1_p3", text="Sustained 350 concurrent read req/sec benchmark", source_file="README.md", source_line=5, source_quote="350 concurrent read req/sec"),
            ]
        ),
        "question": "How did you scale SQLite to handle 350 concurrent read requests per second?",
        "strong": "We configured SQLite in WAL (Write-Ahead Logging) mode and tuned the SQLAlchemy connection pool to pool_size=10. This allowed concurrent readers to query without blocking writes, sustaining 350 concurrent read req/sec in Locust benchmarks.",
        "terse": "Enabled SQLite WAL mode and connection pooling to hit 350 req/sec.",
        "generic": "We indexed database tables, optimized our SQL queries, and added async endpoints in FastAPI to keep requests fast.",
        "wrong_topic": "We deployed an Apache Cassandra multi-master cluster with Raft consensus across 5 AWS regions to handle financial transactions.",
    },
    # Case 2: JWT Auth & RBAC (Task-Tracker API)
    {
        "id": "case_2_jwt_auth",
        "claim": Claim(id="c2", user_id="u", text="Implemented JWT authentication and role-based access control."),
        "rubric": RubricItem(
            id="rubric_jwt_2",
            project_id="task-tracker-api",
            topic="jwt_authentication",
            source_file="README.md",
            source_line=8,
            source_quote="Implemented JWT authentication and role-based access control.",
            criteria={
                0: "Irrelevant, refuses to answer, or makes claims that contradict verified facts.",
                1: "Generic explanation of auth tokens without implementation mechanics.",
                2: "Mentions JWT payload signing but omits role-based access control.",
                3: "Accurately articulates JWT token generation, expiration, and RBAC middleware.",
                4: "Expert mastery explaining token revocation, HMAC-SHA256 vs RS256 trade-offs, and RBAC scopes.",
            },
            atomic_points=[
                AtomicPoint(id="c2_p1", text="JWT token generation and signing", source_file="README.md", source_line=8, source_quote="JWT authentication"),
                AtomicPoint(id="c2_p2", text="Role-based access control (RBAC)", source_file="README.md", source_line=8, source_quote="role-based access control"),
            ]
        ),
        "question": "How did you implement JWT authentication and verify user roles in Task-Tracker API?",
        "strong": "We implemented JWT authentication by signing user payloads with HMAC-SHA256 containing user ID and expiration. FastAPI dependency middleware intercepts each request to decode the token and enforce role-based access control (RBAC) across endpoints.",
        "terse": "Signed JWT tokens with role claims and verified them in FastAPI middleware.",
        "generic": "We secured our web application using modern authentication standards and user passwords.",
        "wrong_topic": "We engineered an eBPF network packet scrubbing filter with XDP to prevent BGP route hijacking.",
    },
    # Case 3: RFC 1035 DNS Header Parsing (Mini-DNS Resolver)
    {
        "id": "case_3_dns_rfc1035",
        "claim": Claim(id="c3", user_id="u", text="Unpacked 12-byte DNS headers and parsed A/CNAME records per RFC 1035."),
        "rubric": RubricItem(
            id="rubric_dns_3",
            project_id="mini-dns-resolver",
            topic="dns_rfc1035",
            source_file="README.md",
            source_line=6,
            source_quote="Unpacked 12-byte DNS headers and parsed A and CNAME records per RFC 1035.",
            criteria={
                0: "Irrelevant or hallucinated claims.",
                1: "Generic networking concepts without byte format.",
                2: "Mentions struct parsing but lacks record details.",
                3: "Accurately articulates 12-byte DNS header and RFC 1035 A/CNAME parsing.",
                4: "Detailed wire-format parsing covering flags, opcode, and recursive resolution.",
            },
            atomic_points=[
                AtomicPoint(id="c3_p1", text="Unpacked 12-byte DNS header", source_file="README.md", source_line=6, source_quote="12-byte DNS headers"),
                AtomicPoint(id="c3_p2", text="Parsed A and CNAME records per RFC 1035", source_file="README.md", source_line=6, source_quote="RFC 1035"),
            ]
        ),
        "question": "How does your resolver unpack and parse DNS wire-format messages?",
        "strong": "We used Python's struct module to unpack the standard 12-byte DNS header including transaction ID, flags, and record counts. We then parsed query labels and question sections according to RFC 1035, correctly decoding both A and CNAME records.",
        "terse": "Unpacked 12-byte DNS headers using struct and parsed RFC 1035 A and CNAME records.",
        "generic": "DNS queries use UDP packets to look up website domain names and return server addresses.",
        "wrong_topic": "We configured a Redis Cluster across three nodes with sentinel failover to cache web session cookies.",
    },
    # Case 4: Raw Socket IPv4 Sniffing (Packet Sniffer)
    {
        "id": "case_4_packet_sniffer",
        "claim": Claim(id="c4", user_id="u", text="Captured packets using AF_INET raw sockets, unpacked 20-byte IPv4 headers."),
        "rubric": RubricItem(
            id="rubric_sniff_4",
            project_id="packet-sniffer",
            topic="raw_socket_ipv4",
            source_file="README.md",
            source_line=4,
            source_quote="Captured packets using AF_INET raw sockets, unpacked 20-byte IPv4 headers.",
            criteria={
                0: "Irrelevant or fabricated claims.",
                1: "High-level networking description.",
                2: "Mentions raw socket without header extraction details.",
                3: "Accurately articulates AF_INET raw socket and 20-byte IPv4 header parsing.",
                4: "In-depth protocol disassembly covering IHL bitmasks and transport demuxing.",
            },
            atomic_points=[
                AtomicPoint(id="c4_p1", text="AF_INET raw sockets for packet capture", source_file="README.md", source_line=4, source_quote="AF_INET raw sockets"),
                AtomicPoint(id="c4_p2", text="Unpacked 20-byte IPv4 headers", source_file="README.md", source_line=4, source_quote="20-byte IPv4 headers"),
            ]
        ),
        "question": "How does your packet sniffer capture and unpack IPv4 packet headers?",
        "strong": "We initialized an AF_INET raw socket to capture live packets at the network layer. We unpacked the first 20 bytes representing the standard IPv4 header using struct.unpack, extracting source/destination IP addresses, protocol ID, and calculating IHL header length.",
        "terse": "Opened an AF_INET raw socket and unpacked the 20-byte IPv4 header.",
        "generic": "The program listens to incoming network traffic and displays connection details on the terminal.",
        "wrong_topic": "We trained a 70B parameter LLM using DeepSpeed ZeRO-3 on 64 Nvidia H100 GPUs with mixed precision.",
    },
    # Case 5: Raft Election Timeouts (Raft-KV)
    {
        "id": "case_5_raft_election",
        "claim": Claim(id="c5", user_id="u", text="Implemented randomized election timeouts between 150ms and 300ms with 50ms heartbeats."),
        "rubric": RubricItem(
            id="rubric_raft_5",
            project_id="raft-kv",
            topic="election_timeouts",
            source_file="README.md",
            source_line=7,
            source_quote="Implemented randomized election timeouts between 150ms and 300ms with 50ms heartbeats.",
            criteria={
                0: "Irrelevant, refuses to answer, or makes claims that contradict verified facts.",
                1: "Generic explanation of distributed high availability.",
                2: "Mentions leader election without specific timing bounds.",
                3: "Accurately articulates 150ms-300ms randomized timeouts and 50ms heartbeats.",
                4: "Deep analysis of split-vote mitigation and network jitter resilience.",
            },
            atomic_points=[
                AtomicPoint(id="c5_p1", text="Randomized election timeouts between 150ms and 300ms", source_file="README.md", source_line=7, source_quote="150ms and 300ms"),
                AtomicPoint(id="c5_p2", text="50ms heartbeat intervals", source_file="README.md", source_line=7, source_quote="50ms heartbeats"),
            ]
        ),
        "question": "How did you configure election timeouts and heartbeats in Raft-KV?",
        "strong": "We configured randomized election timeouts between 150ms and 300ms to prevent split votes during leader failover, while maintaining a steady 50ms heartbeat interval from the leader to keep followers synchronized.",
        "terse": "Set election timeouts between 150ms and 300ms with 50ms heartbeats.",
        "generic": "Distributed consensus servers need timers so that if a node goes down, another node can take over.",
        "wrong_topic": "We set up an in-memory SQLite database with Write-Ahead Logging to achieve 350 concurrent read requests per second.",
    },
]

def run_grader_evaluation(model_name: str = "qwen2.5:7b", use_fake: bool = False):
    print("=" * 80)
    print(f"DISCRIMINATION EVALUATION: {model_name} (Fake={use_fake})")
    print("=" * 80)

    if use_fake:
        client = FakeLLMClient()
    else:
        client = OpenAICompatibleClient()

    grader = AnswerGrader(llm_client=client, model=model_name)

    results = []
    strict_monotonic_count = 0
    weak_monotonic_count = 0
    wrong_topic_zero_count = 0

    for idx, c in enumerate(CASES, start=1):
        print(f"\n--- [Case {idx}: {c['id']}] ---", flush=True)
        scores = {}
        for tier in ["strong", "terse", "generic", "wrong_topic"]:
            ans = c[tier]
            grade = grader.grade_answer(
                claim=c["claim"],
                question_text=c["question"],
                answer_text=ans,
                rubric=c["rubric"],
            )
            scores[tier] = grade.raw_score
            print(f"  * {tier.upper():12s}: Score = {grade.raw_score}/4 | Quote = '{grade.evidence_quote[:45]}' | Latency = {grade.latency_ms:.1f}ms", flush=True)

        s = scores["strong"]
        t = scores["terse"]
        g = scores["generic"]
        w = scores["wrong_topic"]

        # Checks
        is_strict = (s > t) and (t > g) and (g > w)
        is_weak = (s >= t) and (t >= g) and (g >= w) and (s > w)
        is_w_zero = (w == 0)

        if is_strict:
            strict_monotonic_count += 1
        if is_weak:
            weak_monotonic_count += 1
        if is_w_zero:
            wrong_topic_zero_count += 1

        results.append({
            "case": c["id"],
            "scores": scores,
            "strict": is_strict,
            "weak": is_weak,
            "wrong_topic_zero": is_w_zero,
        })

    print("\n" + "=" * 80)
    print(f"SUMMARY FOR {model_name}:")
    print(f"Total Rubric Evaluation Cases: {len(CASES)}")
    print(f"Strict Monotonic Ordering (Strong > Terse > Generic > Wrong): {strict_monotonic_count}/{len(CASES)}")
    print(f"Weak Monotonic Ordering   (Strong >= Terse >= Generic >= Wrong): {weak_monotonic_count}/{len(CASES)}")
    print(f"Wrong-Topic Scored Zero   (Wrong-Topic == 0):                   {wrong_topic_zero_count}/{len(CASES)} ({wrong_topic_zero_count/len(CASES):.1%})")
    print("=" * 80)

    return results

if __name__ == "__main__":
    use_fake = "--fake" in sys.argv
    model = "qwen2.5:7b"
    if "--3b" in sys.argv:
        model = "qwen2.5:3b"
    run_grader_evaluation(model_name=model, use_fake=use_fake)
