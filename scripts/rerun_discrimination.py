import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.grader import AnswerGrader
from resumeproof.rubric.builder import build_rubrics_for_persona, build_rubrics_for_project
from resumeproof.schemas.claims import Claim, ClaimType

def rerun_discrimination():
    llm = OpenAICompatibleClient()
    grader = AnswerGrader(llm_client=llm, model="qwen2.5:7b")
    digest = llm.get_model_digest("qwen2.5:7b") or "845dbda0ea48"

    # Candidate items
    tt_rubrics = build_rubrics_for_project(Path("data/demo/sam_okafor/projects/task-tracker-api"), persona_id="sam_okafor")
    wal_rubric = [r for r in tt_rubrics if "350" in r.source_quote][0]

    candidate_items = [
        # Item 1: Task-Tracker API
        (
            wal_rubric,
            "Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.",
            "How did you configure SQLite connection pooling and WAL mode to sustain 350 concurrent read req/sec in Task-Tracker API?",
            {
                "strong": (
                    "In Task-Tracker API, SQLite can become a write bottleneck, so we enabled WAL (Write-Ahead Logging) "
                    "mode and tuned SQLAlchemy connection pooling to 10 connections. For read-heavy operations, this "
                    "allowed concurrent reads while WAL checkpointing happened in the background, reaching 350 read "
                    "req/sec in our locust test."
                ),
                "terse": "Enabled SQLite WAL mode and set pool_size=10.",
                "generic": "We configured a standard database pool and ensured good performance with standard error handling.",
                "wrong_topic": "We used an Apache Kafka event bus and BGP route scrubbing to stream financial transactions into a Byzantine Raft cluster.",
            }
        ),
        # Item 2: Mini-DNS-Resolver
        (
            [r for r in build_rubrics_for_persona("sam_okafor") if "45ms" in r.source_quote][0],
            "Implemented recursive UDP DNS resolver resolving queries in under 45ms for cached records.",
            "How did you achieve under 45ms cached DNS resolution in your recursive UDP DNS resolver?",
            {
                "strong": (
                    "In Mini-DNS-Resolver, we built a recursive UDP query resolver using Python raw sockets and struct module "
                    "for unpacking UDP payloads. We used an in-memory dictionary cache with TTL expiration, resolving cached "
                    "DNS queries in under 45ms and supporting A and CNAME record types."
                ),
                "terse": "We used an in-memory dict cache with TTL expiration resolving queries in under 45ms.",
                "generic": "We implemented an in-memory caching layer to speed up network query responses.",
                "wrong_topic": "We tuned LightGBM interaction depth to 6 with FP32 ONNX Runtime graph execution on NVIDIA GPUs.",
            }
        ),
        # Item 3: Packet-Sniffer
        (
            [r for r in build_rubrics_for_persona("sam_okafor") if "20-byte" in r.source_quote][0],
            "Captured and decoded IPv4, TCP, and UDP packet headers with custom bitwise unpacking.",
            "How did you unpack and decode IPv4 headers to identify TCP and UDP protocols in Packet-Sniffer?",
            {
                "strong": (
                    "In Packet-Sniffer, we used AF_INET raw sockets in Python to capture packets. "
                    "We unpacked the 20-byte IPv4 headers, extracting IP version, IHL, TTL, and source/dest IPs using bit shift operations, "
                    "and verified protocol extraction for IPv4, TCP, and UDP."
                ),
                "terse": "Unpacked 20-byte IPv4 headers using AF_INET raw sockets to extract TCP and UDP protocol numbers.",
                "generic": "We captured network packets and parsed header information to log network traffic.",
                "wrong_topic": "We deployed Kubernetes with Istio service mesh and Paxos consensus engines across multi-region clusters.",
            }
        ),
        # Item 4: Flow-Limiter
        (
            [r for r in build_rubrics_for_persona("alex_rivera") if "80,000" in r.source_quote][0],
            "Distributed sliding-window token bucket limiter processing 80,000 req/sec at sub-millisecond p99 latency.",
            "How did you design Flow-Limiter to sustain 80,000 req/sec with p99 latency under 0.8ms?",
            {
                "strong": (
                    "In Flow-Limiter, we built a distributed sliding-window rate limiter in Go with Redis Lua scripts. "
                    "An atomic Lua script evaluates token count and timestamps in a single round-trip, processing 80,000 req/sec "
                    "with p99 latency under 0.8ms. A fallback local in-memory leaky bucket activates if Redis heartbeat drops."
                ),
                "terse": "Processed 80,000 req/sec using Redis rate limiting.",
                "generic": "We used Redis cache to rate-limit incoming API traffic and reduce latency spikes.",
                "wrong_topic": "We implemented SQLite WAL checkpointing and tuned async SQLAlchemy pool size to 10 connections.",
            }
        ),
        # Item 5: Raft-KV
        (
            [r for r in build_rubrics_for_persona("alex_rivera") if "45,000" in r.source_quote][0],
            "Sustained 45,000 req/sec write throughput under 3-node cluster benchmark.",
            "How did you sustain 45,000 req/sec write throughput under cluster testing in Raft-KV?",
            {
                "strong": (
                    "In Raft-KV, we implemented the Raft consensus algorithm in Go with leader election randomized timeouts (150ms - 300ms) "
                    "and heartbeats every 50ms. Tested on a 3-node cluster, we sustained a maximum stable write throughput of 45,000 req/sec "
                    "with 99.99% linearizable reads under concurrent Jepsen partition tests."
                ),
                "terse": "Tested on 3 nodes, sustaining 45,000 req/sec write throughput.",
                "generic": "We benchmarked our key-value cluster under heavy load to measure maximum throughput.",
                "wrong_topic": "We parsed RFC 1035 wire format A and CNAME record headers using Python UDP sockets.",
            }
        ),
    ]

    print("=" * 80)
    print("RERUNNING GRADER DISCRIMINATION WITH SPLIT ATOMIC POINTS (qwen2.5:7b)")
    print("=" * 80)

    ordering_holds_count = 0
    results_list = []
    call_telemetry_logs = []

    for r_item, claim_txt, question_txt, answers in candidate_items:
        claim_obj = Claim(
            id=f"c_discrim_{r_item.project_id}",
            user_id="u_discrim",
            persona_id=r_item.persona_id,
            project_id=r_item.project_id,
            claim_type=ClaimType.PROJECT_METRIC,
            text=claim_txt,
        )

        scores = {}
        quotes = {}
        point_evals_by_tier = {}
        for tier in ["strong", "terse", "generic", "wrong_topic"]:
            ans_text = answers[tier]
            t0 = time.perf_counter()
            grade_res = grader.grade_answer(
                claim=claim_obj,
                question_text=question_txt,
                answer_text=ans_text,
                rubric=r_item,
            )
            g_lat = (time.perf_counter() - t0) * 1000.0
            scores[tier] = grade_res.raw_score
            quotes[tier] = grade_res.evidence_quote
            point_evals_by_tier[tier] = {
                pid: {"matched": pe.matched, "quote": pe.evidence_quote}
                for pid, pe in grade_res.point_evaluations.items()
            }
            from resumeproof.core.telemetry import telemetry
            last_metric = telemetry.records[-1] if telemetry.records else None
            call_telemetry_logs.append({
                "timestamp": last_metric.timestamp.isoformat() if last_metric else datetime.now(timezone.utc).isoformat(),
                "project": r_item.project_id,
                "tier": tier,
                "model": "qwen2.5:7b",
                "digest": digest,
                "prompt_tokens": last_metric.prompt_tokens if last_metric else 0,
                "output_tokens": last_metric.completion_tokens if last_metric else 0,
                "total_tokens": last_metric.total_tokens if last_metric else 0,
                "total_ms": last_metric.latency_ms if last_metric else g_lat,
                "score": grade_res.raw_score,
            })

        strict_holds = scores["strong"] > scores["terse"] > scores["generic"] >= scores["wrong_topic"]
        if strict_holds:
            ordering_holds_count += 1

        topic_label = f"{r_item.project_id[:16]}..{r_item.topic[:12]}"
        order_str = "YES" if strict_holds else "NO"
        print(f"{topic_label:<32} | Strong: {scores['strong']} | Terse: {scores['terse']} | Generic: {scores['generic']} | Wrong: {scores['wrong_topic']} | Holds: {order_str}")
        results_list.append({
            "project_id": r_item.project_id,
            "topic": r_item.topic,
            "scores": scores,
            "strict_holds": strict_holds,
            "atomic_points": [{"id": p.id, "text": p.text} for p in r_item.atomic_points],
            "point_evals": point_evals_by_tier,
        })

    print("-" * 80)
    print(f"Strict ordering holds for {ordering_holds_count} / {len(candidate_items)} items.")
    Path("results/rerun_discrimination_results.json").write_text(json.dumps({
        "ordering_holds_count": ordering_holds_count,
        "total_items": len(candidate_items),
        "results": results_list,
        "telemetry_logs": call_telemetry_logs
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    rerun_discrimination()
