"""Benchmark script to measure and log raw per-call grading latencies for 7B and 3B models."""

from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.core.telemetry import telemetry
from resumeproof.core.db import Database
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem, AtomicPoint
from resumeproof.interview.grader import AnswerGrader
from resumeproof.interview.caching import GradingCache

MACHINE_NAME = "AMD Ryzen 7 7735HS (8C/16T) | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) | 16GB DDR5"

STD_CRITERIA = {0: "Wrong", 1: "Generic", 2: "Partial", 3: "Competent", 4: "Exact"}

# 10 distinct technical questions and answers for uncached evaluation
EVAL_CASES = [
    {
        "claim": "FastAPI REST API with SQLite connection pooling sustaining 350 concurrent read req/sec",
        "question": "How did you scale SQLite to handle 350 concurrent read requests per second?",
        "answer": "We configured SQLite in WAL mode and tuned SQLAlchemy pool_size=10, reaching 350 req/sec in locust benchmarks.",
        "rubric": RubricItem(
            id="r_wal_1",
            project_id="task-tracker-api",
            topic="sqlite_wal",
            source_file="README.md",
            source_line=5,
            source_quote="Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_1a", text="Configured SQLite WAL mode", source_file="README.md", source_line=5, source_quote="SQLite WAL mode"),
                AtomicPoint(id="pt_1b", text="SQLAlchemy connection pool with 10 connections", source_file="README.md", source_line=5, source_quote="connection pooling"),
                AtomicPoint(id="pt_1c", text="Sustained 350 concurrent read req/sec benchmark", source_file="README.md", source_line=5, source_quote="350 concurrent read req/sec"),
            ]
        )
    },
    {
        "claim": "Implemented JWT authentication and role-based access control.",
        "question": "How did you verify user roles in Task-Tracker API?",
        "answer": "We signed JWT tokens with user scopes and enforced role-based access control (RBAC) via FastAPI dependency middleware.",
        "rubric": RubricItem(
            id="r_jwt_2",
            project_id="task-tracker-api",
            topic="jwt_rbac",
            source_file="README.md",
            source_line=8,
            source_quote="Implemented JWT authentication and role-based access control.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_2a", text="JWT authentication token signing", source_file="README.md", source_line=8, source_quote="JWT authentication"),
                AtomicPoint(id="pt_2b", text="Role-based access control middleware", source_file="README.md", source_line=8, source_quote="role-based access control"),
            ]
        )
    },
    {
        "claim": "Unpacked 12-byte DNS headers and parsed A/CNAME records per RFC 1035.",
        "question": "How did you unpack DNS wire format packets?",
        "answer": "We parsed the 12-byte DNS headers and extracted A and CNAME records following the RFC 1035 wire specification.",
        "rubric": RubricItem(
            id="r_dns_3",
            project_id="mini-dns-resolver",
            topic="dns_rfc1035",
            source_file="README.md",
            source_line=6,
            source_quote="Unpacked 12-byte DNS headers and parsed A and CNAME records per RFC 1035.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_3a", text="Unpacked 12-byte DNS headers", source_file="README.md", source_line=6, source_quote="12-byte DNS headers"),
                AtomicPoint(id="pt_3b", text="Parsed A and CNAME records", source_file="README.md", source_line=6, source_quote="A and CNAME records"),
                AtomicPoint(id="pt_3c", text="RFC 1035 wire specification compliance", source_file="README.md", source_line=6, source_quote="RFC 1035"),
            ]
        )
    },
    {
        "claim": "Trained defect inspection pipeline reaching 61.7% mAP50 on NEU-DET benchmark across 70/20/10 split.",
        "question": "What benchmark and data split did you use for AeroSentinel?",
        "answer": "We evaluated on NEU-DET benchmark across a leak-free 70/20/10 train/val/test split achieving 61.7% mAP50.",
        "rubric": RubricItem(
            id="r_neu_4",
            project_id="aerosentinel",
            topic="neu_det_benchmark",
            source_file="README.md",
            source_line=12,
            source_quote="Reaching 61.7% mAP50 on the NEU-DET benchmark across a leak-free 70/20/10 train/val/test split.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_4a", text="61.7% mAP50 metric reached", source_file="README.md", source_line=12, source_quote="61.7% mAP50"),
                AtomicPoint(id="pt_4b", text="NEU-DET benchmark dataset", source_file="README.md", source_line=12, source_quote="NEU-DET benchmark"),
                AtomicPoint(id="pt_4c", text="Leak-free 70/20/10 train/val/test split", source_file="README.md", source_line=12, source_quote="70/20/10 train/val/test split"),
            ]
        )
    },
    {
        "claim": "Exported baseline to ONNX format and benchmarked GPU inference latency on AWS EC2 Tesla T4.",
        "question": "What GPU and latency did you profile on AWS?",
        "answer": "Exported to ONNX format and benchmarked on an AWS EC2 Tesla T4 instance, profiling 1.6ms execution time per frame.",
        "rubric": RubricItem(
            id="r_onnx_5",
            project_id="aerosentinel",
            topic="onnx_inference",
            source_file="README.md",
            source_line=15,
            source_quote="Exported baseline to ONNX format and benchmarked GPU inference latency on an AWS EC2 Tesla T4 instance, profiling 1.6ms execution time per frame (416 × 416).",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_5a", text="Exported to ONNX format", source_file="README.md", source_line=15, source_quote="ONNX format"),
                AtomicPoint(id="pt_5b", text="AWS EC2 Tesla T4 instance GPU benchmarking", source_file="README.md", source_line=15, source_quote="AWS EC2 Tesla T4 instance"),
                AtomicPoint(id="pt_5c", text="1.6ms execution time per frame", source_file="README.md", source_line=15, source_quote="1.6ms execution time per frame"),
            ]
        )
    },
    {
        "claim": "Benchmarked 4-head P2 variant gaining on validation (62.2% vs 61.2%) but dropping to 57.3% on test.",
        "question": "What did the ablation study show for the P2 head variant?",
        "answer": "The 4-head P2 variant gained on validation (62.2% vs 61.2%) but dropped to 57.3% on test, so baseline was retained.",
        "rubric": RubricItem(
            id="r_p2_6",
            project_id="aerosentinel",
            topic="p2_ablation",
            source_file="README.md",
            source_line=18,
            source_quote="Benchmarked a 4-head P2 variant that gained on validation (62.2% vs 61.2%) but dropped to 57.3% on test, so the baseline was retained.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_6a", text="4-head P2 architectural variant benchmarked", source_file="README.md", source_line=18, source_quote="4-head P2 variant"),
                AtomicPoint(id="pt_6b", text="Validation gain (62.2% vs 61.2%)", source_file="README.md", source_line=18, source_quote="gained on validation (62.2% vs 61.2%)"),
                AtomicPoint(id="pt_6c", text="Test drop to 57.3% causing baseline retention", source_file="README.md", source_line=18, source_quote="dropped to 57.3% on test, so the baseline was retained"),
            ]
        )
    },
    {
        "claim": "Built academic research tool indexing arXiv papers in ChromaDB with HNSW vector indexing for semantic search.",
        "question": "How did you implement vector search for PaperPulse AI?",
        "answer": "We indexed arXiv papers in ChromaDB using HNSW vector indexing for semantic search over paper embeddings.",
        "rubric": RubricItem(
            id="r_hnsw_7",
            project_id="paperpulse-ai",
            topic="chromadb_hnsw",
            source_file="README.md",
            source_line=9,
            source_quote="Built an academic research tool indexing arXiv papers in ChromaDB with HNSW vector indexing for semantic search via a FastAPI backend.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_7a", text="Indexed arXiv papers in ChromaDB", source_file="README.md", source_line=9, source_quote="ChromaDB"),
                AtomicPoint(id="pt_7b", text="HNSW vector indexing algorithm", source_file="README.md", source_line=9, source_quote="HNSW vector indexing"),
                AtomicPoint(id="pt_7c", text="FastAPI backend semantic search service", source_file="README.md", source_line=9, source_quote="FastAPI backend"),
            ]
        )
    },
    {
        "claim": "Implemented deterministic token-overlap citation guardrail (35% threshold) against retrieved sources.",
        "question": "How do you detect hallucinations or unbacked citations in PaperPulse AI?",
        "answer": "We implemented a deterministic token-overlap citation guardrail with a 35% threshold against retrieved sources to flag or reject claims.",
        "rubric": RubricItem(
            id="r_guard_8",
            project_id="paperpulse-ai",
            topic="citation_guardrail",
            source_file="README.md",
            source_line=14,
            source_quote="Implemented a deterministic token-overlap citation guardrail (35% threshold) against retrieved sources to flag or reject unsupported statements.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_8a", text="Deterministic token-overlap citation guardrail", source_file="README.md", source_line=14, source_quote="deterministic token-overlap citation guardrail"),
                AtomicPoint(id="pt_8b", text="35% threshold for citation verification", source_file="README.md", source_line=14, source_quote="35% threshold"),
                AtomicPoint(id="pt_8c", text="Rejects or flags unsupported statements", source_file="README.md", source_line=14, source_quote="flag or reject unsupported statements"),
            ]
        )
    },
    {
        "claim": "Engineered captive portal authentication engine integrating Cisco Meraki Dashboard API v1 with FastAPI.",
        "question": "How did NetPass Gateway interface with Cisco networking gear?",
        "answer": "Engineered captive portal authentication integrating Cisco Meraki Dashboard API v1 with a FastAPI backend.",
        "rubric": RubricItem(
            id="r_meraki_9",
            project_id="netpass-gateway",
            topic="meraki_api",
            source_file="README.md",
            source_line=4,
            source_quote="Engineered a captive portal authentication engine integrating Cisco Meraki Dashboard API v1 with FastAPI, automating guest client provisioning.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_9a", text="Captive portal authentication engine", source_file="README.md", source_line=4, source_quote="captive portal authentication"),
                AtomicPoint(id="pt_9b", text="Cisco Meraki Dashboard API v1 integration", source_file="README.md", source_line=4, source_quote="Cisco Meraki Dashboard API v1"),
                AtomicPoint(id="pt_9c", text="FastAPI backend provisioning service", source_file="README.md", source_line=4, source_quote="FastAPI"),
            ]
        )
    },
    {
        "claim": "Built JSON-RPC 2.0 stdio MCP server exposing 4 analytical query tools over a relational sales database.",
        "question": "How did you build the communication layer in GameData MCP?",
        "answer": "Built a JSON-RPC 2.0 stdio Model Context Protocol (MCP) server exposing 4 analytical query tools over SQLite.",
        "rubric": RubricItem(
            id="r_mcp_10",
            project_id="gamedata-mcp",
            topic="json_rpc_mcp",
            source_file="README.md",
            source_line=7,
            source_quote="Built a JSON-RPC 2.0 stdio MCP server exposing 4 analytical query tools over a relational sales database with structured telemetry.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_10a", text="JSON-RPC 2.0 stdio protocol", source_file="README.md", source_line=7, source_quote="JSON-RPC 2.0 stdio"),
                AtomicPoint(id="pt_10b", text="Exposed 4 analytical query tools", source_file="README.md", source_line=7, source_quote="4 analytical query tools"),
                AtomicPoint(id="pt_10c", text="Relational database structured telemetry", source_file="README.md", source_line=7, source_quote="relational sales database with structured telemetry"),
            ]
        )
    },
]

def run_latency_benchmark():
    client = OpenAICompatibleClient(base_url="http://127.0.0.1:11434")
    db = Database(":memory:")
    cache = GradingCache(db)

    print("=" * 80)
    print("STARTING LATENCY BENCHMARK ON REAL LOCAL HARDWARE")
    print(f"Machine: {MACHINE_NAME}")
    print("=" * 80)

    # 1. Warm-up 7B
    print("\n[Step 1] Initializing and warming local 7B (qwen2.5:7b)...")
    t0 = time.perf_counter()
    grader_7b = AnswerGrader(llm_client=client, model="qwen2.5:7b")
    c0 = Claim(id="c_warm", user_id="u", text=EVAL_CASES[0]["claim"])
    r0 = EVAL_CASES[0]["rubric"]
    res_warm_7b = grader_7b.grade_answer(claim=c0, question_text=EVAL_CASES[0]["question"], answer_text=EVAL_CASES[0]["answer"], rubric=r0)
    warm_7b_time = (time.perf_counter() - t0) * 1000.0
    print(f"7B warm-up completed in {warm_7b_time:.2f} ms (Cold start / VRAM load)")

    # 2. Run 10 uncached 7B calls
    print("\n[Step 2] Executing 10 uncached 7B grading calls...")
    records_7b_uncached = []
    for idx, case in enumerate(EVAL_CASES):
        claim_obj = Claim(id=f"c_7b_{idx}", user_id="u", text=case["claim"])
        rubric_obj = case["rubric"]
        
        telemetry.records.clear()
        t_start = time.perf_counter()
        res = grader_7b.grade_answer(
            claim=claim_obj,
            question_text=case["question"],
            answer_text=case["answer"],
            rubric=rubric_obj
        )
        t_ms = (time.perf_counter() - t_start) * 1000.0
        
        # Get last logged telemetry call
        call = telemetry.records[-1] if telemetry.records else None
        p_tok = call.prompt_tokens if call else 0
        o_tok = call.completion_tokens if call else 0
        
        # Store in cache for subsequent cached test
        cache.set(
            claim_id=claim_obj.id,
            question_text=case["question"],
            answer_text=case["answer"],
            model="qwen2.5:7b",
            grade=res
        )

        records_7b_uncached.append({
            "call_index": idx + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "qwen2.5:7b",
            "prompt_tokens": p_tok,
            "output_tokens": o_tok,
            "total_ms": round(t_ms, 2),
            "cache_hit": "No",
            "score": res.raw_score,
        })
        print(f"  Call {idx+1}/10: {t_ms:.2f} ms | p_tok={p_tok}, o_tok={o_tok}")

    # 3. Run 10 cached 7B calls
    print("\n[Step 3] Executing 10 cached 7B lookups...")
    records_7b_cached = []
    for idx, case in enumerate(EVAL_CASES):
        t_start = time.perf_counter()
        cached_res = cache.get(
            claim_id=f"c_7b_{idx}",
            question_text=case["question"],
            answer_text=case["answer"],
            model="qwen2.5:7b"
        )
        t_ms = (time.perf_counter() - t_start) * 1000.0
        records_7b_cached.append({
            "call_index": idx + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "qwen2.5:7b",
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_ms": round(t_ms, 3),
            "cache_hit": "Yes",
            "score": cached_res.raw_score if cached_res else None,
        })

    # 4. Warm-up 3B
    print("\n[Step 4] Initializing and warming local 3B (qwen2.5:3b)...")
    t0 = time.perf_counter()
    grader_3b = AnswerGrader(llm_client=client, model="qwen2.5:3b")
    res_warm_3b = grader_3b.grade_answer(claim=c0, question_text=EVAL_CASES[0]["question"], answer_text=EVAL_CASES[0]["answer"], rubric=r0)
    warm_3b_time = (time.perf_counter() - t0) * 1000.0
    print(f"3B warm-up completed in {warm_3b_time:.2f} ms (Model switch / VRAM load)")

    # 5. Run 10 uncached 3B calls
    print("\n[Step 5] Executing 10 uncached 3B grading calls...")
    records_3b_uncached = []
    for idx, case in enumerate(EVAL_CASES):
        claim_obj = Claim(id=f"c_3b_{idx}", user_id="u", text=case["claim"])
        rubric_obj = case["rubric"]
        
        telemetry.records.clear()
        t_start = time.perf_counter()
        res = grader_3b.grade_answer(
            claim=claim_obj,
            question_text=case["question"],
            answer_text=case["answer"],
            rubric=rubric_obj
        )
        t_ms = (time.perf_counter() - t_start) * 1000.0
        
        call = telemetry.records[-1] if telemetry.records else None
        p_tok = call.prompt_tokens if call else 0
        o_tok = call.completion_tokens if call else 0

        cache.set(
            claim_id=claim_obj.id,
            question_text=case["question"],
            answer_text=case["answer"],
            model="qwen2.5:3b",
            grade=res
        )

        records_3b_uncached.append({
            "call_index": idx + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "qwen2.5:3b",
            "prompt_tokens": p_tok,
            "output_tokens": o_tok,
            "total_ms": round(t_ms, 2),
            "cache_hit": "No",
            "score": res.raw_score,
        })
        print(f"  Call {idx+1}/10: {t_ms:.2f} ms | p_tok={p_tok}, o_tok={o_tok}")

    # 6. Run 10 cached 3B calls
    print("\n[Step 6] Executing 10 cached 3B lookups...")
    records_3b_cached = []
    for idx, case in enumerate(EVAL_CASES):
        t_start = time.perf_counter()
        cached_res = cache.get(
            claim_id=f"c_3b_{idx}",
            question_text=case["question"],
            answer_text=case["answer"],
            model="qwen2.5:3b"
        )
        t_ms = (time.perf_counter() - t_start) * 1000.0
        records_3b_cached.append({
            "call_index": idx + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "qwen2.5:3b",
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_ms": round(t_ms, 3),
            "cache_hit": "Yes",
            "score": cached_res.raw_score if cached_res else None,
        })

    # Save to json file
    output_data = {
        "machine": MACHINE_NAME,
        "date": "2026-10-26",
        "cold_start_7b_ms": round(warm_7b_time, 2),
        "model_switch_3b_ms": round(warm_3b_time, 2),
        "records_7b_uncached": records_7b_uncached,
        "records_7b_cached": records_7b_cached,
        "records_3b_uncached": records_3b_uncached,
        "records_3b_cached": records_3b_cached,
    }
    
    out_file = ROOT_DIR / "results" / "raw_latency_logs.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\nSaved raw per-call telemetry to: {out_file}")

if __name__ == "__main__":
    run_latency_benchmark()
