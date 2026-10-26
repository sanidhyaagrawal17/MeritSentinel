"""Evaluation script for 5 items x 5 answer levels across local 7B and hosted model.

Answer levels:
0: none (0% points matched, off-topic or empty)
1: 1/3 of points (1 of 3 points matched, paraphrased with different numeric formats)
2: 2/3 of points (2 of 3 points matched, paraphrased with different numeric formats)
3: all points terse (all 3 points matched, terse phrasing, different numeric format)
4: all points detailed (all 3 points matched, detailed architectural phrasing, different numeric format)
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem, AtomicPoint
from resumeproof.interview.grader import AnswerGrader

STD_CRITERIA = {0: "Wrong", 1: "Generic", 2: "Partial", 3: "Competent", 4: "Exact"}

DISCRIMINATION_ITEMS = [
    # Item 1: SQLite WAL & Connection Pooling (Task-Tracker API)
    {
        "id": "item_1_sqlite_wal",
        "claim": Claim(id="c1", user_id="u", text="FastAPI REST API with SQLite connection pooling sustaining 350 concurrent read req/sec"),
        "rubric": RubricItem(
            id="r_sqlite_1",
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
        ),
        "question": "How did you scale SQLite to handle 350 concurrent read requests per second?",
        "levels": {
            0: "We built standard REST endpoints and followed standard database tutorials to keep the API responsive.",
            1: "We activated Write-Ahead Logging on the SQLite database engine to avoid write locks during concurrent operations.",
            2: "We turned on Write-Ahead Logging on the SQLite engine and allocated a pool of ten connections in SQLAlchemy for database operations.",
            3: "Enabled WAL persistence, capped connections at ten, and sustained 0.35k read ops/sec in throughput testing.",
            4: "To maximize concurrent read throughput, we switched the SQLite journal mode to Write-Ahead Logging so readers do not block writers. We tuned the SQLAlchemy pool to maintain a pool of ten active connections, and verified with load generation that the service sustained three hundred and fifty queries per second under steady load."
        }
    },
    # Item 2: DNS RFC 1035 Header & Record Parsing (Mini-DNS Resolver)
    {
        "id": "item_2_dns_rfc1035",
        "claim": Claim(id="c2", user_id="u", text="Unpacked 12-byte DNS headers and parsed A/CNAME records per RFC 1035."),
        "rubric": RubricItem(
            id="r_dns_2",
            project_id="mini-dns-resolver",
            topic="dns_rfc1035",
            source_file="README.md",
            source_line=6,
            source_quote="Unpacked 12-byte DNS headers and parsed A and CNAME records per RFC 1035.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_2a", text="Unpacked 12-byte DNS headers", source_file="README.md", source_line=6, source_quote="12-byte DNS headers"),
                AtomicPoint(id="pt_2b", text="Parsed A and CNAME records", source_file="README.md", source_line=6, source_quote="A and CNAME records"),
                AtomicPoint(id="pt_2c", text="RFC 1035 wire specification compliance", source_file="README.md", source_line=6, source_quote="RFC 1035"),
            ]
        ),
        "question": "How did you unpack DNS wire format packets per RFC 1035?",
        "levels": {
            0: "We resolved domain names using the standard socket library gethostbyname function.",
            1: "We constructed a bitwise parser for the standard 12-octet DNS header.",
            2: "We implemented unpacking of the twelve-byte DNS header and handled standard IPv4 host address records and canonical name aliases.",
            3: "Followed RFC 1035 wire specifications, unpackaged the 12-octet header, and resolved both A-type and CNAME alias records.",
            4: "We engineered a binary packet parser adhering strictly to the RFC 1035 specification. The code unpacks the twelve-byte header using struct, parses transaction IDs and DNS question records, and decodes resource records including Address (A) mapping and Canonical Name (CNAME) redirects."
        }
    },
    # Item 3: Defect Inspection YOLO11 (AeroSentinel)
    {
        "id": "item_3_neu_det_benchmark",
        "claim": Claim(id="c3", user_id="u", text="Trained defect inspection pipeline reaching 61.7% mAP50 on NEU-DET benchmark across 70/20/10 split."),
        "rubric": RubricItem(
            id="r_yolo_3",
            project_id="aerosentinel",
            topic="neu_det_benchmark",
            source_file="README.md",
            source_line=12,
            source_quote="Reaching 61.7% mAP50 on the NEU-DET benchmark across a leak-free 70/20/10 train/val/test split.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_3a", text="61.7% mAP50 metric reached", source_file="README.md", source_line=12, source_quote="61.7% mAP50"),
                AtomicPoint(id="pt_3b", text="NEU-DET benchmark dataset", source_file="README.md", source_line=12, source_quote="NEU-DET benchmark"),
                AtomicPoint(id="pt_3c", text="Leak-free 70/20/10 train/val/test split", source_file="README.md", source_line=12, source_quote="70/20/10 train/val/test split"),
            ]
        ),
        "question": "What benchmark and dataset split did you evaluate on for AeroSentinel?",
        "levels": {
            0: "We trained computer vision models using standard deep learning workflows.",
            1: "We partitioned the dataset using a seventy/twenty/ten percent distribution across train, val, and test.",
            2: "We evaluated the defect detector on the NEU-DET benchmark using a leak-free 70/20/10 train/val/test split.",
            3: "Attained 0.617 mAP50 on NEU-DET surface defects using a 70/20/10 train/val/test split.",
            4: "We benchmarked our vision pipeline on the NEU-DET defect dataset using an isolated 70% train, 20% validation, and 10% test split. Across the evaluation test set, our trained detector achieved a mean Average Precision of 0.617 at IoU threshold 0.50 (61.7% mAP50)."
        }
    },
    # Item 4: Ablation Study P2 Head (AeroSentinel)
    {
        "id": "item_4_p2_ablation",
        "claim": Claim(id="c4", user_id="u", text="Benchmarked 4-head P2 variant gaining on validation (62.2% vs 61.2%) but dropping to 57.3% on test."),
        "rubric": RubricItem(
            id="r_p2_4",
            project_id="aerosentinel",
            topic="p2_ablation",
            source_file="README.md",
            source_line=18,
            source_quote="Benchmarked a 4-head P2 variant that gained on validation (62.2% vs 61.2%) but dropped to 57.3% on test, so the baseline was retained.",
            criteria=STD_CRITERIA,
            atomic_points=[
                AtomicPoint(id="pt_4a", text="4-head P2 architectural variant benchmarked", source_file="README.md", source_line=18, source_quote="4-head P2 variant"),
                AtomicPoint(id="pt_4b", text="Validation gain (62.2% vs 61.2%)", source_file="README.md", source_line=18, source_quote="gained on validation (62.2% vs 61.2%)"),
                AtomicPoint(id="pt_4c", text="Test drop to 57.3% causing baseline retention", source_file="README.md", source_line=18, source_quote="dropped to 57.3% on test, so the baseline was retained"),
            ]
        ),
        "question": "What were the empirical results of the P2 head ablation study?",
        "levels": {
            0: "We ran experimental iterations in PyTorch to explore neural network architectures.",
            1: "We developed an experimental four-head architecture incorporating a P2 feature layer.",
            2: "Our experimental 4-head P2 variant yielded validation gains reaching 62.2% compared to 61.2% for baseline.",
            3: "A 4-head P2 modification reached 62.2% on val but fell to 0.573 on test, so we reverted.",
            4: "We conducted an ablation by introducing a four-head P2 detection module. While it demonstrated higher validation precision (0.622 versus 0.612 for baseline), test set evaluation revealed generalization failure dropping to 57.3%, leading us to reject the variant and preserve the baseline."
        }
    },
    # Item 5: ONNX Latency & GPU Profiling (AeroSentinel)
    {
        "id": "item_5_onnx_latency",
        "claim": Claim(id="c5", user_id="u", text="Exported baseline to ONNX format and benchmarked GPU inference latency on AWS EC2 Tesla T4."),
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
        ),
        "question": "How did you optimize and benchmark inference latency on cloud infrastructure?",
        "levels": {
            0: "We evaluated inference performance on cloud computing instances with GPU accelerators.",
            1: "We converted our baseline PyTorch model graph into the ONNX open runtime format.",
            2: "We exported our weights to ONNX format and deployed benchmarking onto an Amazon Web Services EC2 node equipped with a Tesla T4 GPU.",
            3: "Exported to ONNX, benchmarked on AWS EC2 T4, clocking 0.0016s per 416-pixel frame.",
            4: "To minimize inference overhead, we converted the baseline weights to the ONNX graph format and profiled inference latency on an AWS EC2 instance featuring a Tesla T4 accelerator. The profiling demonstrated an average frame time of 1.6ms per frame at 416x416 resolution."
        }
    },
]

def evaluate_model(grader: AnswerGrader, model_label: str) -> dict:
    print(f"\n{'='*70}\nEVALUATING DISCRIMINATION: {model_label}\n{'='*70}")
    results = {}
    monotonic_count = 0

    for item in DISCRIMINATION_ITEMS:
        item_id = item["id"]
        results[item_id] = {}
        scores = []
        print(f"\n--> {item_id}:")

        for level in [0, 1, 2, 3, 4]:
            answer_text = item["levels"][level]
            res = grader.grade_answer(
                claim=item["claim"],
                question_text=item["question"],
                answer_text=answer_text,
                rubric=item["rubric"],
            )
            scores.append(res.raw_score)
            results[item_id][f"level_{level}"] = {
                "score": res.raw_score,
                "evidence_quote": res.evidence_quote,
                "feedback": res.feedback,
            }
            print(f"    Level {level} ({['None', '1/3', '2/3', 'Terse', 'Detailed'][level]}): Score {res.raw_score}/4 | Quote: \"{res.evidence_quote[:40]}\"")

        # Check monotonic ordering: S0 <= S1 <= S2 <= S3 <= S4
        is_monotonic = all(scores[i] <= scores[i+1] for i in range(len(scores) - 1))
        results[item_id]["scores"] = scores
        results[item_id]["monotonic"] = is_monotonic
        if is_monotonic:
            monotonic_count += 1
        print(f"    Ordering: {scores} | Monotonic: {is_monotonic}")
        time.sleep(1.0)

    return {
        "model_label": model_label,
        "results": results,
        "monotonic_items": monotonic_count,
        "total_items": len(DISCRIMINATION_ITEMS),
    }

def main():
    # 1. Local 7B evaluation
    client_local = OpenAICompatibleClient(base_url="http://127.0.0.1:11434")
    grader_local_7b = AnswerGrader(llm_client=client_local, model="qwen2.5:7b")
    local_eval = evaluate_model(grader_local_7b, "Local Ollama qwen2.5:7b")

    # 2. Hosted evaluation (OpenRouter OpenAI-compatible client)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    hosted_eval = None
    if openrouter_key:
        client_hosted = OpenAICompatibleClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        grader_hosted = AnswerGrader(llm_client=client_hosted, model="liquid/lfm-2.5-2.6b:free")
        hosted_eval = evaluate_model(grader_hosted, "Hosted OpenRouter liquid/lfm-2.5-2.6b:free")
    else:
        print("\n[NOTE] OPENROUTER_API_KEY not set; skipping hosted model evaluation.")

    # Save to disk
    out_path = ROOT_DIR / "results" / "discrimination_5x5_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": "2026-10-26",
        "local_7b": local_eval,
        "hosted_model": hosted_eval,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved 5x5 discrimination evaluation to: {out_path}")

if __name__ == "__main__":
    main()
