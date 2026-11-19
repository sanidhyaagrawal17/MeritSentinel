"""Evaluates 5 adversarial discrimination cases across 5 rubric items on local Qwen 2.5 7B.

Adversarial case categories:
1. Right numbers on the wrong subject
2. Correct subject with wrong numbers
3. Contradictory statements
4. One-word answers
5. Confident fabrication / irrelevant topic
"""

import json
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.grader import AnswerGrader
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus
from resumeproof.schemas.rubric import RubricItem, AtomicPoint

ADVERSARIAL_TEST_CASES = {
    "item_1_sqlite_wal": {
        "title": "SQLite WAL mode & connection pooling (350 req/sec)",
        "question": "How did you scale SQLite to handle 350 concurrent read requests per second?",
        "claim_text": "Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.",
        "atomic_points": [
            "Configured SQLite journal mode to Write-Ahead Logging (WAL)",
            "Configured connection pooling (e.g. pool_size=10 in SQLAlchemy)",
            "Sustained 350 concurrent read requests per second benchmark",
            "Maintained sub-50ms p95 latency without database lock errors"
        ],
        "cases": [
            {
                "case_type": "right_numbers_wrong_subject",
                "answer": "We tuned an ONNX Runtime GPU inference pipeline to achieve 350 frames per second on an AWS EC2 Tesla T4 instance with pool size 10.",
            },
            {
                "case_type": "correct_subject_wrong_numbers",
                "answer": "We configured SQLite in WAL mode with connection pooling, scaling to 50,000 requests per second with a pool size of 500 connections.",
            },
            {
                "case_type": "contradictory_statements",
                "answer": "We enabled SQLite WAL mode which completely disables connection pooling and locks out all concurrent readers causing 0 req/sec.",
            },
            {
                "case_type": "one_word_answer",
                "answer": "SQLite",
            },
            {
                "case_type": "confident_fabrication",
                "answer": "We deployed a globally distributed Apache Cassandra cluster across 5 AWS regions with Raft multi-master consensus to handle database reads.",
            }
        ]
    },
    "item_2_dns_rfc1035": {
        "title": "DNS RFC 1035 wire parser & struct unpack (12-byte header)",
        "question": "How did your resolver unpack and parse DNS wire-format messages per RFC 1035?",
        "claim_text": "Implemented recursive UDP DNS resolver resolving queries in under 45ms for cached records, unpacking 12-byte headers.",
        "atomic_points": [
            "Unpacked 12-byte DNS wire-format header using struct unpack",
            "Parsed query ID, flags, QDCOUNT, and ANCOUNT adhering to RFC 1035",
            "Decoded length-prefixed domain labels iteratively",
            "Handled A records and followed CNAME recursive aliases"
        ],
        "cases": [
            {
                "case_type": "right_numbers_wrong_subject",
                "answer": "We unpacked 12-byte AES-GCM authentication tags in our WireGuard captive portal network access controller to resolve 45ms tokens.",
            },
            {
                "case_type": "correct_subject_wrong_numbers",
                "answer": "We parsed DNS wire headers using struct unpack on a 64-byte header supporting BGP Anycast routes with 500ms resolution.",
            },
            {
                "case_type": "contradictory_statements",
                "answer": "Our DNS resolver does not parse binary wire packets or headers because RFC 1035 strictly forbids using Python sockets.",
            },
            {
                "case_type": "one_word_answer",
                "answer": "DNS",
            },
            {
                "case_type": "confident_fabrication",
                "answer": "We used Cloudflare Magic Transit and eBPF XDP route scrubbers to mitigate 100 Gbps DDoS amplification attacks against our nameservers.",
            }
        ]
    },
    "item_3_neu_det_benchmark": {
        "title": "NEU-DET 61.7% mAP50 & 70/20/10 split",
        "question": "What benchmark and data splitting methodology did you use for the defect inspection pipeline?",
        "claim_text": "Trained a defect inspection pipeline reaching 61.7% mAP50 on the NEU-DET benchmark across a leak-free 70/20/10 train/val/test split.",
        "atomic_points": [
            "Evaluated on NEU-DET surface defect benchmark dataset",
            "Achieved 61.7% mAP50 detection accuracy metric",
            "Partitioned dataset into leak-free 70/20/10 train/val/test split",
            "Prevented data contamination across split partitions"
        ],
        "cases": [
            {
                "case_type": "right_numbers_wrong_subject",
                "answer": "We split our Redis cache keys across a 70/20/10 cluster ratio and reached a 61.7% cache hit rate under Locust benchmarking.",
            },
            {
                "case_type": "correct_subject_wrong_numbers",
                "answer": "We evaluated defect inspection on NEU-DET using a 90/5/5 data split, achieving 99.4% mAP50 accuracy with zero errors.",
            },
            {
                "case_type": "contradictory_statements",
                "answer": "We tested on NEU-DET but mixed all training and test images into a single set without splitting, which invalidates all 61.7% mAP metrics.",
            },
            {
                "case_type": "one_word_answer",
                "answer": "NEU-DET",
            },
            {
                "case_type": "confident_fabrication",
                "answer": "We fine-tuned a 70B parameter LLaMA 3 vision model across 16 NVIDIA A100 GPUs using Fully Sharded Data Parallelism.",
            }
        ]
    },
    "item_4_p2_ablation": {
        "title": "4-head P2 variant ablation (62.2% vs 57.3%)",
        "question": "What were the empirical results of benchmarking the 4-head P2 architectural variant?",
        "claim_text": "Benchmarked a 4-head P2 variant that gained on validation (62.2% vs 61.2%) but dropped to 57.3% on test, so the baseline was retained.",
        "atomic_points": [
            "Introduced 4-head P2 architectural modification for small defects",
            "Observed validation gain (62.2% vs 61.2% baseline)",
            "Observed test set generalization drop to 57.3%",
            "Retained baseline architecture based on test performance"
        ],
        "cases": [
            {
                "case_type": "right_numbers_wrong_subject",
                "answer": "We configured a 4-head Kafka partition topic where consumer lag was 61.2% on validation and dropped to 57.3% on production nodes.",
            },
            {
                "case_type": "correct_subject_wrong_numbers",
                "answer": "Our 4-head P2 variant achieved 94.5% validation accuracy compared to 80% baseline, and maintained 92% on the test set so we deployed it.",
            },
            {
                "case_type": "contradictory_statements",
                "answer": "We tested a 4-head P2 variant which showed no changes whatsoever on validation or test sets because ablation studies cannot measure neural heads.",
            },
            {
                "case_type": "one_word_answer",
                "answer": "Ablation",
            },
            {
                "case_type": "confident_fabrication",
                "answer": "We trained deep reinforcement learning agents with proximal policy optimization to automatically prune attention heads in real-time.",
            }
        ]
    },
    "item_5_onnx_latency": {
        "title": "ONNX export & AWS EC2 T4 1.6ms latency",
        "question": "How did you profile inference latency and what hardware benchmark did you record?",
        "claim_text": "Exported baseline to ONNX format and benchmarked GPU inference latency on an AWS EC2 Tesla T4 instance, profiling 1.6ms execution time per frame (416 × 416).",
        "atomic_points": [
            "Exported baseline model to ONNX runtime format",
            "Benchmarked on AWS EC2 Tesla T4 GPU instance",
            "Profiled 1.6ms execution latency per frame",
            "Evaluated at 416x416 frame resolution"
        ],
        "cases": [
            {
                "case_type": "right_numbers_wrong_subject",
                "answer": "Our SQLite database logged 1.6ms p99 write latency on an AWS EC2 instance running 416 concurrent threads.",
            },
            {
                "case_type": "correct_subject_wrong_numbers",
                "answer": "We exported to ONNX format and profiled GPU inference on an AWS T4 instance, measuring 45ms latency per frame at 1080p resolution.",
            },
            {
                "case_type": "contradictory_statements",
                "answer": "We exported to ONNX but inference latency cannot be benchmarked on GPUs because ONNX Runtime strictly disables GPU acceleration.",
            },
            {
                "case_type": "one_word_answer",
                "answer": "ONNX",
            },
            {
                "case_type": "confident_fabrication",
                "answer": "We compiled custom CUDA C++ kernels using TensorRT LLM with FP8 quantization running across 8 NVIDIA H100 SXM5 GPUs.",
            }
        ]
    }
}

def run_adversarial_discrimination():
    llm = OpenAICompatibleClient(base_url="http://127.0.0.1:11434")
    grader = AnswerGrader(llm_client=llm, model="qwen2.5:7b")

    all_results = {}
    print("=" * 80)
    print("RUNNING ADVERSARIAL DISCRIMINATION EVALUATION (5 ITEMS x 5 ADVERSARIAL CASES)")
    print("Model: qwen2.5:7b | Evaluator: AnswerGrader")
    print("=" * 80)

    total_cases = 0
    zero_scores = 0

    for item_id, item_data in ADVERSARIAL_TEST_CASES.items():
        print(f"\n--> {item_id}: {item_data['title']}")
        claim = Claim(
            id=f"claim_{item_id}",
            user_id="u_adv",
            project_id=item_id,
            claim_type=ClaimType.PROJECT_METRIC,
            text=item_data["claim_text"],
            tools=[],
            metric_values=[],
            status=ClaimStatus.CONFIRMED,
        )

        atomic_pts = [
            AtomicPoint(
                id=f"{item_id}_p{i}",
                text=pt,
                source_file="README.md",
                source_line=i + 1,
                source_quote=pt,
            )
            for i, pt in enumerate(item_data["atomic_points"])
        ]

        rubric = RubricItem(
            id=f"rubric_{item_id}",
            project_id=item_id,
            topic=item_data["title"],
            source_file="README.md",
            source_line=1,
            source_quote=item_data["atomic_points"][0],
            criteria={
                0: "No evidence of verified implementation or topic fabrication.",
                1: "Mentions basic mechanism but misses quantitative performance.",
                2: "Partially satisfies architectural details.",
                3: "Satisfies core claims and metrics.",
                4: "Fully demonstrates comprehensive verified implementation.",
            },
            atomic_points=atomic_pts,
        )

        item_records = []
        for c in item_data["cases"]:
            total_cases += 1
            start_t = time.perf_counter()
            grade_res = grader.grade_answer(
                claim=claim,
                question_text=item_data["question"],
                answer_text=c["answer"],
                rubric=rubric,
            )
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            score = grade_res.raw_score
            if score == 0:
                zero_scores += 1

            rec = {
                "case_type": c["case_type"],
                "answer": c["answer"],
                "score": score,
                "evidence_quote": grade_res.evidence_quote,
                "feedback": grade_res.feedback,
                "latency_ms": dur_ms,
            }
            item_records.append(rec)
            status_sym = "[BLOCKED / SCORE 0]" if score == 0 else f"[SCORE {score}]"
            print(f"    {status_sym} {c['case_type']}: Score {score} / 4 ({dur_ms}ms)")

        all_results[item_id] = {
            "title": item_data["title"],
            "cases": item_records
        }

    summary = {
        "date": "2026-10-28",
        "evaluator_model": "qwen2.5:7b",
        "total_adversarial_cases": total_cases,
        "zero_score_count": zero_scores,
        "adversarial_rejection_rate": zero_scores / total_cases if total_cases > 0 else 1.0,
        "items": all_results,
    }

    out_file = ROOT_DIR / "results" / "adversarial_discrimination_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"SUMMARY: {zero_scores} / {total_cases} adversarial cases scored 0 ({summary['adversarial_rejection_rate'] * 100:.1f}%)")
    print(f"Saved results to {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_adversarial_discrimination()
