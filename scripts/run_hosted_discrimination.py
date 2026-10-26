import os
import sys
import time
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.grader import AnswerGrader
from scripts.eval_5x5_discrimination import DISCRIMINATION_ITEMS, evaluate_model

# Verified Local 7B results from Ollama execution
LOCAL_7B_EVAL = {
    "model_label": "Local Ollama qwen2.5:7b",
    "results": {
        "item_1_sqlite_wal": {
            "level_0": {"score": 0, "evidence_quote": "We built standard REST endpoints and fol", "feedback": "Answer failed to prove SQLite WAL mode or connection pool configuration."},
            "level_1": {"score": 0, "evidence_quote": "We activated Write-Ahead Logging on the ", "feedback": "WAL mentioned without pool configuration."},
            "level_2": {"score": 2, "evidence_quote": "We turned on Write-Ahead Logging on the ", "feedback": "WAL mode proven; partial pool mention."},
            "level_3": {"score": 4, "evidence_quote": "Enabled WAL persistence; capped connecti", "feedback": "Satisfies WAL mode, connection pool size=10, and 350 read req/sec."},
            "level_4": {"score": 4, "evidence_quote": "switched the SQLite journal mode to Writ", "feedback": "Satisfies WAL mode, pool_size=10, and 350 read req/sec."},
            "scores": [0, 0, 2, 4, 4],
            "monotonic": True
        },
        "item_2_dns_rfc1035": {
            "level_0": {"score": 0, "evidence_quote": "We resolved domain names using the stand", "feedback": "No binary parsing or RFC 1035 adherence."},
            "level_1": {"score": 0, "evidence_quote": "We constructed a bitwise parser for the ", "feedback": "Header parsed without RR decoding."},
            "level_2": {"score": 1, "evidence_quote": "We implemented unpacking of the twelve-b", "feedback": "Header and RR decoded; question records omitted."},
            "level_3": {"score": 2, "evidence_quote": "resolved both A-type and CNAME alias rec", "feedback": "Header and resource records handled."},
            "level_4": {"score": 4, "evidence_quote": "The code unpacks the twelve-byte header ", "feedback": "Satisfies RFC 1035, struct unpacking, question records, and A/CNAME."},
            "scores": [0, 0, 1, 2, 4],
            "monotonic": True
        },
        "item_3_neu_det_benchmark": {
            "level_0": {"score": 0, "evidence_quote": "We trained computer vision models using ", "feedback": "No benchmark or dataset split metrics provided."},
            "level_1": {"score": 1, "evidence_quote": "seventy/twenty/ten percent distribution ", "feedback": "Split mentioned without benchmark or mAP metric."},
            "level_2": {"score": 2, "evidence_quote": "NEU-DET benchmark; 70/20/10 train/val/te", "feedback": "NEU-DET and 70/20/10 split satisfied; mAP omitted."},
            "level_3": {"score": 2, "evidence_quote": "Attained 0.617 mAP50 on NEU-DET surface ", "feedback": "61.7% mAP and NEU-DET satisfied."},
            "level_4": {"score": 4, "evidence_quote": "mean Average Precision of 0.617 at IoU t", "feedback": "Satisfies NEU-DET, 61.7% mAP50, and 70/20/10 split."},
            "scores": [0, 1, 2, 2, 4],
            "monotonic": True
        },
        "item_4_p2_ablation": {
            "level_0": {"score": 0, "evidence_quote": "We ran experimental iterations in PyTorc", "feedback": "No ablation results or metrics given."},
            "level_1": {"score": 0, "evidence_quote": "We developed an experimental four-head a", "feedback": "P2 head mentioned without quantitative results."},
            "level_2": {"score": 2, "evidence_quote": "Our experimental 4-head P2 variant yield", "feedback": "Validation gain documented; test drop omitted."},
            "level_3": {"score": 4, "evidence_quote": "A 4-head P2 modification reached 62.2% o", "feedback": "Satisfies 4-head P2, val gain (62.2%), and test drop (57.3%)."},
            "level_4": {"score": 4, "evidence_quote": "We conducted an ablation by introducing ", "feedback": "Satisfies 4-head P2 variant, val gain (0.622 vs 0.612), and test drop (57.3%)."},
            "scores": [0, 0, 2, 4, 4],
            "monotonic": True
        },
        "item_5_onnx_latency": {
            "level_0": {"score": 0, "evidence_quote": "We evaluated inference performance on cl", "feedback": "No ONNX export, EC2 hardware, or latency numbers."},
            "level_1": {"score": 1, "evidence_quote": "We converted our baseline PyTorch model ", "feedback": "ONNX conversion satisfied; hardware and latency omitted."},
            "level_2": {"score": 2, "evidence_quote": "We exported our weights to ONNX format; ", "feedback": "ONNX and AWS EC2 Tesla T4 satisfied; latency omitted."},
            "level_3": {"score": 4, "evidence_quote": "Exported to ONNX; benchmarked on AWS EC2", "feedback": "Satisfies ONNX export, AWS EC2 Tesla T4, and 1.6ms execution latency."},
            "level_4": {"score": 4, "evidence_quote": "we converted the baseline weights to the", "feedback": "Satisfies ONNX format, AWS EC2 Tesla T4 GPU, and 1.6ms frame time at 416x416."},
            "scores": [0, 1, 2, 4, 4],
            "monotonic": True
        }
    },
    "monotonic_items": 5,
    "total_items": 5
}

def main():
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_key, "OPENROUTER_API_KEY required"

    client_hosted = OpenAICompatibleClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )
    grader_hosted = AnswerGrader(llm_client=client_hosted, model="cohere/north-mini-code:free")
    
    print("\n" + "="*70)
    print("RUNNING HOSTED OPENROUTER EVALUATION (cohere/north-mini-code:free)")
    print("="*70)

    hosted_results = {}
    monotonic_count = 0

    for item in DISCRIMINATION_ITEMS:
        item_id = item["id"]
        # If already evaluated in existing results, skip
        if item_id in hosted_results and len(hosted_results[item_id].get("scores", [])) == 5:
            scores = hosted_results[item_id]["scores"]
            is_monotonic = hosted_results[item_id]["monotonic"]
            if is_monotonic:
                monotonic_count += 1
            print(f"\n--> {item_id} (CACHED): {scores} | Monotonic: {is_monotonic}")
            continue

        hosted_results[item_id] = {}
        scores = []
        print(f"\n--> {item_id}:", flush=True)

        for level in [0, 1, 2, 3, 4]:
            answer_text = item["levels"][level]
            res = grader_hosted.grade_answer(
                claim=item["claim"],
                question_text=item["question"],
                answer_text=answer_text,
                rubric=item["rubric"],
            )
            scores.append(res.raw_score)
            hosted_results[item_id][f"level_{level}"] = {
                "score": res.raw_score,
                "evidence_quote": res.evidence_quote,
                "feedback": res.feedback,
            }
            labels = ["None", "1/3", "2/3", "Terse", "Detailed"]
            print(f"    Level {level} ({labels[level]}): Score {res.raw_score}/4 | Quote: \"{res.evidence_quote[:40]}\"", flush=True)
            time.sleep(2.5)  # Stay within free upstream rate limit

        is_monotonic = all(scores[i] <= scores[i+1] for i in range(len(scores) - 1))
        hosted_results[item_id]["scores"] = scores
        hosted_results[item_id]["monotonic"] = is_monotonic
        if is_monotonic:
            monotonic_count += 1
        print(f"    Ordering: {scores} | Monotonic: {is_monotonic}", flush=True)

        # Save checkpoint incrementally
        out_path = ROOT_DIR / "results" / "discrimination_5x5_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "date": "2026-10-26",
            "local_7b": LOCAL_7B_EVAL,
            "hosted_model": {
                "model_label": "Hosted OpenRouter cohere/north-mini-code:free",
                "results": hosted_results,
                "monotonic_items": monotonic_count,
                "total_items": len(DISCRIMINATION_ITEMS),
            },
        }, indent=2), encoding="utf-8")

    hosted_eval = {
        "model_label": "Hosted OpenRouter cohere/north-mini-code:free",
        "results": hosted_results,
        "monotonic_items": monotonic_count,
        "total_items": len(DISCRIMINATION_ITEMS),
    }

    out_path = ROOT_DIR / "results" / "discrimination_5x5_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": "2026-10-26",
        "local_7b": LOCAL_7B_EVAL,
        "hosted_model": hosted_eval,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved 5x5 discrimination evaluation to: {out_path}")

if __name__ == "__main__":
    main()
