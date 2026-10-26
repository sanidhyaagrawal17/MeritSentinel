"""Runs 5 items x 5 levels 3 times per cell on local qwen2.5:7b, computes mean scores and verifies monotonicity."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("./src").resolve()))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.grader import AnswerGrader
from scripts.eval_5x5_discrimination import DISCRIMINATION_ITEMS

client = OpenAICompatibleClient(base_url="http://127.0.0.1:11434")
grader = AnswerGrader(client, "qwen2.5:7b")

print("=" * 70)
print("RUNNING 5x5 DISCRIMINATION TEST (3 REPETITIONS PER CELL)")
print("=" * 70)

cell_runs = {} # (item_id, level) -> [s1, s2, s3]
item_level_means = {} # item_id -> [mean_L0, mean_L1, mean_L2, mean_L3, mean_L4]
monotonic_flags = {}

for item in DISCRIMINATION_ITEMS:
    item_id = item["id"]
    item_level_means[item_id] = []
    print(f"\n--> {item_id}:")
    
    for level in [0, 1, 2, 3, 4]:
        ans = item["levels"][level]
        scores = []
        for rep in range(3):
            res = grader.grade_answer(
                claim=item["claim"],
                question_text=item["question"],
                answer_text=ans,
                rubric=item["rubric"],
            )
            scores.append(res.raw_score)
        
        mean_score = sum(scores) / len(scores)
        cell_runs[(item_id, level)] = scores
        item_level_means[item_id].append(mean_score)
        level_name = ["None (0)", "1/3 (1)", "2/3 (2)", "Terse (3)", "Detailed (4)"][level]
        print(f"    Level {level} [{level_name}]: Scores={scores} -> Mean={mean_score:.2f}")

    means = item_level_means[item_id]
    is_mono = all(means[i] <= means[i+1] for i in range(len(means) - 1))
    monotonic_flags[item_id] = is_mono
    print(f"    Monotonic: {is_mono} (Trajectory: {means})")

# Mean score by coverage level across all 5 items
print("\n" + "=" * 70)
print("MEAN SCORE BY COVERAGE LEVEL (ACROSS ALL 5 ITEMS)")
print("=" * 70)
coverage_level_means = []
for lvl in range(5):
    all_scores_at_lvl = [item_level_means[it["id"]][lvl] for it in DISCRIMINATION_ITEMS]
    avg = sum(all_scores_at_lvl) / len(all_scores_at_lvl)
    coverage_level_means.append(avg)
    lvl_name = ["Level 0 (None)", "Level 1 (1/3)", "Level 2 (2/3)", "Level 3 (Terse)", "Level 4 (Detailed)"][lvl]
    print(f"  {lvl_name}: Mean = {avg:.2f} / 4.0 (Values across items: {all_scores_at_lvl})")

mono_count = sum(1 for m in monotonic_flags.values() if m)
print(f"\nMonotonic Count: {mono_count} / {len(DISCRIMINATION_ITEMS)} ({mono_count / len(DISCRIMINATION_ITEMS) * 100:.1f}%)")

results_payload = {
    "date": "2026-10-26",
    "model": "qwen2.5:7b",
    "repetitions_per_cell": 3,
    "item_level_means": item_level_means,
    "coverage_level_means": coverage_level_means,
    "monotonic_flags": monotonic_flags,
    "monotonic_count": mono_count,
    "total_items": len(DISCRIMINATION_ITEMS),
}

with open("results/discrimination_5x5_3x_results.json", "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2)

print("\nSaved results to results/discrimination_5x5_3x_results.json")
