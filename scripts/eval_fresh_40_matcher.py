"""Evaluates HybridMatcher on the fresh, held-out 40-pair benchmark using frozen pre-set cutoffs (0.35 strong, 0.20 partial)."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("./src").resolve()))

from resumeproof.tailoring.matcher import HybridMatcher, PINNED_EMBEDDING_MODEL

matcher = HybridMatcher()
fixture_path = Path("tests/fixtures/fresh_40_matcher_benchmark.json")
pairs = json.loads(fixture_path.read_text(encoding="utf-8"))

STRONG_CUTOFF = 0.35
PARTIAL_CUTOFF = 0.20

print("=" * 80)
print(f"FRESH 40-PAIR HELD-OUT BENCHMARK EVALUATION (FROZEN CUTOFFS)")
print(f"Model: {PINNED_EMBEDDING_MODEL} | Strong Cutoff: {STRONG_CUTOFF} | Partial Cutoff: {PARTIAL_CUTOFF}")
print("=" * 80)

results = []
scores_by_label = {"strong": [], "partial": [], "gap": []}

for p in pairs:
    score = matcher.match(p["req"], p["claim"])
    pred = "strong" if score.total >= STRONG_CUTOFF else ("partial" if score.total >= PARTIAL_CUTOFF else "gap")
    scores_by_label[p["label"]].append(score.total)
    results.append({
        "id": p["id"],
        "actual": p["label"],
        "predicted": pred,
        "score": round(score.total, 4),
        "kw": round(score.keyword_score, 4),
        "bm25": round(score.bm25_score, 4),
        "emb": round(score.embedding_score, 4),
        "req": p["req"],
        "claim": p["claim"],
    })

classes = ["strong", "partial", "gap"]
metrics = {}

for cls in classes:
    tp = sum(1 for r in results if r["actual"] == cls and r["predicted"] == cls)
    fp = sum(1 for r in results if r["actual"] != cls and r["predicted"] == cls)
    fn = sum(1 for r in results if r["actual"] == cls and r["predicted"] != cls)
    tn = sum(1 for r in results if r["actual"] != cls and r["predicted"] != cls)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics[cls] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "fpr": fpr}
    print(f"Class '{cls.upper()}': GroundTruth={tp+fn}, TP={tp}, FP={fp}, FN={fn}, TN={tn}, Prec={prec:.4f}, Rec={rec:.4f}, FPR={fpr:.4f}")

mean_strong = sum(scores_by_label["strong"]) / len(scores_by_label["strong"])
mean_partial = sum(scores_by_label["partial"]) / len(scores_by_label["partial"])
mean_gap = sum(scores_by_label["gap"]) / len(scores_by_label["gap"])

print(f"\nScore Distributions:")
print(f"  Strong Mean:  {mean_strong:.4f} (Min: {min(scores_by_label['strong']):.4f}, Max: {max(scores_by_label['strong']):.4f})")
print(f"  Partial Mean: {mean_partial:.4f} (Min: {min(scores_by_label['partial']):.4f}, Max: {max(scores_by_label['partial']):.4f})")
print(f"  Gap Mean:     {mean_gap:.4f} (Min: {min(scores_by_label['gap']):.4f}, Max: {max(scores_by_label['gap']):.4f})")
print(f"  Strong-to-Gap Margin: {mean_strong - mean_gap:.4f}")

misclassified = [r for r in results if r["actual"] != r["predicted"]]
print(f"\nMisclassified Pairs: {len(misclassified)} / {len(pairs)}")
for m in misclassified:
    print(f"  [{m['id']}] Actual: {m['actual'].upper()} -> Predicted: {m['predicted'].upper()} (Score: {m['score']})")
    print(f"      Req:   \"{m['req']}\"")
    print(f"      Claim: \"{m['claim']}\"")

# Save results
out_data = {
    "date": "2026-10-26",
    "cutoffs": {"strong": STRONG_CUTOFF, "partial": PARTIAL_CUTOFF},
    "metrics": metrics,
    "means": {"strong": mean_strong, "partial": mean_partial, "gap": mean_gap},
    "misclassified_count": len(misclassified),
    "total_pairs": len(pairs),
}
Path("results/fresh_40_matcher_results.json").write_text(json.dumps(out_data, indent=2), encoding="utf-8")
print("\nSaved output to results/fresh_40_matcher_results.json")
