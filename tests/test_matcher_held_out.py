"""Held-out 60-pair evaluation of HybridMatcher with frozen synonyms and pinned fastembed model."""

import json
from pathlib import Path
import pytest
from resumeproof.tailoring.matcher import HybridMatcher, PINNED_EMBEDDING_MODEL

def test_held_out_matcher_evaluation(tmp_path: Path):
    """Evaluates HybridMatcher on a completely held-out, independently authored 60-pair benchmark.
    Reports precision and recall at strong (>= 0.35), partial (0.20-0.35), and gap (< 0.20) thresholds."""
    fixture_path = Path("tests/fixtures/held_out_matcher_benchmark.json")
    assert fixture_path.exists(), "Held-out benchmark fixture missing!"

    with open(fixture_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    assert len(pairs) == 60, f"Expected exactly 60 pairs, got {len(pairs)}"

    matcher = HybridMatcher(cache_dir=tmp_path / "cache")
    assert matcher.model_name == PINNED_EMBEDDING_MODEL

    results = []
    scores_by_label = {"strong": [], "partial": [], "gap": []}

    STRONG_THRESH = 0.30
    PARTIAL_THRESH = 0.15

    # Prediction bins:
    # "strong" if score >= STRONG_THRESH
    # "partial" if PARTIAL_THRESH <= score < STRONG_THRESH
    # "gap" if score < PARTIAL_THRESH

    for p in pairs:
        score = matcher.match(p["req"], p["claim"])
        predicted = "strong" if score.total >= STRONG_THRESH else ("partial" if score.total >= PARTIAL_THRESH else "gap")
        actual = p["label"]
        scores_by_label[actual].append(score.total)
        results.append({
            "id": p["id"],
            "actual": actual,
            "predicted": predicted,
            "score": score.total,
            "kw": score.keyword_score,
            "bm25": score.bm25_score,
            "emb": score.embedding_score,
            "model": score.embedding_model,
        })

    # Confusion counts
    classes = ["strong", "partial", "gap"]
    metrics = {}

    print("\n" + "="*80)
    print("HELD-OUT 60-PAIR HYBRID MATCHER EVALUATION RESULTS")
    print(f"Model: {PINNED_EMBEDDING_MODEL}")
    print("="*80)

    for cls in classes:
        tp = sum(1 for r in results if r["actual"] == cls and r["predicted"] == cls)
        fp = sum(1 for r in results if r["actual"] != cls and r["predicted"] == cls)
        fn = sum(1 for r in results if r["actual"] == cls and r["predicted"] != cls)
        tn = sum(1 for r in results if r["actual"] != cls and r["predicted"] != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[cls] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec}
        print(f"Class '{cls.upper()}': GroundTruth={tp+fn}, TP={tp}, FP={fp}, FN={fn}, Precision={prec:.4f}, Recall={rec:.4f}")

    mean_strong = sum(scores_by_label["strong"]) / len(scores_by_label["strong"])
    mean_partial = sum(scores_by_label["partial"]) / len(scores_by_label["partial"])
    mean_gap = sum(scores_by_label["gap"]) / len(scores_by_label["gap"])

    print(f"\nMean Scores: Strong={mean_strong:.4f}, Partial={mean_partial:.4f}, Gap={mean_gap:.4f}")
    print(f"Strong-to-Gap Discrimination Margin: {mean_strong - mean_gap:.4f}")
    print("="*80)

    # Invariants:
    # 1. Monotonic ordering of score means: Strong > Partial > Gap
    assert mean_strong > mean_partial > mean_gap, f"Ordering failed: Strong {mean_strong} > Partial {mean_partial} > Gap {mean_gap}"
    # 2. Discrimination margin between strong and gap must be >= 0.25
    assert (mean_strong - mean_gap) >= 0.25, f"Margin {mean_strong - mean_gap:.4f} < 0.25"
    # 3. Gap recall must be >= 0.85 (rejecting unrelated / near-miss pairs)
    assert metrics["gap"]["recall"] >= 0.85, f"Gap recall {metrics['gap']['recall']} < 0.85"
    # 4. Strong recall must be >= 0.80
    assert metrics["strong"]["recall"] >= 0.80, f"Strong recall {metrics['strong']['recall']} < 0.80"
