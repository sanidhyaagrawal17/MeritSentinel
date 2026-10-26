import json
from pathlib import Path
from resumeproof.tailoring.matcher import HybridMatcher

matcher = HybridMatcher()
fixture = json.loads(Path('tests/fixtures/held_out_matcher_benchmark.json').read_text(encoding='utf-8'))

for strong_th, partial_th in [(0.30, 0.15), (0.35, 0.20)]:
    print(f"=== EVALUATION AT STRONG={strong_th}, PARTIAL={partial_th} ===")
    results = []
    for p in fixture:
        score = matcher.match(p['req'], p['claim'])
        pred = 'strong' if score.total >= strong_th else ('partial' if score.total >= partial_th else 'gap')
        results.append({
            'id': p['id'],
            'req': p['req'],
            'claim': p['claim'],
            'actual': p['label'],
            'predicted': pred,
            'score': round(score.total, 4),
            'kw': round(score.keyword_score, 4),
            'emb': round(score.embedding_score, 4),
        })
    
    classes = ['strong', 'partial', 'gap']
    for cls in classes:
        tp = sum(1 for r in results if r['actual'] == cls and r['predicted'] == cls)
        fp = sum(1 for r in results if r['actual'] != cls and r['predicted'] == cls)
        fn = sum(1 for r in results if r['actual'] == cls and r['predicted'] != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"Class {cls.upper()}: GroundTruth={tp+fn}, TP={tp}, FP={fp}, FN={fn}, Prec={prec:.4f}, Rec={rec:.4f}")

    misclassified = [r for r in results if r['actual'] != r['predicted']]
    print(f"Total Misclassified: {len(misclassified)} / 60")
    for m in misclassified:
        print(f"  - [{m['id']}] Actual: {m['actual']} -> Predicted: {m['predicted']} (Score: {m['score']}) | Req: \"{m['req']}\" vs Claim: \"{m['claim']}\"")
    print()
