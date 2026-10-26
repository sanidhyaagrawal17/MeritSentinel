import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("./src").resolve()))
from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.grader import AnswerGrader
from scripts.eval_5x5_discrimination import DISCRIMINATION_ITEMS

client = OpenAICompatibleClient(base_url="http://127.0.0.1:11434")
grader = AnswerGrader(client, "qwen2.5:7b")

cells = [
    (0, 1), # item 1, level 1
    (1, 1), # item 2, level 1
    (3, 1), # item 4, level 1
    (1, 3), # item 2, level 3
    (2, 3), # item 3, level 3
]

for item_idx, lvl in cells:
    it = DISCRIMINATION_ITEMS[item_idx]
    ans = it["levels"][lvl]
    res = grader.grade_answer(it["claim"], it["question"], ans, it["rubric"])
    print(f"\n=== {it['id']} Level {lvl} ===")
    print(f"Answer: {ans}")
    print(f"Score: {res.raw_score}")
    for pid, pe in res.point_evaluations.items():
        print(f"  Point {pid}: matched={pe.matched}, quote={pe.evidence_quote!r}")
