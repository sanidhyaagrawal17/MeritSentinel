import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.grader import AnswerGrader
from scripts.eval_5x5_discrimination import DISCRIMINATION_ITEMS

key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAICompatibleClient(base_url="https://openrouter.ai/api/v1", api_key=key)
grader = AnswerGrader(client, model="cohere/north-mini-code:free")

item = DISCRIMINATION_ITEMS[0]
for lvl in [0, 1, 2, 3, 4]:
    res = grader.grade_answer(
        claim=item["claim"],
        question_text=item["question"],
        answer_text=item["levels"][lvl],
        rubric=item["rubric"],
    )
    print(f"Level {lvl}: Score {res.raw_score} | Quote: '{res.evidence_quote[:40]}'")
