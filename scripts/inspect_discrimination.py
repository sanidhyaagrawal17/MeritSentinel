import json

with open("results/discrimination_5x5_results.json") as f:
    d = json.load(f)

print("LOCAL 7B:")
for item_id, item_data in d["local_7b"]["results"].items():
    print(f"=== {item_id} ===")
    for lvl in ["level_0", "level_1", "level_2", "level_3", "level_4"]:
        s = item_data[lvl]["score"]
        q = item_data[lvl]["evidence_quote"][:50]
        fb = item_data[lvl]["feedback"]
        print(f"  {lvl}: score={s} | quote={q!r} | fb={fb}")
