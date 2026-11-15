"""추천 질문을 실제 대화 모델로 한 번씩 돌려 답변을 저장한다 (PRD F-04, A5).

    python scripts/save_answers.py [시나리오 id …]     # id를 주면 그것만 새로 받는다

앱 서버가 8000번에서 돌고 API 키가 설정돼 있어야 한다. 결과는 backend/suwon_chat/saved_answers.json에 쓴다.
저장한 뒤에는 사람이 읽고 고친다. 대화 모델을 쓸 수 없을 때 이 글이 그대로 화면에 나간다.
"""
import datetime
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend" / "suwon_chat" / "saved_answers.json"
SCENARIOS = json.loads((ROOT / "backend" / "suwon_chat" / "scenarios.json").read_text(encoding="utf-8"))


def ask(prompt: str) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "panel": None}).encode()
    req = urllib.request.Request("http://localhost:8000/api/chat", body, {"content-type": "application/json"})
    text, panel, event = "", None, None
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if event == "text":
                    text += d["delta"]
                elif event == "panel":
                    panel = {**d["state"], "compare": d["compare"]}
                elif event == "saved":
                    sys.exit("대화 모델 대신 저장된 답변이 돌아왔다. API 키와 연결을 확인한다.")
                elif event == "error":
                    sys.exit(f"대화 실패: {d}")
    return {"text": text.strip(), "panel": panel}


def main() -> None:
    only = set(sys.argv[1:])
    saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    status = json.load(urllib.request.urlopen("http://localhost:8000/api/chat/status"))
    for s in SCENARIOS:
        if only and s["id"] not in only:
            continue
        answer = ask(s["prompt"])
        want = s["panel"]
        if want and (answer["panel"] is None or any(answer["panel"][k] != want[k] for k in want)):
            print(f"  주의: {s['id']}의 답변이 띄운 장면이 추천 질문의 장면과 다르다: {answer['panel']}")
        if want:
            answer["panel"] = dict(want)          # 장면은 추천 질문에 정해 둔 것으로 고정한다
        saved[s["id"]] = {"prompt": s["prompt"], **answer, "model": status["model"],
                          "saved_at": datetime.date.today().isoformat(), "reviewed": False}
        print(f"{s['id']}: {len(answer['text'])}자")
    OUT.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
