"""추천 질문을 실제 대화 모델로 여러 번 돌려, 답변 문장의 숫자가 도구 결과(=차트와 같은 자료)에 있는 값인지 본다 (PRD A3).

    python scripts/check_answers.py [반복 횟수]

앱 서버가 8000번에서 돌고 API 키가 설정돼 있어야 한다. 실제 API를 부르므로 요금이 든다(질문마다 1~2회).
숫자가 도구 결과에 있는지만 본다. 그 숫자가 맞는 집단에 붙었는지는 사람이 읽어 봐야 한다.
"""
import json, re, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from suwon_chat.store import Store
from suwon_chat.predict import LookupBackend
from suwon_chat.service import Service
from suwon_chat.scenarios import SCENARIOS

st = Store(); sv = Service(st, LookupBackend(st))
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NUM = re.compile(r"-?\d+(?:\.\d+)?")

def numbers_in(obj, out):
    if isinstance(obj, bool) or obj is None: return
    if isinstance(obj, (int, float)): out.add(round(abs(float(obj)), 6))
    elif isinstance(obj, str):
        for m in NUM.findall(obj): out.add(round(abs(float(m)), 6))
    elif isinstance(obj, dict):
        for k, v in obj.items(): numbers_in(k, out); numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj: numbers_in(v, out)

def allowed_for(brief):
    base = set(); numbers_in(brief, base)
    ok = set(base)
    for v in base:                      # 같은 값을 덜 자세히 반올림해 쓴 것은 허용한다
        for d in (0, 1, 2, 3): ok.add(round(v, d))
        ok.add(round(v * 100, 1)); ok.add(round(v * 100, 0))     # WD 0.036 → 3.6%처럼 쓴 경우
    ok.update(float(i) for i in range(0, 13)); ok.update({2023.0, 2024.0, 2025.0, 100.0})
    return ok

def ask(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "panel": None}).encode()
    req = urllib.request.Request("http://localhost:8000/api/chat", body, {"content-type": "application/json"})
    text, panels, errors, event = "", [], [], None
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "): event = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if event == "text": text += d["delta"]
                elif event == "panel": panels.append((d["state"], d["compare"]))
                elif event == "error": errors.append(d)
    return text, panels, errors

total = bad = 0
for rnd in range(1, ROUNDS + 1):
    for s in SCENARIOS:
        text, panels, errors = ask(s["prompt"])
        allowed = set()
        for state, _ in panels: allowed |= allowed_for(sv.prediction_brief(**state))
        found = [round(abs(float(m)), 6) for m in NUM.findall(text)]
        unknown = sorted({v for v in found if v not in allowed}) if panels else []
        want = s["panel"]
        got = panels[-1] if panels else None
        panel_ok = (want is None) or (got and all(got[0][k] == want[k] for k in ("qkey", "attribute", "adapter")) and got[1] == want["compare"])
        total += 1
        flag = "OK " if not unknown and not errors and panel_ok and len(text) > 80 else "CHECK"
        if flag == "CHECK": bad += 1
        print(f"[{rnd}] {s['id']:4s} {flag} 숫자 {len(found)}개, 근거 없는 숫자 {unknown}, 패널 {'일치' if panel_ok else got}, 오류 {errors}")
        if unknown:
            for u in unknown:
                for m in re.finditer(re.escape(str(u).rstrip('0').rstrip('.')), text):
                    print("       …" + text[max(0, m.start()-45): m.end()+25].replace("\n", " ") + "…"); break
print(f"{total}건 가운데 확인 필요 {bad}건")
