"""실시간 예측이 정리본에 저장된 예측과 같은 값을 내는지 확인한다 (PRD A6·A7).

    python -m suwon_chat.live_check                 # 빠른 점검: 어댑터마다 문항 × 속성 24쌍 (뺀 집단·문항을 반드시 포함)
    python -m suwon_chat.live_check --all           # 전부: 어댑터마다 3,729칸. 시연에 쓰기 전에 한 번은 돌린다
    python -m suwon_chat.live_check --adapter exp1 --pairs 5
    python -m suwon_chat.live_check --csv 결과.csv   # 칸마다의 값을 파일로 남긴다

칸마다 "저장된 예측과 실시간 예측 사이의 거리"를 잰다(순서형·이분형은 WD, 명목형은 TV. 0이면 같다).
어댑터 예측과 학습 전 모델 예측을 따로 재고, 어댑터가 제대로 바뀌어 끼워졌는지도 본다
(exp2로 돌린 값이 저장된 exp1보다 저장된 exp2에 더 가까워야 한다).

저장된 예측은 연구실 GPU 서버에서 돌린 값이다. 장치가 다르면 조금 다르게 나온다. 정리본은 GPU 종류만
달라도 칸당 WD 평균이 0.0025쯤 달라진다고 적었다. 통과 기준은 칸별 거리의 평균 0.005 이하다.
참고로 저장된 exp1과 exp2의 예측 사이 거리는 평균 0.018쯤이라, 어댑터가 뒤바뀌면 이 기준에 걸린다.
4비트로 올리면 기준을 넘을 수 있다.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time

from . import config
from .live import LiveBackend, LiveError
from .metrics import distance
from .store import Store

TOLERANCE = 0.005


def pick_pairs(store: Store, adapter: str, n: int, seed: int) -> list[tuple[str, str]]:
    """문항 × 속성 쌍을 뽑는다. 그 어댑터가 학습에서 뺀 집단의 속성과 뺀 문항을 먼저 넣는다."""
    rng = random.Random(seed)
    questions = [q.qkey for q in store.target_questions]
    attributes = list(store.groups)
    spec = config.ADAPTERS[adapter]
    must = [(rng.choice(questions), a) for a, _ in spec["held_out_groups"]]
    must += [(q, rng.choice(attributes)) for q in spec["held_out_questions"]]
    kinds = {}
    for q in store.target_questions:                    # 문항 종류마다 하나씩
        kinds.setdefault(q.scale_type, (q.qkey, rng.choice(attributes)))
    must += list(kinds.values())
    rest = [(q, a) for q in questions for a in attributes]
    rng.shuffle(rest)
    out = list(dict.fromkeys(must + rest))
    return out[:max(n, len(dict.fromkeys(must)))]


def main() -> int:
    ap = argparse.ArgumentParser(description="실시간 예측을 저장된 예측과 견준다")
    ap.add_argument("--pairs", type=int, default=24, help="어댑터마다 볼 문항 × 속성 쌍의 수 (빠른 점검)")
    ap.add_argument("--all", action="store_true", help="전부 본다 (어댑터마다 3,729칸)")
    ap.add_argument("--adapter", choices=list(config.ADAPTERS), action="append", help="확인할 어댑터 (여러 번 줄 수 있다)")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--batch", type=int, default=1, help="한 번에 돌릴 프롬프트 수. 기본 1 (묶어 돌릴 때의 차이를 빼고 잰다)")
    ap.add_argument("--csv", help="칸마다의 결과를 쓸 파일")
    args = ap.parse_args()

    store = Store()
    backend = LiveBackend(store, batch_size=args.batch)
    print("모델을 올린다. 처음에는 가중치 약 16GB를 내려받는다…", flush=True)
    try:
        backend.load()
    except LiveError as e:
        print(f"실패: {e}", file=sys.stderr)
        return 2
    print("올렸다:", backend.describe(), flush=True)

    adapters = args.adapter or list(config.ADAPTERS)
    rows, ok = [], True
    for adapter in adapters:
        pairs = ([(q.qkey, a) for q in store.target_questions for a in store.groups] if args.all
                 else pick_pairs(store, adapter, args.pairs, args.seed))
        d_lora, d_base, d_to_saved = [], [], {a: [] for a in config.ADAPTERS}
        e_live, e_saved, worst = [], [], (0.0, "")
        t0 = time.time()
        for n, (qkey, attribute) in enumerate(pairs, 1):
            q = store.questions[qkey]
            got = backend.predict(qkey, attribute, adapter)
            for group, dists in got.items():
                saved = store.predictions[(adapter, qkey, attribute, group)]
                dl = distance(saved["lora"], dists["lora"], q.ordinal)
                db = distance(saved["base"], dists["base"], q.ordinal)
                d_lora.append(dl); d_base.append(db)
                for other in config.ADAPTERS:        # 어댑터가 제대로 끼워졌는지: 저장된 각 어댑터의 예측과의 거리
                    d_to_saved[other].append(distance(store.predictions[(other, qkey, attribute, group)]["lora"],
                                                      dists["lora"], q.ordinal))
                e_live.append(distance(saved["gold"], dists["lora"], q.ordinal))
                e_saved.append(distance(saved["gold"], saved["lora"], q.ordinal))
                if dl > worst[0]:
                    worst = (dl, f"{qkey} · {attribute} · {group}")
                rows.append({"adapter": adapter, "qkey": qkey, "attribute": attribute, "group": group,
                             "distance_adapter": round(dl, 5), "distance_base": round(db, 5),
                             "live": dists["lora"], "saved": saved["lora"]})
            if n % 10 == 0 or n == len(pairs):
                done = time.time() - t0
                print(f"  [{adapter}] {n}/{len(pairs)}쌍, {done:.0f}초 지남, 남은 시간 약 {done / n * (len(pairs) - n):.0f}초", flush=True)
        cells = len(d_lora)
        mean = lambda v: sum(v) / len(v)
        nearest = min(d_to_saved, key=lambda a: mean(d_to_saved[a]))
        checks = {
            "어댑터 예측이 저장된 값과 같다": mean(d_lora) <= TOLERANCE,
            "학습 전 모델 예측이 저장된 값과 같다": mean(d_base) <= TOLERANCE,
            "맞는 어댑터가 끼워졌다": nearest == adapter,
        }
        ok = ok and all(checks.values())
        seconds = time.time() - t0
        print(f"\n[{adapter}] {cells}칸 ({len(pairs)}쌍), {seconds:.0f}초 (쌍당 {seconds / len(pairs):.1f}초. 학습 전 모델 계산 포함)")
        print(f"  저장된 예측과의 거리(칸 평균)  어댑터 {mean(d_lora):.4f} · 학습 전 모델 {mean(d_base):.4f}   (기준 {TOLERANCE} 이하)")
        print(f"  가장 많이 다른 칸  {worst[0]:.4f}  {worst[1]}")
        print("  저장된 각 어댑터의 예측과의 거리  " + " · ".join(f"{a} {mean(v):.4f}" for a, v in d_to_saved.items())
              + f"   → 가장 가까운 것 {nearest}")
        print(f"  실측과의 오차 평균  실시간 {mean(e_live):.4f} · 저장된 예측 {mean(e_saved):.4f}")
        for name, passed in checks.items():
            print(f"  {'통과' if passed else '실패'}  {name}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("칸마다의 결과:", args.csv)
    if not ok:
        print("\n결과: 실패. 저장된 예측과 다르다. 장치·정밀도(4비트 여부)를 확인하고, 시연에는 조회 버전을 쓴다.")
        return 1
    if args.all and set(adapters) == set(config.ADAPTERS):
        print("\n결과: 통과. 세 어댑터의 모든 칸이 저장된 예측과 맞는다. 실시간 예측을 시연에 써도 된다.")
    else:
        print("\n결과: 빠른 점검 통과. 본 것은 일부다. 시연에 쓰기 전에 `--all`로 전부 확인한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
