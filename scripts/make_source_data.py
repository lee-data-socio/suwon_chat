"""정리본에서 앱과 테스트에 필요한 파일만 골라 source_data/ 로 복사한다 (저장소에 올리는 묶음).

    python scripts/make_source_data.py

폴더 구조는 정리본과 같다. 정리본 폴더가 없는 PC에서는 앱이 source_data/ 를 대신 읽는다.
저장된 예측 파일(pred_*.json)의 `adapter` 값은 연구실 서버의 경로(계정 이름 포함)라서 폴더 이름만 남긴다.
나머지 파일은 그대로 복사한다.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "수원서베이_학습결과_정리본"
OUT = ROOT / "source_data"

COPY = [
    "07_데이터/suwon-v2-full/processed/suwon-v2-full.csv",        # 3개 연도의 집단별 응답 분포
    "07_데이터/suwon-v2-full/processed/opnqa_QA_test.csv",        # 평가용 프롬프트 (프롬프트 대조 테스트)
    "07_데이터/suwon-v2-exp2/processed/opnqa_QA_test.csv",
    "07_데이터/suwon-v2-exp3/processed/opnqa_QA_test.csv",
    "07_데이터/suwon/codebook/문항표_v2.csv",
    "07_데이터/suwon/processed/집단별_응답자수_v2.csv",
    "07_데이터/suwon/processed/lower_bound_v2_2025_cells.csv",
    "07_데이터/subpopulation_metadata/demographics_suwon_v2.csv",
    "07_데이터/subpopulation_metadata/steering_prompts_suwon_v2.json",
    "01_결과표/시험1_미래예측/칸단위.csv",                          # 칸 단위 오차 (숫자 대조 테스트)
    "01_결과표/시험2_미관측집단/칸단위.csv",
    "01_결과표/시험3_새질문/칸단위.csv",
]
PREDICTIONS = [
    "03_예측/시험1_미래예측/pred_bllossom-8b-b128-lr5e-5_suwon-v2-full_1003857.json",
    "03_예측/시험2_미관측집단/pred_bllossom-8b-b128-lr5e-5_suwon-v2-exp2_1006787.json",
    "03_예측/시험3_새질문/pred_bllossom-8b-b128-lr5e-5_suwon-v2-exp3_1006789.json",
]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"정리본 폴더가 없다: {SRC}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for rel in COPY:
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, OUT / rel)
    for rel in PREDICTIONS:
        data = json.loads((SRC / rel).read_text(encoding="utf-8"))
        data["adapter"] = "/".join(Path(data["adapter"]).parts[-2:])      # 서버 경로에서 실행 폴더 이름만 남긴다
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 앱이 읽는 원본 자료\n\n"
        "`수원서베이_학습결과_정리본`에서 앱과 테스트에 필요한 파일만 골라 같은 폴더 구조로 복사한 것이다.\n"
        "`scripts/make_source_data.py`가 만든다. 직접 고치지 않는다.\n\n"
        "- `07_데이터/`: 2023~2025년 수원서베이의 집단별 응답 분포(개인 단위 자료는 없다), 문항표, 집단 정보, 평가용 프롬프트\n"
        "- `03_예측/`: 학습한 어댑터 세 개가 연구실 GPU 서버에서 낸 예측. `adapter` 값의 서버 경로만 폴더 이름으로 줄였다\n"
        "- `01_결과표/`: 칸 단위 오차. 앱이 계산한 값과 견주는 테스트에 쓴다\n", encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"{len(files)}개 파일, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB → {OUT}")


if __name__ == "__main__":
    main()
