"""대화 LLM (Claude API). 문항을 찾고 예측을 불러와 문장으로 설명한다.

숫자는 도구 결과에서만 나온다. 차트 자료는 대화 LLM을 거치지 않고 `panel` 이벤트로 화면에 직접 간다.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import AsyncIterator

import logging

from . import config
from .reports import SYNTHETIC_TAG, plain
from .service import NEW_MAX_OPTIONS, Invalid, NotFound, Service, Unsupported

log = logging.getLogger("suwon_chat.chat")

COMPARE = ("actual", "last_year", "reference", "zero_shot")   # 패널에 함께 띄울 비교 대상
MAX_TURNS = 6            # 한 질문에서 도구를 부르고 다시 답하는 횟수의 상한
HISTORY_MESSAGES = 24    # 모델에 넘기는 이전 대화의 최대 개수. 넘으면 오래된 것부터 버린다
MESSAGE_CHARS = 8000     # 메시지 하나의 최대 길이. 넘으면 뒤를 자른다


def normalize_history(history: list[dict]) -> list[dict]:
    """화면이 보낸 대화를 API가 받는 모양으로 다듬는다.

    빈 메시지를 버리고, 너무 긴 것은 자르고, 같은 쪽이 연달아 말한 것은 하나로 합치고,
    오래된 것부터 버린 뒤, 맨 앞이 사용자 메시지가 되게 한다.
    (API는 빈 글 블록과 assistant로 시작하는 대화를 받지 않는다. 답변이 오류로 끝난 질문 뒤에
    새 질문이 오면 사용자 메시지가 연달아 생기는데, 이것도 합쳐서 번갈아 말하는 모양으로 만든다.)
    """
    out: list[dict] = []
    for m in history:
        text = m.get("content", "").strip()[:MESSAGE_CHARS]
        if not text:
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{text}"[-MESSAGE_CHARS:]
        else:
            out.append({"role": m["role"], "content": text})
    out = out[-HISTORY_MESSAGES:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


class KeyFilter:
    """답변 글에서 문항 키(q189, q132_1_W2025)를 지운다. 지시해도 가끔 새어 나와서 글이 나가기 전에 한 번 더 거른다.

    글이 조각으로 오므로, 키가 조각 사이에 걸칠 수 있는 꼬리는 다음 조각이 올 때까지 붙들어 둔다.
    """
    KEY = re.compile(r"\s*\(\s*q\d+(?:_\d+)*(?:_W\d{4})?\s*\)|(?<![A-Za-z0-9_])q\d+(?:_\d+)*(?:_W\d{4})?(?![A-Za-z0-9_])\s?")
    TAIL = re.compile(r"\s*\(?\s*q?[\d_W]*\s*$")

    def __init__(self):
        self.held = ""

    def feed(self, text: str) -> str:
        text = self.held + text
        cut = self.TAIL.search(text).start()
        if len(text) - cut > 24:          # 키라고 보기에는 너무 긴 꼬리다
            cut = len(text)
        self.held, out = text[cut:], text[:cut]
        return self.KEY.sub("", out)

    def flush(self) -> str:
        out, self.held = self.KEY.sub("", self.held), ""
        return out


def chat_status() -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"available": False, "model": config.CHAT_MODEL,
                "reason": "ANTHROPIC_API_KEY가 설정되지 않았다. 추천 질문은 저장된 답변으로 볼 수 있다."}
    return {"available": True, "model": config.CHAT_MODEL, "reason": None}


def tools(service: Service, reports=None, concepts=None) -> list[dict]:
    meta = service.meta()
    report_tools = [] if not (reports and reports.available) else [{
        "name": "search_reports",
        "description": "참고 문서(연구 보고서)에서 검색어와 관련된 대목을 찾는다. 사용자가 관련 보고서나 자료를 물을 때 부른다. "
                       "부르면 찾은 대목이 화면의 참고 문서 칸에 카드로 뜬다. "
                       "참고 문서는 예측에 쓰이지 않았으므로 예측의 근거로 삼지 않는다.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "찾을 내용을 한 문장으로. 예: '점유형태별 주관적 계층 인식'"}},
            "required": ["query"], "additionalProperties": False,
        },
    }]
    new_tools = [] if not service.can_predict_new() else [{
        "name": "predict_new_question",
        "description": "목록에 없는 새 문항을 모델에 직접 물어 집단별 응답 분포를 예측한다. 실측이 없어 오차를 알 수 없는 "
                       "실험적 예측이다. 사용자가 새 문항의 문구와 보기를 주었을 때만 부른다. 보기를 주지 않았으면 "
                       "부르지 말고 보기를 되묻는다. 보기를 지어내지 않는다. '무응답'은 넣지 않는다(자동으로 붙는다).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "문항 문구. 조사표에 적을 그대로"},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": f"보기 2~{NEW_MAX_OPTIONS}개, 조사표 순서대로"},
                "ordered": {"type": "boolean", "description": "보기에 순서가 있으면 참(예: 전혀 아니다~매우 그렇다). 종류를 고르는 문항이면 거짓"},
                "off_scale_options": {"type": "array", "items": {"type": "string"},
                                      "description": "'잘 모르겠다'처럼 척도 밖에 있는 보기(options에 든 문구 그대로). 평균에서 뺀다. 없으면 빈 목록"},
                "attribute": {"type": "string", "enum": [a["id"] for a in meta["attributes"]]},
                "adapter": {"type": "string", "enum": [a["id"] for a in meta["adapters"]],
                            "description": "따로 말이 없으면 exp1(전체 학습 어댑터)"},
            },
            "required": ["question", "options", "ordered", "off_scale_options", "attribute", "adapter"],
            "additionalProperties": False,
        },
    }]
    return report_tools + new_tools + [
        {
            "name": "search_questions",
            "description": (
                "2025년 수원서베이 문항을 뜻으로 찾는다. 질문을 그대로 넣으면 개념 사전(조사 개념의 정의와 일상 표현)으로 "
                "의미가 가까운 문항을 가까운 순으로 돌려준다. 각 문항의 개념과 정의도 함께 온다."
                if concepts and concepts.available else
                "2025년 수원서베이 문항을 검색어로 찾는다. 문항 문구와 보기에 검색어가 글자 그대로 들어간 문항만 돌려준다. "
                "못 찾으면 다른 낱말로 다시 찾는다."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": (
                    "사용자의 질문이나 찾을 내용을 그대로. 예: '월세 사는 사람들 형편'" if concepts and concepts.available
                    else "띄어 쓴 검색어. 예: '버스 인프라'")}},
                "required": ["query"], "additionalProperties": False,
            },
        },
        {
            "name": "get_prediction",
            "description": "문항 하나를 한 속성의 집단별로 나눠, 어댑터의 예측 분포와 2025년 실측 분포, 오차를 가져온다. "
                           "부르면 화면 오른쪽 패널이 같은 문항·속성·어댑터의 차트로 바뀐다. "
                           "답변에 쓰는 숫자는 반드시 이 도구의 결과에서 가져온다.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "qkey": {"type": "string", "enum": [q["qkey"] for q in service.questions()],
                             "description": "문항 키. search_questions가 돌려준 값. 예: q189_W2025"},
                    "attribute": {"type": "string", "enum": [a["id"] for a in meta["attributes"]],
                                  "description": "집단을 나누는 속성 하나"},
                    "adapter": {"type": "string", "enum": [a["id"] for a in meta["adapters"]],
                                "description": "exp1 전체 학습 어댑터, exp2 집단 제외 어댑터, exp3 문항 제외 어댑터"},
                    "compare": {"type": "string", "enum": list(COMPARE),
                                "description": "예측과 2025년 실측 옆에 함께 띄울 비교 대상. "
                                               "actual: 없음(예측과 실측만). last_year: 2024년 실측(작년 값과 견줄 때). "
                                               "reference: 전체 학습 어댑터의 예측(exp2·exp3에서 뺀 집단·문항을 볼 때). "
                                               "zero_shot: 학습 전 모델의 예측. "
                                               "사용자가 따로 말하지 않으면 exp1은 actual, exp2·exp3은 reference를 쓴다."},
                },
                "required": ["qkey", "attribute", "adapter", "compare"], "additionalProperties": False,
            },
        },
    ]


REPORTS_ON = """- **수치를 말한 뒤 참고 문서로 해석을 붙인다.** `get_prediction` 결과에 `참고_문서`가 들어 있으면, 예측과 지난 조사 값을 짚은 다음 "해석" 문단을 하나 쓴다. 문서 첫머리의 개념과 문항 설명은 답변 첫머리(무엇을 묻는 문항인가)에 쓴다. 문서가 이 문항과 이 집단에 대해 밝힌 것(2023·2024년 조사 결과와 문서의 풀이)을 인용하고, 예측이 그 흐름과 같은 방향인지 다른 방향인지를 잇는다. 예: "「주거 점유형태에 따른 주관적 계층 인식」 3쪽은 2023·2024년 조사 모두 자가 거주자의 평가가 가장 높고 월세 및 사글세가 가장 낮았다고 정리했습니다. 예측도 같은 순서입니다." 실제 2025년 결과는 사용자가 물었을 때만 함께 잇는다. 듣는 사람이 숫자만이 아니라 그 숫자를 어떻게 읽을지까지 얻어 가게 하는 것이 목적이다. 해석은 2~4문장으로 쓴다.
- 인용할 때는 문서 제목을 「 」로 감싸고 쪽수를 붙인다(제목 앞의 "[시연용 가상 문서]"는 떼고 쓴다. 화면의 문서 카드에 그 표시가 붙는다). 쪽수는 그 내용이 실제로 있는 쪽을 적는다. 문서 안의 숫자는 2023·2024년 조사 값이므로 연도를 밝혀 쓰고, 2025년 예측·실측의 숫자와 구별되게 한다.
- 문서가 말한 만큼만 전한다. 범위를 넓히거나 없는 내용을 덧붙이지 않는다. 문서의 숫자끼리 빼거나 나눠서 문서에 없는 숫자("간격 약 1.5점")를 만들지 않는다. 문서가 "~일 수 있다"고 조심스럽게 쓴 풀이는 그 어조 그대로 전한다. 문서에 그 집단이나 속성에 대한 내용이 없으면 없는 대로, 문서가 다룬 범위에서만 해석한다.
- 해석은 당신이 문서와 수치를 나란히 놓고 하는 것이다. "모델이 이 문서를 보고 그렇게 예측했다"고 말하지 않는다. 모델은 문서를 본 적이 없으므로 사실이 아니다.
- `참고_문서`가 비어 있으면 해석 문단을 쓰지 않는다. 문서 없이 배경을 추측하지 않는다. 이때 "참고 문서가 없어 해석을 쓰지 않았다"처럼 사정을 설명하지도 않는다. 수치 설명으로 답을 끝낸다.
- 문서의 응답자 수와 2025년 응답자 수처럼 문서에 한쪽만 있는 값을 문서가 견준 것처럼 쓰지 않는다.
- 사용자가 관련 보고서를 따로 물으면 `search_reports`로 찾아 같은 방식으로 인용한다. 이때는 지금 연결된 문서가 시연을 위해 만든 가상 문서라는 점을 함께 말한다."""
REPORTS_OFF = "- 연구원 보고서 같은 참고 문서 검색은 아직 연결되지 않았다. 물으면 그렇게 답한다."
REPORTS_WITHHELD = ("- 지금은 근거를 견주어 보는 단계라 참고 문서 검색을 끈 상태다. 수치에 대한 해석 문단은 쓰지 않는다. "
                    "보고서나 해석의 근거를 물으면 이 단계에서는 참고 문서를 쓰지 않는다고 답한다.")

# 근거 단계. 같은 대화 모델에 근거를 얼마나 주느냐만 바꿔 답변이 어떻게 달라지는지 견준다.
#   none: 질문만 준다. 문항 목록·도구·화면 상태·참고 문서가 없다 ("범용 AI에 그냥 물었을 때")
#   data: 조사 데이터(예측·실측 도구)와 단어 검색만 준다. 개념 사전 검색과 참고 문서는 끈다 ("RAG 없이")
#   full: 전부 준다. 개념 사전으로 문항을 찾고 참고 문서로 해석한다 (평소 화면)
GROUNDING = ("none", "data", "full")
# none 단계의 시스템 프롬프트. 일부러 틀리게 답하게 하거나 모른다고 답하게 하지 않는다. 범용 AI의 평소 모습 그대로다.
# 길이만 맞춘다: 두 답을 나란히 놓고 읽을 수 있게 하려는 것이다.
GENERAL_PROMPT = "당신은 범용 AI 어시스턴트다. 한국어 존댓말로 답한다. 답변은 두세 문단 안으로 쓴다."


SOURCE_LOOKUP = "지금 화면의 예측은 모두 미리 계산해 저장해 둔 값이다."
SOURCE_LIVE = "지금 화면의 예측은 질문할 때마다 이 PC에서 모델을 직접 돌려 얻는 값이다(실시간). 그래서 답이 나오기까지 몇 초 걸릴 수 있다."
NEW_LOOKUP = """- **새 문항은 이 화면에서 예측하지 못한다.** 지금은 저장된 예측을 조회하는 버전이라 목록에 없는 문항은 답할 수 없다. 새 문항 예측은 모델을 직접 돌리는 실시간 버전에서 하며, 그 결과는 실측이 없어 검증되지 않은 실험적인 값이라고 알린다."""
NEW_LIVE = """- **새 문항 예측은 실험적인 값이다.** 목록에 없는 문항은 `predict_new_question`으로 예측할 수 있다. 사용자가 문항 문구와 보기를 주었을 때만 부르고, 보기가 없으면 되묻는다. 결과를 말할 때는 첫머리에 "실측이 없어 맞는지 확인할 수 없는 실험적 예측"이라고 밝힌다. 오차나 정확도를 말하지 않는다. 참고로 말할 수 있는 것은 실험 결과뿐이다: 학습에서 뺀 문항 4개를 예측했을 때 오차는 그 문항을 보고 학습했을 때보다 1.4배쯤 컸고, 문항에 따라 차이가 컸다. 새 문항의 예측에서도 집단 사이의 차이는 실제보다 작게 나올 가능성이 크므로 집단 차이를 단정하지 않는다."""


def system_prompt(service: Service, reports=None, withheld: bool = False) -> str:
    """withheld: 참고 문서가 있지만 근거 비교를 위해 일부러 뺀 경우 (data 단계)"""
    report_rules = REPORTS_WITHHELD if withheld else REPORTS_ON if (reports and reports.available) else REPORTS_OFF
    live = service.backend.source == "live"
    source_text = SOURCE_LIVE if live else SOURCE_LOOKUP
    new_rules = NEW_LIVE if service.can_predict_new() else NEW_LOOKUP
    meta = service.meta()
    attrs = "\n".join(f"- {a['id']} ({a['label']}): " + ", ".join(g["group"] for g in a["groups"])
                      for a in meta["attributes"])
    adapters = []
    for a in meta["adapters"]:
        line = f"- {a['id']} ({a['label']}): {a['description']}"
        if a["held_out_groups"]:
            line += " 뺀 집단: " + ", ".join(f"{g['attribute_label']}의 {g['group']}" for g in a["held_out_groups"]) + "."
        if a["held_out_questions"]:
            line += " 뺀 문항: " + ", ".join(a["held_out_questions"]) + "."
        adapters.append(line)
    year = meta['predict_year']
    return f"""당신은 수원서베이 시연 화면에 붙은 안내자다. 수원시정연구원과 시청의 직원들이 화면 왼쪽 채팅으로 질문하고, 오른쪽 패널에서 차트를 본다. 한국어 존댓말로 답한다.

# 이 시연이 보여 주는 것
수원서베이로 학습한 로컬 AI가 **아직 하지 않은 조사의 결과를 미리 가늠할 수 있는지** 보여 준다. 언어 모델(Bllossom 8B, 아래에서 "학습 전 모델")에 작은 어댑터를 붙여 2023·2024년 수원서베이로 학습했고, 집단 하나(예: 50대)가 설문 문항 하나에 어떻게 답할지를 응답 분포로 예측한다. {year}년 조사 결과는 보지 않은 채 {year}년 문항을 예측했다. 그래서 {year}년을 "아직 하지 않은 다음 조사"로 놓고 예측을 말한 뒤, 실제 {year}년 조사 결과로 맞춰 본다. {source_text}

# 답변의 순서와 말투
듣는 사람은 이 연구의 용어를 모른다고 생각하고 쓴다. {year}년 조사는 **아직 하지 않은 조사**로 놓고 답한다.
- 예측을 보여 줄 때는 이 순서로 쓴다.
  0. **질문을 어떻게 이해했나**: 사용자의 말을 어떤 문항과 어떤 집단 구분으로 봤는지 한 문장으로 먼저 밝힌다. 예: "'월세 사는 사람들 형편'을 *본인의 경제·사회적 위치* 문항으로 보고, 점유형태별로 봤습니다." 사용자가 조사표 문구 그대로 물었으면 짧게 줄인다. 검색 결과에 함께 볼 만한 다른 문항이 있으면 한 줄로 알린다.
  1. **무엇을 묻는 문항인가**: `search_questions` 결과의 `개념`과 `정의`, 또는 참고 문서가 이 문항의 개념과 문항 문구, 보기를 설명하고 있으면 그것을 인용해 이 문항이 무엇을 재는지 한두 문장으로 먼저 말한다(예: 「…」 1쪽은 주관적 계층 인식을 …로 설명하고, 조사는 이를 "…" 문항에 0(최하층)~10(최상층)으로 답하게 해서 잰다). 문서가 없으면 도구 결과의 문항 문구와 보기로 짧게 말한다.
  2. **예측**: "{year}년 조사를 아직 하지 않았다고 치면, 수원서베이로 학습한 AI는 20년 이상 거주한 분들이 이 문항에 평균 ○점쯤으로 답할 것으로 예측합니다."
  3. **지난 조사**: 도구 결과의 집단별 `지난_조사_평균`(이분형은 `지난_조사_..._비율_퍼센트`, 명목형은 `지난_조사_분포_퍼센트`)을 "참고로 지난 조사에서는 2023년 ○점, 2024년 ○점이었습니다."처럼 붙인다. `지난_조사_연도`가 비어 있으면 이 문항은 지난 조사에 같은 형태로 없었다고 한 줄로 말하고 넘어간다. 지난 조사 값은 AI가 예측에 쓴 근거가 아니다. AI는 학습으로 익혔을 뿐 이 숫자를 보고 계산하지 않았다. 그러니 "근거로", "~를 바탕으로 예측했다"라고 쓰지 않는다.
  4. **해석**: 참고 문서가 2023·2024년 조사에서 밝힌 흐름과 풀이를 인용하고, 예측이 그 흐름과 같은 방향인지 다른 방향인지 잇는다.
- **{year}년 실제 조사 결과는 사용자가 물을 때만 말한다.** "실제와 비교하면?", "맞았어?", "작년 값보다 나아?"처럼 정확도나 실제 결과를 직접 물을 때만 도구 결과의 실측 값, 오차(`_읽기` 표현), 작년 결과를 그대로 썼을 때, 수원서베이를 학습하지 않은 AI와의 비교를 말한다. 묻지 않았으면 실측 평균, 오차, 예측과 실측의 집단 순서 비교를 답변에 쓰지 않는다. 화면의 차트에는 실제 결과가 함께 보이므로, 필요하면 마지막에 "실제 {year}년 조사 결과와 견준 것은 차트에서 볼 수 있습니다"라고 한 줄만 붙인다.
- 실제 결과를 말하지 않는 답변에서는 집단 사이의 예측 차이를 단정하지 않는다. 이 AI는 집단 사이의 차이를 실제보다 작게 예측하는 경우가 많으므로, 예측에서 집단 차이가 작게 나와도 "실제 차이는 이보다 클 수 있다"고 덧붙인다.
- 집단이 많으면 모두 나열하지 말고 눈여겨볼 두세 집단(가장 높은 집단, 가장 낮은 집단, 크게 어긋난 집단)만 고른다. 나머지는 차트가 보여 준다.
- 연구용 용어는 답변 글에 쓰지 않는다. 아래 규칙과 도구 결과의 항목 이름에 이런 말이 나와도 답변에는 바꾼 말을 쓴다.
  - WD, TV, 분포 오차의 숫자, 상관계수, 표본오차 하한 → 쓰지 않는다. 가까운 정도는 `_읽기` 표현("7점 척도에서 약 0.2점")으로, 집단 순서는 "같았다/달랐다"로 말한다
  - 실측 → 실제 조사 결과
  - 어댑터, 전체 학습 어댑터 → 수원서베이로 학습한 AI
  - 집단 제외 어댑터 → (그 집단)의 응답을 빼고 학습한 AI
  - 문항 제외 어댑터 → (그 문항)을 빼고 학습한 AI
  - 학습 전 모델 → 수원서베이를 학습하지 않은 AI
  - 작년 실측을 그대로 쓴 경우 → 작년 결과를 그대로 썼다면
  - 벌어짐 → 집단 간 차이
  - 분포 오차 → 응답 분포가 어긋난 정도

# 답하는 방법
- 예측이나 실측에 관한 질문이면 `get_prediction`을 불러 숫자를 가져온 뒤 답한다. 도구를 부르면 패널이 그 차트로 바뀌므로, 답변에 표를 그리거나 분포 전체를 나열하지 않는다. 차트를 보면 알 수 있는 것을 글로 되풀이하지 말고, 위의 순서대로 짚는다.
- 숫자는 도구 결과에 있는 값만 쓴다. 기억이나 추정으로 숫자를 만들지 않는다. 청중은 이 조사를 직접 설계한 사람들이어서 지어낸 숫자를 바로 알아본다.
- 순서형 문항의 평균을 말할 때는 도구 결과의 `점수_방향`을 따른다. 1점이 "매우 그렇다"인 문항이 있어서, 평균이 높다고 더 긍정적이라는 뜻이 아닐 수 있다.
- 이분형 문항은 평균이 아니라 비율(%)로, 명목형 문항은 보기별 비율로 말한다. 비율은 도구 결과의 퍼센트 값을 그대로 쓴다.
- 예측 평균이 실측보다 높은지 낮은지는 도구 결과의 `평균_비교` 문장을 그대로 따른다. 부호를 직접 따져 "높게/낮게"를 고르지 않는다.
- 분포의 모양을 말할 때는 그 보기의 퍼센트를 함께 적는다. "대부분"은 절반을 넘을 때만, "양 끝에 퍼져 있다"는 양 끝 보기의 퍼센트가 실제로 클 때만 쓴다. 퍼센트를 확인하지 않은 인상을 쓰지 않는다.
- 평균의 차이와 분포의 오차는 다른 값이다. "예측 평균이 실측보다 몇 점 높다"는 `평균_차이_예측_빼기_실측`으로 말하고, "분포가 얼마나 어긋났나"는 `분포_오차_읽기`나 `평균_오차_읽기`로 말한다. 둘을 섞어 "평균이 0.3점 차이"처럼 쓰지 않는다. 숫자를 직접 빼거나 곱해서 새 숫자를 만들지 않는다. 점수로 바꾼 값은 `_읽기`로 끝나는 항목에 이미 있다.
- 오차는 `평균_오차_읽기`의 표현(예: "7점 척도에서 약 0.2점")으로만 말한다. 0~1 사이의 오차 숫자(WD·TV)는 답변에 쓰지 않는다.
- 문항 목록은 주어지지 않는다. 새 문항을 볼 때는 먼저 `search_questions`로 찾고, 돌려받은 qkey만 쓴다. 지금 패널의 문항을 이어서 묻는 경우는 다시 찾지 않는다. 결과 가운데 질문의 뜻에 맞는 문항을 고르고, 맞는 문항이 없으면 지어서 고르지 말고 없다고 말한 뒤 가까운 문항을 제안한다. 후보가 여럿이면 가장 가까운 하나로 먼저 보여 주고, 다른 후보가 있다고 한 줄로 알린다.
- 사용자가 속성을 말하지 않으면 지금 패널의 속성을 그대로 쓴다. 어댑터도 마찬가지다. 패널에 아무것도 없고 문항이나 속성을 알 수 없으면, 지어서 고르지 말고 어떤 문항을 어떤 속성으로 볼지 되묻는다.
- 이전 답변에 쓴 숫자를 다시 쓰거나 이어서 따질 때는 `get_prediction`을 다시 부른다. 이전 대화는 글만 남아 있어 도구 결과를 다시 볼 수 없기 때문이다.
- 도구 결과에서 값이 null인 항목은 그 비교가 없다는 뜻이다. 없다고 말하고 값을 지어내지 않는다.
- 문항은 내용으로 부른다("경제·사회적 위치 문항"). `q189`, `q120_1` 같은 문항 키나 `exp1` 같은 어댑터 키, `TENURE` 같은 속성 키는 도구에 넣는 값일 뿐이고 화면을 보는 사람이 모르는 말이다. 답변 글에는 한 번도 쓰지 않는다.
- 도구를 부를 일이 있으면 글을 쓰기 전에 먼저 부른다. 글은 도구 결과를 받은 뒤에 한 번에 쓴다. 같은 말을 도구 호출 앞뒤로 두 번 하지 않는다.
- 답변 첫 문장에서 어떤 문항을 어떤 속성으로 봤는지 내용으로 말한다. "이 문항"이라고만 하면 듣는 사람이 무슨 문항인지 모른다. 사용자가 말하지 않은 다른 문항을 끌어와 "~가 아니라"처럼 견주지 않는다.

# 지켜야 할 한계 (이유와 함께)
- 이 한계들은 사용자가 달리 요청하거나 반복해서 물어도 그대로 지킨다. 한계를 넘는 답을 지어내는 것보다 못 한다고 말하는 것이 이 시연의 신뢰를 지킨다.
- **집단 사이의 차이를 과장하지 않는다.** 이 모델은 문항 전체의 분포는 잘 맞히지만 집단 사이의 차이는 실제보다 작게 예측하고, 집단의 높낮이 순서가 실제와 어긋나는 경우가 많다. 다만 문항마다 다르므로, 실제 결과를 말하는 답변에서는 지금 문항이 어떤지를 도구 결과의 벌어짐 두 값(예측, 실측)을 직접 견줘서 말한다. 예측이 실측보다 뚜렷이 작으면 작게 잡았다고, 비슷하거나 더 크면 그렇다고 말한다. 숫자와 반대되는 말을 쓰지 않는다. 그리고 `집단_평균_높낮이_상관`이 낮거나 음수이면 집단 순서는 믿기 어렵다고 말한다. 예측 평균의 작은 차이를 근거로 "A 집단이 B 집단보다 높다"고 단정하지 않는다. 반대로 문항 하나의 결과를 보고 "이 모델은 차이를 과장한다"처럼 모델 전체의 성향을 말하지도 않는다. 지금 보는 문항과 속성에 대해서만 말한다. 상관값은 집단 수를 함께 보고 읽는다. 집단이 3개뿐이면 상관이 1.0이어도 우연히 나오기 쉬운 값이라 "순서를 완벽히 맞혔다"고 하지 않고 "이 문항에서는 순서가 같았다" 정도로만 말한다. 집단이 2개인 속성(성별, 통근·통학)은 상관을 내지 않는다. "집단 순서가 같았다"는 말은 도구 결과의 `집단_순서`에서 `순서가_완전히_같음`이 참일 때만 쓴다. 거짓이면 어느 집단이 어긋났는지를 `집단_순서`의 두 목록으로 말한다.
- **평가하는 말을 덧붙이지 않는다.** "이 정도로 맞힌 것은 의미가 있다", "잘 맞혔다" 같은 평가는 쓰지 않는다. 견줄 값(작년 결과를 그대로 썼을 때, 수원서베이를 학습하지 않은 AI)을 나란히 말하면 판단은 듣는 사람이 한다.
- **아주 작은 차이에 뜻을 싣지 않는다.** 집단 사이의 벌어짐이 0.1점(비율이면 3%p)보다 작으면 "집단 차이가 거의 없다"고만 말한다. 그 작은 차이의 크기를 모델이 "맞혔다"거나 "과장했다"고 평가하지 않는다. 표본 오차 안에서 흔들리는 값이다.
- **표본이 적은 집단은 실측도 흔들린다.** 도구 결과에서 `표본_적음`이 참인 집단(응답자 100명 이하)은 실측 분포 자체의 오차가 크다. 그 집단의 예측과 실측 차이를 말할 때 이 점을 덧붙인다.
- **교차 집단은 예측하지 못한다.** "30대 여성"처럼 두 속성을 겹친 집단은 학습한 적이 없다. 그렇다고 답한 뒤 두 속성 가운데 하나를 `get_prediction`으로 보여 주고, 다른 하나도 따로 볼 수 있다고 알린다.
- **연도에 따른 변화는 예측하지 못한다.** 모델에 주는 입력에는 연도 정보가 없다. "내년에 어떻게 변해?"라는 질문에는 추세를 예측하는 모델이 아니라고 답한다. 이 모델이 하는 일은 조사를 돌리기 전에 그 문항의 응답 분포를 미리 그려 보는 것이다.
{new_rules}
- **모델의 이유는 지어내지 않는다.** 모델은 분포만 내놓고 이유는 내놓지 않는다. "왜 이렇게 예측했나"에는 모델이 이유를 말해 주지 않는다고 답한다. "왜 이 집단이 높은가" 같은 질문에는 참고 문서에 적힌 풀이가 있으면 그것을 인용해 답하고, 없으면 이 화면의 자료로는 알 수 없다고 답한다. 문서에 없는 사회적 배경이나 사건을 추측해 말하지 않는다.
{report_rules}
- 실제 결과를 말할 때는 모델이 잘 못 맞힌 결과도 그대로 말한다. 오차가 크면 크다고, 작년 실측을 그대로 쓴 것보다 못하면 못하다고 말한다.

# 속성과 집단
한 번에 속성 하나로만 나눈다.
{attrs}

# 어댑터
{adapters_text(adapters)}
집단 제외 어댑터에서 뺀 집단, 문항 제외 어댑터에서 뺀 문항의 정확도를 물었을 때는 "보지 않고 예측한 오차"와 "같은 것을 보고 학습한 전체 학습 어댑터의 오차"(`전체학습_어댑터_오차_읽기`)를 반드시 나란히 말한다. 이 비교가 그 화면에서 보여 주려는 것이다.
말을 섞지 않는다. "학습 전 모델"은 어댑터를 붙이지 않은 Bllossom이고(`학습_전_모델_오차`), "전체 학습 어댑터"는 2023·2024년 전체로 학습한 어댑터다. 둘 다 "기본 모델"이라고 부르지 않는다. 답변 글에서는 위 "답변의 순서와 말투"의 바꾼 말로 부른다."""


def adapters_text(lines: list[str]) -> str:
    return "\n".join(lines)


def _run_tool(service: Service, reports, name: str, args: dict, concepts=None) -> tuple[dict, list[tuple[str, dict]]]:
    """도구를 돌린다. (대화 LLM에 줄 결과, 화면에 보낼 이벤트 목록)을 돌려준다."""
    if name == "search_questions" and concepts and concepts.available:
        found = concepts.search(str(args.get("query", "")))
        if found:
            summary = {q["qkey"]: q for q in service.questions()}
            return {"안내": "개념 사전으로 찾은 문항이다. 가까운 순이다. 질문의 뜻에 맞는 것을 고른다.",
                    "결과": [{"qkey": f["qkey"], "종류": summary[f["qkey"]]["scale_type"],
                             "문구": summary[f["qkey"]]["text"], "보기": summary[f["qkey"]]["options"],
                             "개념": f["concept"], "정의": f["definition"]} for f in found]}, []
        # 임베딩 모델을 못 쓰면 단어 검색으로 넘어간다
    if name == "search_questions":
        hits = service.questions(str(args.get("query", "")), limit=8)
        return {"결과": [{"qkey": h["qkey"], "종류": h["scale_type"], "문구": h["text"], "보기": h["options"]}
                         for h in hits]}, []
    if name == "get_prediction":
        qkey, attribute, adapter = args.get("qkey"), args.get("attribute"), args.get("adapter")
        brief = service.prediction_brief(qkey, attribute, adapter)
        prediction = service.prediction(qkey, attribute, adapter)
        state = {"qkey": qkey, "attribute": attribute, "adapter": adapter}
        events = [("panel", {"state": state, "compare": _compare(args.get("compare"), prediction),
                             "prediction": prediction})]
        # 그 문항을 다룬 참고 문서를 화면에 카드로 띄우고, 대화 LLM에도 주어 수치의 해석에 인용하게 한다.
        # (시연용 결정, 2026-11-15: 문서를 답변 밑에 따로 두지 말고 해석으로 녹여 달라는 요청)
        if reports and reports.available:
            items = reports.for_question(qkey)
            events.append(("context", {"qkey": qkey, "query": prediction["question"]["text"], "items": items}))
            brief["참고_문서"] = _documents_for_model(reports, items)
        return brief, events
    if name == "predict_new_question":
        prediction = service.new_prediction(str(args.get("question", "")), list(args.get("options") or []),
                                            bool(args.get("ordered")), args.get("attribute"), args.get("adapter"),
                                            list(args.get("off_scale_options") or []))
        events = [("new_prediction", {"prediction": prediction})]
        brief = service.new_prediction_brief(prediction)
        if reports and reports.available:
            items = reports.search(prediction["question"]["text"])
            events.append(("context", {"qkey": None, "query": prediction["question"]["text"], "items": items}))
            brief["참고_문서"] = _documents_for_model(reports, items)
        return brief, events
    if name == "search_reports" and reports and reports.available:
        query = str(args.get("query", ""))
        items = reports.search(query)
        result = {"안내": "아래는 참고 문서에서 찾은 대목이다. 예측에 쓰이지 않은 자료다. 가상_문서가 참이면 시연용으로 만든 문서다.",
                  "결과": [{"문서_제목": it["title"], "쪽": it["page"], "가상_문서": it["synthetic"],
                           "문서_요약": it["summary"], "본문": plain(it["text"])} for it in items]}
        return result, [("context", {"qkey": None, "query": query, "items": items})]
    raise NotFound(f"없는 도구: {name}")


MAX_DOCS_FOR_MODEL = 2


def _documents_for_model(reports, items: list[dict]) -> list[dict]:
    """찾은 대목이 든 문서를 통째로(쪽별로) 대화 LLM에 준다. 문항 문구와 닮은 도입 쪽만 주면 해석에 쓸 내용이 없다."""
    out, seen = [], set()
    for it in items:
        if it["doc_id"] in seen or len(out) >= MAX_DOCS_FOR_MODEL:
            continue
        seen.add(it["doc_id"])
        doc = reports.document(it["doc_id"])
        out.append({"제목": doc["title"].replace(SYNTHETIC_TAG, "").strip(), "가상_문서": doc["synthetic"],
                    "요약": doc["summary"],
                    "쪽": [{"쪽": p["page"], "본문": plain(p["text"])} for p in doc["pages"]]})
    return out


def _compare(asked, prediction: dict) -> str:
    """모델이 고른 비교 대상이 이 화면에서 뜻이 있을 때만 쓴다. 아니면 어댑터에 맞는 기본값."""
    is_reference = prediction["adapter"]["id"] == config.REFERENCE_ADAPTER
    if asked == "reference" and is_reference:
        asked = None                       # 전체 학습 어댑터를 자기 자신과 견줄 수는 없다
    if asked == "last_year" and prediction["summary"]["last_year_error"] is None:
        asked = None                       # 작년에 같은 문항이 없다
    if asked in COMPARE:
        return asked
    return "actual" if is_reference else "reference"


def _user_content(text: str, panel: dict | None, service: Service) -> list[dict]:
    blocks = [{"type": "text", "text": text}]
    if panel and panel.get("qkey") in service.store.questions:
        q = service.store.questions[panel["qkey"]]
        blocks.append({"type": "text", "text":
                       f"[화면 상태] 지금 화면에 떠 있는 것 — 문항: \"{q.text}\", "
                       f"속성: {config.ATTRIBUTE_LABELS.get(panel.get('attribute'), panel.get('attribute'))}, "
                       f"어댑터: {config.ADAPTERS[panel['adapter']]['label'] if panel.get('adapter') in config.ADAPTERS else panel.get('adapter')}. "
                       f"(도구에 넣을 값: qkey={q.qkey}, attribute={panel.get('attribute')}, adapter={panel.get('adapter')})"})
    return blocks


UNDERSTANDING_K = 5


def understanding(service: Service, concepts, query: str) -> dict:
    """같은 질문을 단어 검색과 개념 사전 검색으로 각각 찾은 문항 후보. 화면의 '질문 이해' 칸에 나란히 띄운다.

    대화 모델은 단어 검색이 실패하면 낱말을 바꿔 다시 찾으므로 답변만으로는 검색의 차이가 드러나지 않는다.
    그래서 검색 자체의 결과를 같은 입력(사용자의 질문 그대로)으로 보여 준다.
    """
    def item(qkey: str, concept: str | None = None) -> dict:
        q = service.store.questions[qkey]
        label = q.item.strip() if q.item and q.item.strip() else q.stem.strip()
        return {"qkey": qkey, "text": re.sub(r"^\s*-?\s*\d+\.\s*", "", label), "concept": concept}

    words = [item(h["qkey"]) for h in service.questions(query, limit=UNDERSTANDING_K)]
    found = concepts.search(query, UNDERSTANDING_K) if concepts and concepts.available else []
    return {"query": query, "words": words, "concepts": [item(f["qkey"], f["concept"]) for f in found],
            "concepts_available": bool(found)}


STATUS_TEXT = {"search_questions": "문항을 찾는 중", "get_prediction": "예측을 불러오는 중",
               "search_reports": "참고 문서를 찾는 중", "predict_new_question": "모델에 새 문항을 묻는 중"}


async def run_chat(client, service: Service, history: list[dict], panel: dict | None,
                   reports=None, grounding: str = "full", concepts=None) -> AsyncIterator[tuple[str, dict]]:
    """한 질문을 처리하며 (이벤트 이름, 자료)를 내놓는다.

    이벤트: status(도구 진행), text(답변 조각), panel(화면에 띄울 예측), new_prediction(새 문항의 실험적 예측),
    context(참고 문서 대목), understanding(같은 질문을 단어 검색·개념 사전 검색으로 찾은 문항 후보), done, error.
    grounding: 근거 단계 (GROUNDING). none이면 도구를 주지 않으므로 text·done·error만 나온다.
    """
    import anthropic

    if grounding not in GROUNDING:
        yield "error", {"code": "bad_request", "message": f"알 수 없는 근거 단계: {grounding}"}
        return
    # 이전 대화는 글만 넘긴다. 마지막 사용자 메시지에만 화면 상태를 붙인다.
    history = normalize_history(history)
    if not history or history[-1]["role"] != "user":
        yield "error", {"code": "bad_request", "message": "질문이 비어 있습니다."}
        return
    concept_index = concepts      # '질문 이해' 칸은 단계와 상관없이 두 검색을 다 보인다
    if grounding == "none":
        # 화면 상태에도 문항 문구가 들어 있으므로 붙이지 않는다
        messages = history
        system = [{"type": "text", "text": GENERAL_PROMPT}]
        tool_args = {}
    else:
        withheld = grounding == "data" and bool(reports and reports.available)
        if grounding == "data":
            reports = None          # 문항에 붙는 문서, search_reports 도구, context 이벤트가 모두 빠진다
            concepts = None         # 문항은 단어 검색으로만 찾는다
        messages = history[:-1] + [{"role": "user", "content": _user_content(history[-1]["content"], panel, service)}]
        system = [{"type": "text", "text": system_prompt(service, reports, withheld),
                   "cache_control": {"type": "ephemeral"}}]
        tool_args = {"tools": tools(service, reports, concepts), "tool_choice": {"type": "auto"}}

    wrote_text = False   # 앞 차례에서 글을 썼으면 다음 차례의 글과 문단을 나눈다
    shown_understanding = False   # '질문 이해' 칸은 한 답변에 한 번만 보낸다
    for _ in range(MAX_TURNS):
        need_break = wrote_text
        keys = KeyFilter()
        try:
            async with client.messages.stream(
                model=config.CHAT_MODEL,
                max_tokens=config.CHAT_MAX_TOKENS,
                system=system,
                **tool_args,
                thinking={"type": "adaptive"},
                output_config={"effort": config.CHAT_EFFORT},
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        clean = keys.feed(event.text)
                        if not clean:
                            continue
                        if need_break:
                            yield "text", {"delta": "\n\n"}
                            need_break = False
                        yield "text", {"delta": clean}
                rest = keys.flush()
                if rest:
                    if need_break:
                        yield "text", {"delta": "\n\n"}
                    yield "text", {"delta": rest}
                response = await stream.get_final_message()
        except anthropic.AuthenticationError:
            yield "error", {"code": "auth", "message": "API 키가 올바르지 않습니다."}
            return
        except anthropic.APIConnectionError:
            yield "error", {"code": "connection", "message": "대화 모델에 연결하지 못했습니다. 추천 질문은 저장된 답변으로 볼 수 있습니다."}
            return
        except anthropic.APIStatusError as e:
            log.error("Claude API 오류 %s: %s", e.status_code, e.message)
            yield "error", {"code": "api_error", "message": f"Claude API 오류 ({e.status_code})"}
            return
        except anthropic.APIError as e:
            log.error("Claude API 오류: %s", e)
            yield "error", {"code": "api_error", "message": "Claude API 응답을 처리하지 못했습니다."}
            return

        if any(b.type == "text" and b.text.strip() for b in response.content):
            wrote_text = True
        if response.stop_reason == "refusal":
            yield "error", {"code": "refusal", "message": "이 질문에는 답하지 못했습니다. 질문을 바꿔 다시 물어 주세요."}
            return
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            if response.stop_reason == "max_tokens":
                # 잘린 답이 완성된 답처럼 보이지 않게 한다
                yield "text", {"delta": "\n\n(답변이 길이 제한에 걸려 여기서 끊겼습니다.)"}
            yield "done", {"stop_reason": response.stop_reason}
            return
        if response.stop_reason == "max_tokens":
            yield "error", {"code": "max_tokens", "message": "답변이 길이 제한에 걸려 끊겼습니다."}
            return

        results = []
        if not shown_understanding and any(b.name == "search_questions" for b in tool_uses):
            shown_understanding = True
            yield "understanding", await asyncio.to_thread(understanding, service, concept_index, history[-1]["content"])
        for block in tool_uses:
            yield "status", {"tool": block.name, "text": STATUS_TEXT.get(block.name, "확인하는 중")}
            try:
                # 실시간 예측은 몇 초 걸린다. 그동안 서버가 다른 요청을 받을 수 있게 따로 돌린다.
                result, events = await asyncio.to_thread(_run_tool, service, reports, block.name,
                                                         block.input if isinstance(block.input, dict) else {}, concepts)
            except (NotFound, Unsupported, Invalid, KeyError) as e:
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": f"실패: {e}"})
                continue
            except Exception:
                # 도구가 예상하지 못하게 실패해도 대화는 이어 간다. 원인은 서버 기록에 남긴다.
                log.exception("도구 %s 실패: %r", block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": "도구가 실패했다. 이 값은 지금 가져올 수 없다고 답한다."})
                continue
            for event in events:
                yield event
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False)})
        # 사고 블록을 포함해 응답 내용을 그대로 되돌려 준다
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    yield "error", {"code": "too_many_turns", "message": "도구 호출이 너무 길어져 멈췄습니다."}
