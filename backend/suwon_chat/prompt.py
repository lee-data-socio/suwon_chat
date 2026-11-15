"""모델에 주는 프롬프트. 학습과 평가에 쓴 형식(SubPOP의 QA 유형)과 글자 하나까지 같아야 한다.

형식 (정리본 `opnqa_QA_test.csv`의 `input_prompt`):

    Question: <집단을 묻는 질문>
    A. <집단 1>
    B. <집단 2>

    Answer: B. <집단 2>

    Answer the following question keeping in mind your previous answers.
    Question: <문항>
    A. <보기 1>
    ...
    H. 무응답
    Answer as a choice between A.,B.,...,H.

    Answer:

모델은 마지막 `Answer:` 다음에 올 보기 글자의 확률을 내놓는다.
"""
from __future__ import annotations

REFUSAL = "무응답"
BRIDGE = "Answer the following question keeping in mind your previous answers."
MAX_OPTIONS = 26


def letter(i: int) -> str:
    return chr(ord("A") + i)


def _question(text: str, options: list[str], forcing: bool) -> str:
    lines = [f"Question: {text.strip()}"] + [f"{letter(i)}. {o.strip()}" for i, o in enumerate(options)]
    if forcing:
        lines.append("Answer as a choice between " + ",".join(f"{letter(i)}." for i in range(len(options))))
    return "\n".join(lines).strip() + "\n\nAnswer:"


def build_prompt(group_question: str, groups: list[str], group: str, question: str, options: list[str]) -> str:
    """한 집단이 한 문항에 답하는 프롬프트. `options`는 끝의 무응답까지 포함한 보기다."""
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"보기는 2~{MAX_OPTIONS}개여야 한다: {len(options)}개")
    i = groups.index(group)
    return (f"{_question(group_question, groups, forcing=False)} {letter(i)}. {group.strip()}\n\n"
            f"{BRIDGE}\n{_question(question, options, forcing=True)}")


def with_refusal(options: list[str]) -> list[str]:
    """새 문항의 보기 끝에 무응답을 붙인다. 학습한 문항은 모두 마지막 보기가 무응답이었다."""
    options = [o.strip() for o in options if o and o.strip()]
    return options if options and options[-1] == REFUSAL else options + [REFUSAL]
