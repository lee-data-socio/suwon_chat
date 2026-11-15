"""추천 질문 (PRD 5절 시나리오). 버튼을 누르면 패널이 이 상태로 바뀌고, 같은 문장이 채팅으로 간다.

패널 상태가 여기 박혀 있으므로 대화 LLM이나 인터넷이 없어도 패널은 돈다.
"""

SCENARIOS = [
    {
        "id": "S1", "title": "기본 예측",
        "prompt": "점유형태별로 본인의 경제·사회적 위치를 어떻게 답할지 예측해 줘.",
        "panel": {"qkey": "q189_W2025", "attribute": "TENURE", "adapter": "exp1", "compare": "actual"},
    },
    {
        "id": "S2", "title": "작년 값과 비교",
        "prompt": "일자리 복지서비스가 충분한지 묻는 문항을 점유형태별로 보여 줘. 작년 실측을 그대로 쓰는 것보다 나아?",
        "panel": {"qkey": "q132_1_W2025", "attribute": "TENURE", "adapter": "exp1", "compare": "last_year"},
    },
    {
        "id": "S3", "title": "못 본 집단",
        "prompt": "50대 자료를 한 번도 보여 주지 않고 학습했다면 50대 예측은 어떻게 돼?",
        "panel": {"qkey": "q189_W2025", "attribute": "AGE", "adapter": "exp2", "compare": "reference"},
    },
    {
        "id": "S4a", "title": "못 본 문항 · 잘 되는 경우",
        "prompt": "버스 인프라 문항을 학습에서 뺐다면 점유형태별 예측이 어떻게 돼?",
        "panel": {"qkey": "q51_1_W2025", "attribute": "TENURE", "adapter": "exp3", "compare": "reference"},
    },
    {
        "id": "S4b", "title": "못 본 문항 · 안 되는 경우",
        "prompt": "'지난 한 주 동안 행복했다' 문항을 학습에서 뺐다면 점유형태별 예측이 어떻게 돼?",
        "panel": {"qkey": "q120_1_W2025", "attribute": "TENURE", "adapter": "exp3", "compare": "reference"},
    },
    {
        "id": "S5", "title": "할 수 없는 것",
        "prompt": "30대 여성은 어떻게 답할 것 같아?",
        "panel": None,
    },
]
