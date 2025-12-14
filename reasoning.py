import json
from typing import List

import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError

from google import genai
from google.genai.errors import APIError


GEMINI_API_KEY = "..."
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
FILE_PATH = "/"

client = genai.Client(api_key=GEMINI_API_KEY)


#data load
file_path = FILE_PATH + "math_data.csv"
df = pd.read_csv(file_path)


#define reasoning output josn schema and prompt
class ReasoningOutput(BaseModel):
    planning: List[str]
    reasoning_steps: List[str]
    answer: str

def get_reasoning_prompt(q, a):
    output_json_schema = """
    {
    "planning": [
        "문제에서 구해야 할 값을 정의한다",
        "필요한 수학적 개념이나 공식을 식별한다",
        "해당 개념을 적용해 계산 절차를 구성한다"
    ],
    "reasoning_steps": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
    ],
    "answer": "ANSWER"
    }
    """
    prompt = f"""
    당신은 수학 문제의 풀이 과정을 생성하는 전문가입니다.

    아래에는 수학 문제와 해당 문제의 정답이 주어져 있습니다.
    정답은 최종 확인 용도로만 사용하며,
    풀이 과정에서 정답을 근거로 사용하는 역추론은 절대 하지 마세요.

    [사고 및 내용 규칙]
    - 풀이 계획(planning)은 문제만 보고 작성해야 하며, 정답을 사용하지 않습니다.
    - 풀이 과정(reasoning)은 반드시 계획을 따라 단계적으로 전개되어야 합니다.
    - 각 단계는 이전 단계의 결과를 명확히 활용해야 합니다.
    - 불필요한 직관적 설명이나 결과 중심 설명은 피하세요.

    [문제]
    {q}

    [정답]
    {a}

    출력은 반드시 아래 JSON 형식을 정확히 따르세요.
    추가 설명이나 자연어 문장은 출력하지 마세요.

    [출력 및 구조 규칙]
    - reasoning_steps는 "단계(step) 단위"의 리스트입니다.
    - 리스트의 각 요소는 하나의 Step 전체 설명을 담은 문자열이어야 합니다.
    - 하나의 Step 안에는 여러 문장, 줄바꿈, 수식($$ $$)이 모두 포함될 수 있습니다.
    - Step 내부 문장을 나누어 여러 리스트 요소로 분리하지 마세요.
    - "Step 1", "Step 2"와 같이 단계 번호를 명시하세요.

    [출력 JSON 형식]
    {output_json_schema}
    """
    return prompt

def extract_json_str(raw_text: str) -> str:
    """
    응답에서 JSON 객체 부분만 잘라내는 함수.
    ```json ... ``` 형태를 쓰더라도 웬만하면 버텨주도록 구현
    """
    text = raw_text.strip()

    # 코드블록 제거
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[len("json"):].strip()
            if p.startswith("{") and p.endswith("}"):
                return p

    # 첫 '{' ~ 마지막 '}' 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]

    return text

#generate reasoning
df_len = len(df)
results = []
for i in range(df_len):
    question = df.iloc[i]['question']
    answer = df.iloc[i]['answer']
    
    REASONING_PROMPT = get_reasoning_prompt(question, answer)

    try:
        reasoning_response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=REASONING_PROMPT,
        )

        text = reasoning_response.text
        r_json_str = extract_json_str(text)
        cleaned_json_str = r_json_str.replace('\\', '\\\\')

        data_dict = json.loads(cleaned_json_str)
        reasoning_result = ReasoningOutput.model_validate(data_dict)
        results.append(reasoning_result)
    except (APIError, json.JSONDecodeError, KeyError) as e:
        print(f"오류 발생 (index: {i}): {type(e).__name__} - {e}")
        if not results:
            print('저장할 결과가 없어 저장 건너뜀')
        try:
            result_df = pd.DataFrame(r.model_dump() for r in results)
            result_df.to_csv(FILE_PATH + "res_fail_saved.csv", index=False)
            print(f"{len(result_df)}개 결과 저장")
        except Exception as e:
            print(e)
        break
else:
    result_df = pd.DataFrame(r.model_dump() for r in results)

    df_out = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    df_out.to_csv(FILE_PATH + "reasoning_data.csv", index=False)

    print('success reasoning generation')


#define eval json schema and prompt
class EvaluationOutput(BaseModel):
    answer_match: int
    step_integrity: int
    math_validity: int
    completeness: int
    no_answer_leak: int
    formatting_ok: int
    overall_score: float
    eval_result: str
    fail_reasons: List[str]
    minimal_fix_suggestion: str

def get_eval_prompt(q, a, reas_steps, model_answer):
    eval_json_schema = """
    {
    "answer_match": 0 or 1,
    "step_integrity": 0 or 1,
    "math_validity": 0 or 1,
    "completeness": 0 or 1,
    "no_answer_leak": 0 or 1,
    "formatting_ok": 0 or 1,
    "overall_score": 0.0-1.0,
    "eval_result": "PASS" or "FAIL",
    "fail_reasons": [
        "FAIL인 경우에만: 어떤 규칙을 왜 위반했는지 짧게"
    ],
    "minimal_fix_suggestion": "FAIL인 경우에만: 고치기 위한 최소 수정 방향 1~2문장"
    }
    """

    eval_prompt = f""" 
    당신은 수학 풀이 과정(Reasoning)을 검수하는 검수자입니다.
    아래 입력(문제, 정답, 풀이 과정, 모델정답)을 읽고, 고품질 기준 충족 여부를 판정하세요.

    [중요]
    - 당신은 새로운 풀이를 "작성"하는 역할이 아닙니다. 오직 "검수"만 합니다.
    - 기준은 엄격하게 적용합니다. 애매하면 FAIL로 처리합니다.
    - 정답이 맞아도 reasoning이 부실하거나 비약이 있으면 FAIL입니다.

    [문제]
    {q}

    [정답]
    {a}

    [풀이 과정]
    {reas_steps}

    [모델 정답]
    {model_answer}

    [검수 기준: Binary (0/1)]
    1) answer_match: 최종 답이 주어진 정답과 정확히 일치한다.
    2) step_integrity: 각 Step이 독립 문장 나열이 아니라, 이전 Step 결과를 활용해 다음 Step으로 이어진다.
    3) math_validity: 핵심 수식 변형/부등식 처리/대입/계산에서 명백한 수학적 오류가 없다.
    4) completeness: 정답 도출에 필요한 핵심 단계(정의/식 세팅/주요 변형/결론)가 누락되지 않았다.
    5) no_answer_leak: “정답이 ~이므로/정답을 만족하도록” 같은 역추론(끼워맞추기) 패턴이 없다.
    6) formatting_ok: 요구된 형식(JSON, step 묶음 등)을 충족한다. (파싱 가능)

    [최종 판정 규칙]
    - 위 6개 항목이 모두 1이면 PASS, 하나라도 0이면 FAIL.
    - overall_score는 (합계/6)으로 0~1 값을 출력.

    [출력 JSON 형식]
    다른 불필요한 출력은 제외하고, 무조건 아래의 출력을 따르세요.
    {eval_json_schema}
    """
    return eval_prompt


#eval reasoning
eval_res = []
for i in range(df_len):
    question = df_out.iloc[i]['question']
    answer = df_out.iloc[i]['answer']
    reas_steps = df_out.iloc[i]['reasoning_steps']
    model_ans = df_out.iloc[i]['answer']

    eval_prompt = get_eval_prompt(question, answer, reas_steps, model_ans)

    try:
        eval_result = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=eval_prompt,
        )

        eval_text = eval_result.text
        e_json_str = extract_json_str(eval_text)
        eval_result = EvaluationOutput.model_validate_json(e_json_str)

        eval_res.append(eval_result)
    except (APIError, json.JSONDecodeError, KeyError) as e:
        print(f"오류 발생 (index: {i}): {type(e).__name__} - {e}")
        if not eval_res:
            print("저장할 결과가 없어 저장 건너뜀")
        try:
            eval_df = pd.DataFrame(e.model_dumnp() for e in eval_res)
            eval_df.to_csv(FILE_PATH + "eval_fail_saved.csv")
            print(f"{len(eval_df)}개 결과 저장")
        except Exception as e:
            print(e)
        break
else:
    eval_df = pd.DataFrame([e.model_dump() for e in eval_res])

    final_df = pd.concat([df_out.reset_index(drop=True), eval_df], axis=1)
    final_df.to_csv(FILE_PATH + "math_data_with_reasoning.csv", index=False)


print("Saved with reasoning:", final_df.shape)

