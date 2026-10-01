"""
RAG 평가 예제

이 스크립트는 RAG 파이프라인의 Retrieval 및 Generation 성능을 평가하는 과정을 보여줍니다.
주어진 테스트 케이스에 대해 Hit Rate, MRR을 계산하고, LLM이 생성한 답변의 Groundedness를 평가합니다.
"""
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_core import build_retriever, format_docs

load_dotenv()

retriever = build_retriever(
    k=3,
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

test_cases = [
    {
        "question": "평일 운영시간은 언제인가요?",
        "expected_keyword": "오전 9시부터 오후 9시",
    },
    {
        "question": "한 사람이 빌릴 수 있는 책은 최대 몇 권인가요?",
        "expected_keyword": "최대 5권",
    },
    {
        "question": "노트북은 어디에서 사용할 수 있나요?",
        "expected_keyword": "2층 디지털자료실",
    },
]


def hit_at_k(question: str, expected_keyword: str) -> int:
    docs = retriever.invoke(question)
    joined = "\n".join(doc.page_content for doc in docs)
    return 1 if expected_keyword in joined else 0


def reciprocal_rank(question: str, expected_keyword: str) -> float:
    docs = retriever.invoke(question)
    for rank, doc in enumerate(docs, start=1):
        if expected_keyword in doc.page_content:
            return 1.0 / rank
    return 0.0


answer_prompt = ChatPromptTemplate.from_template("""
아래 문서만 근거로 질문에 답하세요.
문서에서 확인할 수 없으면 확인할 수 없다고 답하세요.

[문서]
{context}

[질문]
{question}
""")


def generate_answer(question: str, docs) -> str:
    context = format_docs(docs)
    messages = answer_prompt.invoke({
        "context": context,
        "question": question,
    })
    return llm.invoke(messages).content


def judge_groundedness(question: str, context: str, answer: str) -> int:
    prompt = f"""
당신은 RAG 답변 평가자입니다.
아래 답변이 제공된 문서에 얼마나 충실하게 근거하는지 1~5점으로 평가하세요.

5점: 답변의 핵심 내용이 모두 문서에 명확히 근거함
4점: 거의 모두 근거하며 사소한 표현 차이만 있음
3점: 일부 근거하지만 해석이나 추가 내용이 섞임
2점: 근거가 부족하고 추측이 많음
1점: 문서와 무관하거나 모순됨

숫자 하나만 출력하세요.

[질문]
{question}

[문서]
{context}

[답변]
{answer}
"""
    result = llm.invoke(prompt).content.strip()
    match = re.search(r"[1-5]", result)
    return int(match.group()) if match else 1


hits = []
rr_scores = []

print("=== Retrieval 평가 ===")
for case in test_cases:
    hit = hit_at_k(case["question"], case["expected_keyword"])
    rr = reciprocal_rank(case["question"], case["expected_keyword"])
    hits.append(hit)
    rr_scores.append(rr)
    print(case["question"], "→", "HIT" if hit else "MISS", "RR =", round(rr, 3))

print("Hit Rate:", round(sum(hits) / len(hits), 3))
print("MRR:", round(sum(rr_scores) / len(rr_scores), 3))

print("\n=== Generation 평가 ===")
for case in test_cases:
    question = case["question"]
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = generate_answer(question, docs)
    score = judge_groundedness(question, context, answer)

    print("\n============================")
    print("질문:", question)
    print("답변:", answer)
    print("Groundedness:", score, "/ 5")
