"""
RAG Pipeline 예제

이 스크립트는 저장된 FAISS Vector Store에서 관련 문서를 검색하고,
검색된 문서를 Prompt에 넣어 LLM이 최종 답변을 생성하는 RAG 파이프라인을 구성하는 과정을 보여줍니다.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

if __package__:
    from .rag_core import build_retriever, format_docs
else:
    from rag_core import build_retriever, format_docs

# 1. 검색된 문서를 근거로 답하도록 Prompt 구성
prompt = ChatPromptTemplate.from_template("""
당신은 제공된 문서를 근거로 답하는 질문-답변 도우미입니다.

규칙:
1. 아래 [문서]에 있는 내용만 근거로 답하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 확인할 수 없는 경우 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.
4. 답변은 간결하고 명확하게 작성하세요.

[문서]
{context}

[질문]
{question}

[답변]
""")

def image_references(docs):
    """검색된 그림만 중복 없이 답변에 첨부합니다."""
    references = []
    seen = set()
    for doc in docs:
        image_path = doc.metadata.get("image_path")
        if image_path and image_path not in seen and Path(image_path).is_file():
            seen.add(image_path)
            path = Path(image_path).resolve().as_uri()
            label = doc.metadata.get("image_label", "실험 그림")
            page = doc.metadata.get("page", 0) + 1
            references.append(f"![{label}, {page}쪽]({path})")
    return references

# 5. 질문 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF와 카탈로그 이미지로 질문에 답합니다.")
    parser.add_argument("question", nargs="?", default="콜로이드 합성 실험의 두 번째 단계는 무엇인가요?")
    question = parser.parse_args().question
    load_dotenv()
    retriever = build_retriever(k=8)
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    documents = retriever.invoke(question)
    answer = chain.invoke({"context": format_docs(documents), "question": question})
    print("[질문]\n" + question + "\n\n[답변]\n" + answer)
    images = image_references(documents)
    if images and "제공된 문서에서 확인할 수 없습니다." not in answer:
        print("\n[관련 이미지]\n" + "\n".join(images))
