"""
검색 품질 평가 예제

이 스크립트는 다양한 검색 방식(Similarity, MMR, BM25, Hybrid Search)을 사용해 문서를 검색하고,
LLM을 사용해 검색 결과의 관련성을 평가하는 과정을 보여줍니다.
"""
import re

from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pages = [
    Document(page_content="RAG는 외부 문서를 검색한 뒤 검색 결과를 LLM의 Context에 추가하여 답변을 생성한다."),
    Document(page_content="RAG의 Retriever는 사용자 질문과 관련된 문서를 Vector Store에서 검색한다."),
    Document(page_content="MMR은 관련성이 높은 문서를 찾으면서도 서로 너무 비슷한 검색 결과의 중복을 줄인다."),
    Document(page_content="BM25는 단어의 등장 빈도와 문서 내 중요도를 이용하는 키워드 기반 검색 방식이다."),
    Document(page_content="Hybrid Search는 Dense Retrieval과 Sparse Retrieval을 결합하여 검색 Recall과 정확도를 보완한다."),
    Document(page_content="Reranker는 1차 검색으로 가져온 후보 문서를 질문과 다시 비교하여 순서를 재정렬한다."),
    Document(page_content="DEMO-2026-R7은 검색 실습에서 사용하는 예시 식별자이다."),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(pages, embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

question = "RAG 검색 결과의 중복을 줄이는 방법은 무엇인가요?"

print("=== Similarity Top-3 ===")
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
similarity_docs = similarity_retriever.invoke(question)
for i, doc in enumerate(similarity_docs, start=1):
    print(i, doc.page_content)

print("\n=== MMR Top-3 ===")
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5,
    },
)
mmr_docs = mmr_retriever.invoke(question)
for i, doc in enumerate(mmr_docs, start=1):
    print(i, doc.page_content)

print("\n=== BM25 Top-3 ===")
bm25 = BM25Retriever.from_documents(pages)
bm25.k = 3
keyword_question = "DEMO-2026-R7이 무엇인가요?"
for i, doc in enumerate(bm25.invoke(keyword_question), start=1):
    print(i, doc.page_content)


def unique_merge(*doc_lists):
    merged = []
    seen = set()
    for docs in doc_lists:
        for doc in docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


print("\n=== Hybrid Candidates ===")
dense = vectorstore.as_retriever(search_kwargs={"k": 3})
sparse = BM25Retriever.from_documents(pages)
sparse.k = 3
query = "DEMO-2026-R7 검색 시스템 식별자"

dense_docs = dense.invoke(query)
sparse_docs = sparse.invoke(query)
hybrid_candidates = unique_merge(dense_docs, sparse_docs)

for i, doc in enumerate(hybrid_candidates, start=1):
    print(i, doc.page_content)


def relevance_score(question: str, document: str) -> int:
    prompt = f"""
다음 질문과 문서의 관련성을 0~100 사이 정수 하나로 평가하세요.
설명하지 말고 숫자만 출력하세요.

[질문]
{question}

[문서]
{document}
"""
    response = llm.invoke(prompt).content.strip()
    match = re.search(r"\d+", response)
    if not match:
        return 0
    return min(100, max(0, int(match.group())))


def rerank(question: str, docs, top_n: int = 3):
    scored = []
    for doc in docs:
        score = relevance_score(question, doc.page_content)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


print("\n=== Reranked Top-3 ===")
for rank, (score, doc) in enumerate(
    rerank(query, hybrid_candidates, top_n=3),
    start=1,
):
    print(f"{rank}. score={score}")
    print(doc.page_content)

print("\n=== ParentDocumentRetriever ===")
parent_docs = TextLoader("data/sample.txt", encoding="utf-8").load()
parent_vectorstore = InMemoryVectorStore(embeddings)
parent_docstore = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=parent_vectorstore,
    docstore=parent_docstore,
    parent_splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    ),
    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
    ),
)
parent_retriever.add_documents(parent_docs)

for i, doc in enumerate(
    parent_retriever.invoke("대출기간과 연장 조건을 알려 주세요."),
    start=1,
):
    print(i, doc.page_content)

print("\n=== MultiQueryRetriever ===")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
multi_query = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,
)
for i, doc in enumerate(
    multi_query.invoke("검색 결과에 비슷한 문서가 반복될 때 어떻게 줄일 수 있나요?"),
    start=1,
):
    print(i, doc.page_content)

print("\n=== EnsembleRetriever ===")
dense = vectorstore.as_retriever(search_kwargs={"k": 4})
sparse = BM25Retriever.from_documents(pages)
sparse.k = 4
ensemble = EnsembleRetriever(
    retrievers=[dense, sparse],
    weights=[0.6, 0.4],
)
for i, doc in enumerate(
    ensemble.invoke("DEMO-2026-R7 검색 시스템 식별자"),
    start=1,
):
    print(i, doc.page_content)

print("\n=== LongContextReorder ===")
reorder_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
candidates = reorder_retriever.invoke(
    "RAG 검색 품질을 높이는 방법을 설명해 주세요."
)
reordered_docs = LongContextReorder().transform_documents(candidates)

print("=== 원래 순서 ===")
for i, doc in enumerate(candidates, start=1):
    print(i, doc.page_content)

print("\n=== 재배치 후 ===")
for i, doc in enumerate(reordered_docs, start=1):
    print(i, doc.page_content)
