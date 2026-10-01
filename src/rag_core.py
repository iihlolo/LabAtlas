"""
RAG (Retrieval-Augmented Generation) Core Module

저장된 FAISS Vector Store에서 질문과 관련된 Document를 검색하는 Retriever를 제공합니다.
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_image_documents(data_dir: Path) -> list[Document]:
    """images.json의 각 이미지를 검색 가능한 문서로 변환합니다."""
    documents = []
    for manifest in data_dir.rglob("named_images/images.json"):
        for image in json.loads(manifest.read_text(encoding="utf-8")):
            if not Path(image["image_path"]).is_file():
                continue
            documents.append(Document(
                page_content="\n".join(filter(None, [
                    image["label"], image["description"], image["context"],
                ])),
                metadata={
                    "source": image["source_pdf"],
                    "page": image["page"] - 1,
                    "image_path": image["image_path"],
                    "image_label": image["label"],
                    "kind": "image",
                },
            ))
    return documents


def _experiment_header(experiment: dict) -> str:
    lines = [f"실험명: {experiment['실험명']}"]
    if experiment.get("주제"):
        lines.append(f"주제: {experiment['주제']}")
    return "\n".join(lines)


def load_experiment_documents(data_dir: Path) -> list[Document]:
    """experiments.json을 단계 및 정보 종류별 검색 문서로 변환합니다."""
    documents = []
    for manifest in data_dir.rglob("experiments.json"):
        experiments = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(experiments, list):
            raise ValueError(f"experiments.json의 최상위 값은 배열이어야 합니다: {manifest}")
        for experiment in experiments:
            header = _experiment_header(experiment)
            common_metadata = {
                "source": str(manifest.resolve()),
                "experiment_name": experiment["실험명"],
                "topic": experiment.get("주제") or "",
                "kind": "experiment",
            }
            for step in experiment.get("실험 방법", []):
                documents.append(Document(
                    page_content=f"{header}\n실험 방법 {step['순서']}. {step['내용']}",
                    metadata={
                        **common_metadata,
                        "section": "실험 방법",
                        "step": step["순서"],
                    },
                ))
            for section in ("준비물", "주의사항"):
                items = experiment.get(section, [])
                if items:
                    content = "\n".join(f"- {item}" for item in items)
                    documents.append(Document(
                        page_content=f"{header}\n{section}:\n{content}",
                        metadata={**common_metadata, "section": section},
                    ))
    return documents


def add_json_documents(vectorstore: FAISS, data_dir: Path) -> tuple[int, int]:
    """images.json과 experiments.json 문서를 기존 벡터스토어에 추가합니다."""
    image_documents = load_image_documents(data_dir)
    experiment_documents = load_experiment_documents(data_dir)
    documents = image_documents + experiment_documents
    if documents:
        vectorstore.add_documents(documents)
    return len(image_documents), len(experiment_documents)


def build_retriever(store_path: Path | None = None, k: int = 3):
    """질문과 관련된 상위 k개 Document를 반환하는 Retriever를 만듭니다."""

    store_path = store_path or Path(__file__).resolve().parent.parent / "vector_store_path"
    if not (store_path / "index.faiss").is_file() or not (store_path / "index.pkl").is_file():
        raise FileNotFoundError(f"저장된 벡터 DB가 없습니다: {store_path}")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(store_path), embeddings, allow_dangerous_deserialization=True
    )

    # similarity 검색 결과 중 상위 k개만 반환합니다.
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def format_docs(docs):
    """검색된 Document 목록을 LLM Prompt에 넣을 문자열로 합칩니다."""

    return "\n\n".join(
        f"[문서 {i}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )
