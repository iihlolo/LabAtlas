"""data 폴더의 PDF를 임베딩하여 로컬 FAISS 저장소를 생성합니다."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__:
    from .rag_core import add_json_documents, load_experiment_documents, load_image_documents
else:
    from rag_core import add_json_documents, load_experiment_documents, load_image_documents


def load_and_split(file_path: Path):
    """PDF 파일을 읽고 검색용 청크로 나눕니다."""
    docs = PyPDFLoader(str(file_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    return splitter.split_documents(docs)


def main():
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")
    pdf_paths = sorted(
        path for path in (project_root / "data").rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    data_dir = project_root / "data"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pdf_documents = [
        document
        for pdf_path in pdf_paths
        for document in load_and_split(pdf_path)
    ]
    if pdf_documents:
        vectorstore = FAISS.from_documents(pdf_documents, embeddings)
        image_count, experiment_count = add_json_documents(vectorstore, data_dir)
    else:
        json_documents = load_image_documents(data_dir)
        experiment_documents = load_experiment_documents(data_dir)
        json_documents.extend(experiment_documents)
        if not json_documents:
            raise FileNotFoundError("data 폴더에서 인덱싱할 PDF 또는 JSON을 찾을 수 없습니다.")
        vectorstore = FAISS.from_documents(json_documents, embeddings)
        image_count = len(json_documents) - len(experiment_documents)
        experiment_count = len(experiment_documents)

    store_path = project_root / "vector_store_path"
    vectorstore.save_local(str(store_path))
    print(f"저장된 PDF: {len(pdf_paths)}개 ({store_path})")
    print("총 벡터 개수:", vectorstore.index.ntotal)
    print("이미지 인덱스:", image_count, "개")
    print("실험 JSON 인덱스:", experiment_count, "개")


if __name__ == "__main__":
    main()