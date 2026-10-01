# https://www.google.com/search?client=safari&rls=en&q=%EC%A0%84%EA%B8%B0%ED%9A%8C%EB%A1%9C%EC%8B%A4%ED%97%98+%EC%98%81%EC%96%B4%EB%A1%9C&ie=UTF-8&oe=UTF-8
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def main():
  project_root = Path(__file__).resolve().parent.parent
  load_dotenv(dotenv_path=project_root / ".env")
  store_path = project_root / "vector_store_path"
  if not (store_path / "index.faiss").is_file() or not (store_path / "index.pkl").is_file():
    raise FileNotFoundError(f"저장된 벡터 DB가 없습니다: {store_path}")

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  saved_store = FAISS.load_local(
    str(store_path),
    embeddings,
    allow_dangerous_deserialization=True,
  )
  for doc_id in saved_store.index_to_docstore_id.values():
    doc = saved_store.docstore.search(doc_id)
    if not isinstance(doc, Document):
      raise ValueError(f"저장된 청크를 찾을 수 없습니다: {doc_id}")
    print(f"ID: {doc_id}")
    print(f"내용: {doc.page_content[:100]}...")
    print(f"메타데이터: {doc.metadata}\n")

  print("총 벡터 개수:", saved_store.index.ntotal)


if __name__ == "__main__":
  main()
