"""
환경 확인 스크립트

이 스크립트는 필수 환경변수와 핵심 패키지 설치 여부를 확인합니다.
"""
import os
import sys
from importlib.metadata import version
from importlib.util import find_spec

from dotenv import load_dotenv

load_dotenv()

print("Python:", sys.version.split()[0])

env_vars = [
    ("OPENAI_API_KEY", True),
    ("UPSTAGE_API_KEY", False),
    ("PINECONE_API_KEY", False),
    ("LANGSMITH_API_KEY", False),
    ("LANGSMITH_TRACING", False),
    ("LANGSMITH_PROJECT", False),
]

print("\n[환경변수]")
for name, required in env_vars:
    value = os.getenv(name)
    status = "설정됨" if value else "없음"
    label = "필수" if required else "선택"
    print(f"{name}: {status} ({label})")

import faiss
import langchain
import langchain_classic
import langchain_community
import langchain_openai
import langchain_text_splitters
import langgraph
import pypdf
import rank_bm25

packages = [
    "langchain",
    "langchain-openai",
    "langchain-community",
    "langchain-classic",
    "langchain-text-splitters",
    "langgraph",
    "faiss-cpu",
    "pypdf",
    "python-dotenv",
    "rank-bm25",
]

print("\n[핵심 패키지]")
for package in packages:
    print(f"{package}: {version(package)}")

provider_modules = {
    "langchain_upstage": "Upstage (langchain-upstage)",
    "langchain_pinecone": "Pinecone (langchain-pinecone)",
    "langsmith": "LangSmith (langsmith)",
    "langchain_ollama": "Ollama (langchain-ollama)",
}

print("\n[Provider 통합 패키지]")
for module, label in provider_modules.items():
    print(label + ":", "설치됨" if find_spec(module) else "미설치")

issues = []
if sys.version_info[:2] != (3, 11):
    issues.append(f"Python 3.11이 필요합니다. 현재 버전: {sys.version.split()[0]}")

for name, required in env_vars:
    if required and not os.getenv(name):
        issues.append(f"필수 환경변수가 없습니다: {name}")

if issues:
    print("\n[확인 필요]")
    for issue in issues:
        print("-", issue)
    raise SystemExit(1)

print("\n핵심 환경 확인: OK")
