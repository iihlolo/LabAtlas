uv sync

프로젝트 루트의 `.env`에 `OPENAI_API_KEY`를 설정한 뒤, PDF가 변경될 때 벡터 DB를 생성합니다.

```sh
uv run --locked python src/index_documents.py
```

이 명령은 PDF 청크와 함께 `data/**/named_images/images.json`, `data/**/experiments.json`을 읽어 FAISS에 저장합니다. 실험 방법은 단계별 문서로, 준비물과 주의사항은 각 항목을 빠뜨리지 않도록 종류별 문서로 인덱싱됩니다. PDF가 없어도 두 JSON 중 하나가 있으면 벡터 DB를 만들 수 있습니다.

이후 질문 파이프라인은 저장된 `vector_store_path`를 재사용합니다. PDF를 다시 임베딩하지 않습니다.

```sh
uv run --locked python src/rag_pipeline.py
```

저장된 청크와 메타데이터 확인:

```sh
uv run --locked python src/check_faiss.py
```

추출된 모든 `data/**/text.txt`를 실험별 JSON으로 변환:

```sh
uv run --locked python src/manual_text_to_json.py
```

OpenAI 모델을 temperature 0과 엄격한 JSON Schema로 호출하며, 원문에서 확인되지 않는 문자열이 하나라도 있으면 저장하지 않습니다. 각 입력 파일 옆에 `experiments.json`이 생성되고, 계층형 실험 단계는 등장 순서대로 1부터 다시 번호가 매겨집니다. 모델은 `--model`, 입력 루트는 `--data-dir`, 출력 파일명은 `--output-name`으로 변경할 수 있습니다.

Adobe PDF Extract API로 PDF에서 텍스트, 표, 그림 추출:

1. Adobe PDF Services 자격 증명을 발급받아 프로젝트 루트의 `.env`에 `ADOBE_CLIENT_ID`와 `ADOBE_CLIENT_SECRET`을 설정합니다.
2. 다음 명령을 실행합니다.

```sh
uv run --locked python src/extract_from_pdf.py "data/화학공학_실험_매뉴얼/실험1_매뉴얼.pdf"
```

`--output`을 생략하면 PDF 옆의 `<PDF이름>_extracted` 폴더에 저장됩니다. 결과 폴더에는 원본 `result.zip`, 구조와 읽기 순서를 포함한 `structuredData.json`, 텍스트를 모은 `text.txt`, 이미지 렌디션 PNG와 표 렌디션 CSV가 생성됩니다(문서에 해당 요소가 있을 때). 수식도 Adobe가 텍스트로 인식한 경우 원래 문자가 보존되지만, 그림이나 스캔으로 된 수식의 LaTeX 변환은 제공하지 않습니다. 이미지로 인식된 수식은 그림 렌디션을 확인하세요.

이미 추출한 그림을 RAG 답변에서 사용하려면 PDF를 재추출하지 않고 다음을 실행합니다.

```sh
uv run --locked python src/catalog_images.py 'data/실험_매뉴얼/lab_extracted' --prefix lab_1 --vision
uv run --locked python src/index_documents.py
uv run --locked python src/rag_pipeline.py 'YADH 효소 용액을 어떻게 만들어야 하나요?'
```

카탈로그 모듈은 원본 PNG를 건드리지 않고 `named_images/`에 이름 붙인 사본과 `images.json`을 생성합니다. `--vision`은 OpenAI 이미지 분석을 사용하므로 API 과금이 발생하며, 그림의 내용과 주변 글을 근거로 `lab_1_p05_enzyme_solution_002.png` 같은 이름과 검색 설명을 만듭니다. 이를 생략하면 API 호출 없이 번호 기반 이름과 근처 텍스트만 저장하므로, 그림의 실제 부품을 식별하는 검색은 제한됩니다. 인덱싱 단계는 PDF를 Adobe에 재전송하지 않지만 OpenAI 임베딩 API를 호출합니다. 검색된 그림이 있으면 RAG 답변 뒤에 해당 로컬 PNG 링크가 출력됩니다.