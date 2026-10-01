"""실험 매뉴얼 text.txt를 임베딩용 구조화 JSON으로 변환한다."""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


OUTPUT_SCHEMA = {
    "name": "lab_manual_experiments",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "experiments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "실험명": {"type": "string"},
                        "주제": {"type": ["string", "null"]},
                        "실험 방법": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "순서": {"type": "integer"},
                                    "내용": {"type": "string"},
                                },
                                "required": ["순서", "내용"],
                                "additionalProperties": False,
                            },
                        },
                        "준비물": {"type": "array", "items": {"type": "string"}},
                        "주의사항": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["실험명", "주제", "실험 방법", "준비물", "주의사항"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["experiments"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """당신은 실험 매뉴얼을 기계적으로 구조화하는 추출기다.
다음 규칙을 반드시 지켜라.
1. 원문에 명시된 내용만 추출하고 추론, 보완, 요약, 번역, 교정하지 않는다.
2. 모든 문자열은 원문에 실제로 있는 연속된 문구를 그대로 복사한다.
3. 한 파일에 여러 실험이 있으면 실험별로 분리한다.
4. STEP, 하위 번호, 글머리표 등 단계의 계층은 무시하고, 실제 수행 절차를 원문 등장 순서대로 1부터 다시 번호 매긴다.
5. 제목, Objective, 분석 결과 요구처럼 수행 절차가 아닌 내용은 실험 방법에 넣지 않는다.
6. 주제, 준비물, 주의사항이 원문에 없으면 각각 null 또는 빈 배열을 사용한다.
7. 표의 값은 원문만으로 의미가 명확할 때만 해당 절차에 포함하고, 내용을 재구성하지 않는다.
8. 누락을 줄이되 불확실한 내용은 만들지 말고 제외한다.
"""


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _validate_grounding(result: dict[str, Any], source: str) -> None:
    normalized_source = _normalized(source)
    experiments = result.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("응답에 실험이 없습니다.")

    for experiment_index, experiment in enumerate(experiments, start=1):
        steps = experiment["실험 방법"]
        expected_orders = list(range(1, len(steps) + 1))
        actual_orders = [step["순서"] for step in steps]
        if actual_orders != expected_orders:
            raise ValueError(f"실험 {experiment_index}의 단계 순서가 연속적이지 않습니다.")

        values = [experiment["실험명"]]
        if experiment["주제"] is not None:
            values.append(experiment["주제"])
        values.extend(step["내용"] for step in steps)
        values.extend(experiment["준비물"])
        values.extend(experiment["주의사항"])
        for value in values:
            if not isinstance(value, str) or not _normalized(value):
                raise ValueError(f"실험 {experiment_index}에 비어 있거나 잘못된 문자열이 있습니다.")
            if _normalized(value) not in normalized_source:
                raise ValueError(
                    f"실험 {experiment_index}에 원문 근거가 없는 내용이 있습니다: {value!r}"
                )


def convert_text(text: str, *, model: str = "gpt-4.1-mini", client: Any = None) -> dict[str, Any]:
    """원문 하나를 구조화하고 모든 추출 문자열이 원문에 있는지 검증한다."""
    if not text.strip():
        raise ValueError("입력 텍스트가 비어 있습니다.")
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": OUTPUT_SCHEMA},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<manual>\n{text}\n</manual>"},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI 응답이 비어 있습니다.")
    result = json.loads(content)
    _validate_grounding(result, text)
    return result


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    model: str = "gpt-4.1-mini",
    client: Any = None,
) -> None:
    """텍스트 파일 하나를 변환하고 검증이 끝난 결과만 저장한다."""
    text = input_path.read_text(encoding="utf-8")
    result = convert_text(text, model=model, client=client)
    output_path.write_text(
        json.dumps(result["experiments"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_manuals(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("text.txt") if path.is_file())


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="data 아래의 실험 매뉴얼 text.txt를 OpenAI로 구조화합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-name", default="experiments.json")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    load_dotenv(project_root / ".env")
    manuals = find_manuals(args.data_dir)
    if not manuals:
        raise FileNotFoundError(f"text.txt를 찾을 수 없습니다: {args.data_dir}")

    for manual in manuals:
        output_path = manual.with_name(args.output_name)
        convert_file(manual, output_path, model=args.model)
        print(f"변환 완료: {manual} -> {output_path}")


if __name__ == "__main__":
    main()