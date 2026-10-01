"""기존 Adobe 추출물의 그림을 이름 붙이고 검색용 목록을 생성한다."""

import argparse
import base64
import json
import re
import shutil
from pathlib import Path


def nearby_text(elements: list[dict], position: int) -> str:
    page = elements[position].get("Page")
    before = [item["Text"].strip() for item in elements[max(0, position - 12):position]
              if item.get("Page") == page and isinstance(item.get("Text"), str) and item["Text"].strip()]
    after = [item["Text"].strip() for item in elements[position + 1:position + 6]
             if item.get("Page") == page and isinstance(item.get("Text"), str) and item["Text"].strip()]
    return " ".join(before[-4:] + after[:2])[:1000]


def describe_image(image_path: Path, context: str) -> tuple[str, str]:
    from openai import OpenAI

    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "Describe the actual image for a lab manual search index. Return JSON with "
                "'label' (2-6 English words, lowercase) and 'description' (one factual sentence). "
                "Do not infer components not visible in the image."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": f"Nearby PDF text (may not describe this figure): {context}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "high"}},
            ]},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"이미지 설명 응답이 비어 있습니다: {image_path}")
    result = json.loads(content)
    return str(result["label"]), str(result["description"])


def catalog(extracted_dir: Path, *, vision: bool = False, prefix: str | None = None) -> Path:
    data = json.loads((extracted_dir / "structuredData.json").read_text(encoding="utf-8"))
    elements = data["elements"]
    output_dir = extracted_dir / "named_images"
    output_dir.mkdir(exist_ok=True)
    prefix = re.sub(
        r"[^a-z0-9]+", "_", (prefix or extracted_dir.name.removesuffix("_extracted")).lower()
    ).strip("_")[:48] or "lab"
    records = []
    for position, element in enumerate(elements):
        if "Figure" not in element.get("Path", ""):
            continue
        for relative_path in element.get("filePaths", []):
            source = (extracted_dir / relative_path).resolve()
            if not source.is_relative_to(extracted_dir.resolve()) or not source.is_file():
                raise FileNotFoundError(f"그림 파일을 찾을 수 없습니다: {relative_path}")
            context = nearby_text(elements, position)
            label, description = describe_image(source, context) if vision else ("figure", "")
            slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:60] or "figure"
            page = element.get("Page", 0) + 1
            name = f"{prefix}_p{page:02d}_{slug}_{len(records) + 1:03d}{source.suffix.lower()}"
            target = output_dir / name
            shutil.copyfile(source, target)
            records.append({
                "image_path": str(target.resolve()),
                "source_image": relative_path,
                "page": page,
                "label": label,
                "description": description,
                "context": context,
                "source_pdf": str(extracted_dir.with_name(extracted_dir.name.removesuffix("_extracted") + ".pdf").resolve()),
            })
    manifest = output_dir / "images.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="기존 Adobe 추출 결과에서 이미지 검색 목록을 생성합니다.")
    parser.add_argument("extracted_dir", type=Path, help="structuredData.json이 있는 폴더")
    parser.add_argument("--vision", action="store_true", help="OpenAI 비전으로 이미지 이름과 설명 생성 (API 과금)")
    parser.add_argument("--prefix", help="파일명 접두사 (예: ecl_1)")
    args = parser.parse_args()
    if args.vision:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    result_path = catalog(args.extracted_dir, vision=args.vision, prefix=args.prefix)
    print(f"이미지 목록 생성 완료: {result_path}")