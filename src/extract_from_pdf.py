"""Adobe PDF Extract API로 PDF의 텍스트, 표, 그림을 추출한다."""

import argparse
import json
import logging
import os
import shutil
import unicodedata
from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult


def extract_pdf(input_path: Path, output_dir: Path) -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
    client_id = os.getenv("ADOBE_CLIENT_ID")
    client_secret = os.getenv("ADOBE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            ".env에 ADOBE_CLIENT_ID와 ADOBE_CLIENT_SECRET을 설정하세요 "
        )

    if not input_path.is_file():
        decomposed_path = input_path.with_name(unicodedata.normalize("NFD", input_path.name))
        if decomposed_path.is_file():
            input_path = decomposed_path

    with input_path.open("rb") as input_file:
        input_stream = input_file.read()

    credentials = ServicePrincipalCredentials(client_id=client_id, client_secret=client_secret)
    pdf_services = PDFServices(credentials=credentials)
    input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
    params = ExtractPDFParams(
        table_structure_type=TableStructureType.CSV,
        elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
        elements_to_extract_renditions=[
            ExtractRenditionsElementType.FIGURES,
            ExtractRenditionsElementType.TABLES,
        ],
    )
    location = pdf_services.submit(ExtractPDFJob(input_asset=input_asset, extract_pdf_params=params))
    result = pdf_services.get_job_result(location, ExtractPDFResult).get_result().get_resource()
    content = pdf_services.get_content(result).get_input_stream()

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "result.zip"
    zip_path.write_bytes(content)

    with ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            destination = (output_dir / entry.filename).resolve()
            if not destination.is_relative_to(output_dir.resolve()) or destination == zip_path.resolve():
                raise ValueError(f"ZIP 내부 경로가 출력 디렉터리를 벗어납니다: {entry.filename}")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    structured_data = json.loads((output_dir / "structuredData.json").read_text(encoding="utf-8"))
    text = "\n".join(
        element["Text"]
        for element in structured_data["elements"]
        if isinstance(element.get("Text"), str)
    )
    (output_dir / "text.txt").write_text(text, encoding="utf-8")
    logging.info("추출 완료: %s (구조화 JSON, 텍스트, 표 및 이미지 렌디션)", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adobe PDF Extract API로 PDF 내용을 추출합니다.")
    parser.add_argument("pdf", type=Path, help="입력 PDF 파일")
    parser.add_argument("--output", type=Path, help="출력 폴더 (기본값: PDF 이름_extracted)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    extract_pdf(args.pdf, args.output or args.pdf.with_name(f"{args.pdf.stem}_extracted"))