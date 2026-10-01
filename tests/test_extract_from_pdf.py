import io
import json
import os
import tempfile
import unittest
import unicodedata
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from src import extract_from_pdf


class ExtractPDFTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pdf = self.root / "sample.pdf"
        self.pdf.write_bytes(b"%PDF-1.7")
        self.output = self.root / "sample_extracted"

    def run_with_archive(self, entries, credentials=None, input_path=None):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            for name, content in entries.items():
                archive.writestr(name, content)

        if credentials is None:
            credentials = {
                "PDF_SERVICES_CLIENT_ID": "test-id",
                "PDF_SERVICES_CLIENT_SECRET": "test-secret",
            }
        with patch.dict(os.environ, credentials, clear=True), patch.object(extract_from_pdf, "load_dotenv") as load_env, patch.object(
            extract_from_pdf, "ServicePrincipalCredentials"
        ) as credentials_class, patch.object(extract_from_pdf, "PDFServices") as service_class, patch.object(
            extract_from_pdf, "ExtractPDFJob"
        ) as job_class:
            service = service_class.return_value
            service.get_content.return_value.get_input_stream.return_value = buffer.getvalue()
            extract_from_pdf.extract_pdf(input_path or self.pdf, self.output)
            load_env.assert_called_once_with(dotenv_path=Path(extract_from_pdf.__file__).resolve().parent.parent / ".env")
            credentials_class.assert_called_once_with(client_id="test-id", client_secret="test-secret")
            return job_class.call_args.kwargs["extract_pdf_params"]

    def test_reads_existing_adobe_env_keys(self):
        self.run_with_archive(
            {"structuredData.json": json.dumps({"elements": [{"Text": "hello"}]})},
            credentials={"ADOBE_CLIENT_ID": "test-id", "ADOBE_CLIENT_SECRET": "test-secret"},
        )
        self.assertEqual((self.output / "text.txt").read_text(), "hello")

    def test_accepts_nfc_name_for_nfd_pdf(self):
        name = "화학공학생명실험1 매뉴얼.pdf"
        self.pdf.rename(self.root / unicodedata.normalize("NFD", name))
        self.run_with_archive(
            {"structuredData.json": json.dumps({"elements": []})},
            input_path=self.root / name,
        )
        self.assertTrue((self.output / "result.zip").exists())

    def test_extracts_text_and_renditions(self):
        entries = {
            "structuredData.json": json.dumps({
                "elements": [{"Text": "E = mc²"}, {"Text": "∫ f(x) dx"}, {"Path": "//Document/Table"}]
            }),
            "figures/figure1.png": b"image",
            "tables/table1.csv": b"a,b\n1,2\n",
        }
        params = self.run_with_archive(entries)

        self.assertEqual((self.output / "text.txt").read_text(), "E = mc²\n∫ f(x) dx")
        self.assertEqual((self.output / "figures/figure1.png").read_bytes(), b"image")
        self.assertEqual((self.output / "tables/table1.csv").read_bytes(), b"a,b\n1,2\n")
        self.assertTrue((self.output / "result.zip").exists())
        self.assertEqual(params.get_elements_to_extract(), ["text", "tables"])
        self.assertEqual(params.get_elements_to_extract_renditions(), ["figures", "tables"])

    def test_rejects_zip_path_outside_output(self):
        with self.assertRaises(ValueError):
            self.run_with_archive({"../outside.txt": "unsafe"})
        self.assertFalse((self.root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()