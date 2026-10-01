import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.catalog_images import catalog
from src.index_documents import load_image_documents
from src.rag_pipeline import image_references


class CatalogImagesTests(unittest.TestCase):
    def test_existing_extract_generates_searchable_image(self):
        with tempfile.TemporaryDirectory() as directory:
            extracted = Path(directory) / "sample_extracted"
            figures = extracted / "figures"
            figures.mkdir(parents=True)
            (figures / "fileoutpart0.png").write_bytes(b"png")
            (extracted / "structuredData.json").write_text(json.dumps({"elements": [
                {"Page": 0, "Text": "Build a resistor diode circuit"},
                {"Page": 0, "Path": "//Document/Figure", "filePaths": ["figures/fileoutpart0.png"]},
            ]}))

            with patch("src.catalog_images.describe_image", return_value=(
                "resistor diode circuit", "A resistor and diode connected in series."
            )):
                manifest = catalog(extracted, vision=True, prefix="ecl_1")

            record = json.loads(manifest.read_text())[0]
            self.assertEqual(Path(record["image_path"]).name, "ecl_1_p01_resistor_diode_circuit_001.png")
            self.assertEqual(record["context"], "Build a resistor diode circuit")
            self.assertEqual(Path(record["image_path"]).read_bytes(), b"png")
            self.assertEqual((figures / "fileoutpart0.png").read_bytes(), b"png")

            docs = load_image_documents(Path(directory))
            self.assertEqual(len(docs), 1)
            self.assertIn("resistor diode circuit", docs[0].page_content)
            self.assertIn("A resistor and diode", docs[0].page_content)
            self.assertEqual(docs[0].metadata["page"], 0)
            self.assertEqual(len(image_references(docs)), 1)

    def test_references_only_retrieved_images_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            path.write_bytes(b"png")
            doc = Document(page_content="circuit", metadata={
                "image_path": str(path), "image_label": "circuit", "page": 1
            })
            references = image_references([Document(page_content="other"), doc, doc])
            self.assertEqual(len(references), 1)
            self.assertIn("figure.png", references[0])


if __name__ == "__main__":
    unittest.main()