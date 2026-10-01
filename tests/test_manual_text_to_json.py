import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.rag_core import load_experiment_documents
from src.manual_text_to_json import convert_file, convert_text, find_manuals


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.call = None

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(self.result, ensure_ascii=False))
        )])


def fake_client(result):
    completions = FakeCompletions(result)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions_call=completions,
    )


class ManualTextToJsonTests(unittest.TestCase):
    def test_multiple_experiments_are_flattened_and_saved(self):
        source = """실험 1. 합성
주제: 시료 합성
STEP 1. 준비
① 물 10mL를 넣는다.
1) 혼합
② 1분간 섞는다.
Caution
장갑을 착용한다.
실험 2. 측정
Materials
비커, 온도계
① 온도를 측정한다.
"""
        result = {"experiments": [
            {
                "실험명": "실험 1. 합성",
                "주제": "시료 합성",
                "실험 방법": [
                    {"순서": 1, "내용": "물 10mL를 넣는다."},
                    {"순서": 2, "내용": "1분간 섞는다."},
                ],
                "준비물": [],
                "주의사항": ["장갑을 착용한다."],
            },
            {
                "실험명": "실험 2. 측정",
                "주제": None,
                "실험 방법": [{"순서": 1, "내용": "온도를 측정한다."}],
                "준비물": ["비커, 온도계"],
                "주의사항": [],
            },
        ]}
        client = fake_client(result)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "text.txt"
            output_path = Path(directory) / "experiments.json"
            input_path.write_text(source, encoding="utf-8")
            convert_file(input_path, output_path, client=client)

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(saved), 2)
        self.assertEqual([step["순서"] for step in saved[0]["실험 방법"]], [1, 2])
        call = client.completions_call.call
        self.assertEqual(call["temperature"], 0)
        self.assertTrue(call["response_format"]["json_schema"]["strict"])

    def test_ungrounded_content_is_rejected(self):
        source = "실험 1. 측정\n① 온도를 측정한다."
        result = {"experiments": [{
            "실험명": "실험 1. 측정",
            "주제": None,
            "실험 방법": [{"순서": 1, "내용": "보안경을 착용한다."}],
            "준비물": [],
            "주의사항": [],
        }]}

        with self.assertRaisesRegex(ValueError, "원문 근거가 없는 내용"):
            convert_text(source, client=fake_client(result))

    def test_finds_nested_text_files_only(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            nested = data_dir / "분야" / "매뉴얼_extracted"
            nested.mkdir(parents=True)
            expected = nested / "text.txt"
            expected.write_text("실험", encoding="utf-8")
            (nested / "other.txt").write_text("제외", encoding="utf-8")

            self.assertEqual(find_manuals(data_dir), [expected])

    def test_experiments_are_loaded_as_search_documents(self):
        experiments = [{
            "실험명": "고분자합성",
            "주제": "Styrene 중합",
            "실험 방법": [
                {"순서": 1, "내용": "Water bath를 준비한다."},
                {"순서": 2, "내용": "시료를 넣는다."},
            ],
            "준비물": ["유리 vial", "AIBN"],
            "주의사항": ["후드 내에서 진행한다."],
        }]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "분야" / "experiments.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(experiments, ensure_ascii=False), encoding="utf-8")

            documents = load_experiment_documents(Path(directory))

        self.assertEqual(len(documents), 4)
        self.assertEqual(documents[0].metadata["step"], 1)
        self.assertEqual(documents[1].metadata["section"], "실험 방법")
        self.assertIn("실험명: 고분자합성", documents[0].page_content)
        self.assertIn("유리 vial\n- AIBN", documents[2].page_content)
        self.assertEqual(documents[3].metadata["section"], "주의사항")


if __name__ == "__main__":
    unittest.main()