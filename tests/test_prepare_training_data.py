from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "pipeline" / "05_reward_model"
SCRIPT = STAGE_DIR / "prepare_training_data.py"

_SPEC = importlib.util.spec_from_file_location("prepare_training_data", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
prepare_training_data = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(STAGE_DIR))
try:
    _SPEC.loader.exec_module(prepare_training_data)
finally:
    sys.path.remove(str(STAGE_DIR))
    for module_name in list(sys.modules):
        if module_name == "sources" or module_name.startswith("sources."):
            del sys.modules[module_name]


class PrepareTrainingDataTests(unittest.TestCase):
    def test_truncate_prompt_preserves_tail_question_and_answer_marker(self) -> None:
        prompt = "Legal context:\n" + ("x" * 200) + "\nQuestion: Q?\n\nAnswer:\n"

        truncated = prepare_training_data.truncate_prompt(prompt, max_chars=40)

        self.assertTrue(truncated.startswith("[Legal context truncated"))
        self.assertIn("Question: Q?", truncated)
        self.assertTrue(truncated.endswith("Answer:\n"))

    def test_build_combined_training_records_truncates_only_lrb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lrb_path = Path(tmp) / "lrb.jsonl"
            lrb_path.write_text(
                '{"prompt": "Legal context:\\n'
                + ("x" * 200)
                + '\\nQuestion: Q?\\n\\nAnswer:\\n", "chosen": "yes", "rejected": "no", "split": "correctness"}\n',
                encoding="utf-8",
            )

            with mock.patch.object(
                prepare_training_data,
                "load_cjb_training_records",
                return_value=[{"prompt": "cjb prompt", "chosen": "a", "rejected": "b", "split": "faithfulness_qa"}],
            ):
                records, stats = prepare_training_data.build_combined_training_records(
                    lrb_train=lrb_path,
                    cjb_prompt_template="template",
                    cjb_splits=("faithfulness_qa",),
                    lrb_max_chars=40,
                )

        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["prompt"].startswith("[Legal context truncated"))
        self.assertEqual(records[1]["prompt"], "cjb prompt")
        self.assertEqual(stats["lrb_records"], 1)
        self.assertEqual(stats["lrb_truncated"], 1)
        self.assertEqual(stats["cjb_records"], 1)


if __name__ == "__main__":
    unittest.main()
