"""Regression boundaries for Chinese number normalization exclusions."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class ChineseItnReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real ITN package without initializing desktop logging.
        name = "_chinese_itn_review"
        path = ROOT / "core/tools/chinese_itn/__init__.py"
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)],
        )
        cls.module = importlib.util.module_from_spec(spec)
        cls.modules_patch = patch.dict(sys.modules)
        cls.modules_patch.start()
        cls.addClassCleanup(cls.modules_patch.stop)
        sys.modules[name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_omitted_leading_one_is_a_number_not_a_rate_term(self):
        for before, after in (
            ("百二十三", "123"),
            ("百零三", "103"),
            ("百三", "130"),
            ("负百二十三", "-123"),
            ("千三百二十", "1320"),
            ("千三", "1300"),
            ("万三千", "13000"),
            ("一万三", "13000"),
        ):
            with self.subTest(before=before):
                self.assertEqual(self.module.chinese_to_num(before), after)

    def test_leading_decimals_with_units_do_not_depend_on_a_sign(self):
        for before, after in (
            ("点五个百分点", ".5个百分点"),
            ("负点五个百分点", "-.5个百分点"),
            ("正点五个百分点", "+.5个百分点"),
            ("点三毫米", ".3毫米"),
            ("负点三毫米", "-.3毫米"),
            ("正点三毫米", "+.3毫米"),
            ("增加点五个百分点", "增加.5个百分点"),
            ("厚度点三毫米", "厚度.3毫米"),
            ("点五米", ".5米"),
        ):
            with self.subTest(before=before):
                self.assertEqual(self.module.chinese_to_num(before), after)

    def test_action_counters_keep_dot_as_a_verb(self):
        for before, after in (
            ("点一下按钮", "点一下按钮"),
            ("请点两下按钮", "请点两下按钮"),
            ("点三次", "点三次"),
            ("点击三次", "点击3次"),
        ):
            with self.subTest(before=before):
                self.assertEqual(self.module.chinese_to_num(before), after)

    def test_fee_shorthand_and_words_stay_intact(self):
        for before in ("手续费万三", "手续费万二点五", "万一失败", "上百万"):
            with self.subTest(before=before):
                self.assertEqual(self.module.chinese_to_num(before), before)


if __name__ == "__main__":
    unittest.main()
