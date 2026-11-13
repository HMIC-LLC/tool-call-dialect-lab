import unittest
from pathlib import Path

from dialect_lab import load_cases, score
from tool_dialect import parse_detail


HERE = Path(__file__).resolve().parents[1]


class DialectLabTests(unittest.TestCase):
    def test_synthetic_fixture_reports_strictness_without_model_claim(self):
        cases = load_cases(HERE / "examples" / "synthetic-cases.jsonl")
        report = score(cases, "0" * 64)
        self.assertEqual(report["summary"]["tagged"], {"exact_matches": 4, "total": 5})
        self.assertEqual(report["summary"]["standard"], {"exact_matches": 5, "total": 5})
        self.assertEqual(report["summary"]["lenient"], {"exact_matches": 4, "total": 5})
        self.assertFalse(any("model capability score" == row.get("classification") for row in report["results"]))

    def test_offered_tool_constraint_rejects_unlisted_untagged_call(self):
        result = parse_detail('{"name":"service_restart","arguments":{"service":"billing"}}',
                              tools=["service_status"], level="standard")
        self.assertEqual(result.calls, [])
        self.assertTrue(result.refused)


if __name__ == "__main__":
    unittest.main()
