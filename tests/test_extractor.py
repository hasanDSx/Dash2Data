"""Extractor tests with a fake Gemini SDK (no network, no google-genai needed)."""
import json
import os
import sys
import types as pytypes
import unittest
from unittest import mock

from core.pipeline import build_dataset
from core.schemas import SpecError, load_spec

CASES = os.path.join(os.path.dirname(__file__), "..", "benchmark", "cases")
GOOD = open(os.path.join(CASES, "retail.spec.json"), encoding="utf-8").read()


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


def install_fake_sdk(responses):
    """responses: list of str (text) or int (raise APIError(code))."""
    queue = list(responses)
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append((model, contents))
            item = queue.pop(0)
            if isinstance(item, int):
                raise FakeAPIError(item)
            return pytypes.SimpleNamespace(text=item)

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    gtypes = pytypes.ModuleType("google.genai.types")
    errors = pytypes.ModuleType("google.genai.errors")
    genai.Client = Client
    genai.types, genai.errors = gtypes, errors
    gtypes.GenerateContentConfig = lambda **kw: kw
    gtypes.Part = pytypes.SimpleNamespace(from_bytes=lambda data, mime_type: ("img", mime_type))
    errors.APIError = FakeAPIError
    google.genai = genai
    mods = {"google": google, "google.genai": genai, "google.genai.types": gtypes, "google.genai.errors": errors}
    return mock.patch.dict(sys.modules, mods), calls


class TestExtractor(unittest.TestCase):
    def run_extract(self, responses):
        patcher, calls = install_fake_sdk(responses)
        with patcher, mock.patch("time.sleep"):
            import importlib
            import core.extractor as ex
            importlib.reload(ex)
            return ex.extract_spec([(b"x", "image/png")], "key", hints="EGP"), calls

    def test_success(self):
        (spec, warns, raw), calls = self.run_extract([GOOD])
        self.assertEqual(len(calls), 1)
        self.assertIn("USER HINTS", calls[0][1][-1])
        self.assertTrue(spec.targets)

    def test_fenced_json_accepted(self):
        (spec, _, _), _ = self.run_extract([f"```json\n{GOOD}\n```"])
        self.assertTrue(spec.measures)

    def test_repair_loop(self):
        (spec, warns, _), calls = self.run_extract(["not json at all", GOOD])
        self.assertEqual(len(calls), 2)
        self.assertIn("rejected", calls[1][1][-1])
        self.assertTrue(any("repair" in w for w in warns))

    def test_gives_up_after_repairs(self):
        with self.assertRaises(SpecError):
            self.run_extract(["bad", "bad", "bad"])

    def test_retries_on_503(self):
        (spec, _, _), calls = self.run_extract([503, 503, GOOD])
        self.assertEqual(len(calls), 3)

    def test_non_retryable_error_propagates(self):
        with self.assertRaises(FakeAPIError):
            self.run_extract([400])


class TestSpecRoundTrip(unittest.TestCase):
    def test_regenerate_from_edited_json_is_stable(self):
        spec, _ = load_spec(json.loads(GOOD))
        a = build_dataset(spec, seed=4)
        spec2, _ = load_spec(json.loads(json.dumps(spec.to_dict())))
        b = build_dataset(spec2, seed=4)
        self.assertEqual(a.summary, b.summary)


if __name__ == "__main__":
    unittest.main()
