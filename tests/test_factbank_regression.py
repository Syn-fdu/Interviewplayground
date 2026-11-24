"""Pinned parity with the pre-refactor v15 classifier; no model or network calls."""
import json, pathlib, sys, unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from python_factbank import classify
from python_binary_factbank import classify_source

class FactbankRegression(unittest.TestCase):
    def test_pinned_pre_refactor_results_for_every_rule_category(self):
        cases=json.loads((ROOT/'tests/fixtures/factbank_parity.json').read_text('utf8'))
        self.assertGreaterEqual(len(cases),20)
        for index,case in enumerate(cases):
            with self.subTest(index=index,text=case['input']['original_text'][:30]):
                self.assertEqual(classify(case['input']),case['expected'])
    def test_binary_source_adapter_is_retained(self):
        result=classify_source({'text':'I use Copilot every day.',
            'source_question':'What is your daily workflow?'})
        self.assertIn(result['fact_category'],('reality','memory'))
        self.assertIn('classification_reason',result)
    def test_only_one_canonical_classifier(self):
        self.assertTrue((ROOT/'python_factbank.py').is_file())
        self.assertFalse(list(ROOT.glob('python_factbank_v*.py')))
        self.assertFalse((ROOT/'legacy_question_first_v013.py').exists())

if __name__=='__main__': unittest.main()
