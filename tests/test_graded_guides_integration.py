import unittest
from playground.research.graded_guides import list_datasets, variants, match, question_view
from playground.research.data_intake import candidate_id

class GradedGuideIntegrationTests(unittest.TestCase):
    def test_26_dataset_coverage(self):
        self.assertEqual(len(list_datasets()), 26)
    def test_source_questions_preserved(self):
        for ds in list_datasets():
            for variant in variants(ds):
                for level in ('level0','level1','level2'):
                    if not variant.get(level):
                        continue
                    view=question_view(variant,level)
                    self.assertEqual([x['text'] for x in view[0]['questions']],
                        [q if isinstance(q,str) else q['question'] for q in variant[level]
                         if (q if isinstance(q,str) else q.get('question'))])
    def test_exact_news_interview_mapping(self):
        ds='news_interview'
        record=variants(ds)[0]
        key=record['guide_id'].split('::',1)[1]
        chosen=match(ds,candidate_id(ds,key))
        self.assertEqual(chosen['guide_id'],record['guide_id'])
        self.assertIsNone(match(ds,candidate_id(ds,'nonexistent_interview_unique')))
    def test_category_requires_explicit_selection(self):
        ds='astronomy_open_data';c=candidate_id(ds,'test')
        self.assertIsNone(match(ds,c))
        self.assertEqual(match(ds,c,'data_governance')['category'],'data_governance')
    def test_unified_guide_is_shared(self):
        self.assertEqual(match('delivery_riders',candidate_id('delivery_riders','any'))['guide_kind'],'unified')
    def test_empty_guides_do_not_silently_pass(self):
        record=next((r for r in variants('news_interview') if not r.get('level1')),None)
        if record:
            with self.assertRaises(ValueError):question_view(record,'level1')

if __name__=='__main__':unittest.main()
