"""Offline integration: audited guide -> native Outline -> level-specific preparation cache."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from playground.interview_agent.outline.models import Outline
from playground.research import group_interviewer_bridge as bridge
from playground.research import interviewer_preparation as prep
from playground.research.outline_levels import (LEVELS, catalog, build_native_outline, preview,
                                                has_refined_guide, normalize_level)


class ThreeLevelOutlinesTest(unittest.TestCase):
    def test_all_refined_datasets_parse_in_native_agent(self):
        self.assertEqual(len(catalog()), 7)
        core_datasets = ('ai_coding_workflows', 'ai_developer_interviews',
                         'astronomy_open_data', 'international_student_barriers',
                         'oss_dependency_security', 'parental_engagement_teachers',
                         'securops_experts')
        self.assertTrue(all(has_refined_guide(ds) for ds in core_datasets))
        for dataset, original in catalog().items():
            with self.subTest(dataset=dataset):
                level0 = Outline.from_dict(build_native_outline(dataset, 'level0'))
                level1 = Outline.from_dict(build_native_outline(dataset, 'level1'))
                level2 = Outline.from_dict(build_native_outline(dataset, 'level2'))
                self.assertTrue(1 <= len(level0.questions()) <= 6)
                self.assertTrue(5 <= len(level1.questions()) <= 15)
                self.assertEqual([q.text for q in level1.questions()],
                                 [q.text for q in level2.questions()])
                self.assertEqual([q.id for q in level1.questions()],
                                 [q.id for q in level2.questions()])
                self.assertEqual(sum(len(q.probes) for q in level2.questions()),
                                 len(original['level2']))
                self.assertTrue(5 <= len(original['level2']) <= 10)
                self.assertFalse(any(q.probes for q in level1.questions()))
                self.assertEqual(len(set(q.id for q in level2.questions())),len(level2.questions()))
                # Strict separation: these resources contain no raw participant answers.
                self.assertNotIn('respondent_answer', json.dumps(build_native_outline(dataset,'level2')))

    def test_dataset_local_loading_and_exact_preview(self):
        from playground.research.outline_levels import BUNDLED_ROOT, summary, get_guide
        self.assertEqual(summary()['storage'], 'dataset_local_bundled_data')
        for dataset in catalog():
            files=list(BUNDLED_ROOT.glob(f'*/social_research/{dataset}/processed/outline_levels.json'))
            self.assertEqual(len(files),1)
            original=get_guide(dataset)['levels']
            for level in LEVELS:
                value=preview(dataset,level)
                self.assertEqual(value['questions'],original[level])
                self.assertEqual(value['level'],level)
        self.assertFalse((BUNDLED_ROOT.parent/'outlines/refined_three_levels.json').exists())

    def test_no_fabricated_three_levels_for_unreviewed_datasets(self):
        self.assertFalse(has_refined_guide('tv_dialogues'))
        for bad in ('level3', 'source', None, 1):
            with self.assertRaises(ValueError):
                normalize_level(bad)
        with self.assertRaises(ValueError):
            build_native_outline('tv_dialogues', 'level0')

    def test_actual_preparation_path_and_native_bridge_receive_correct_level(self):
        dataset='ai_developer_interviews'
        scenario={'scenario_id':'scene_test', 'provenance':{
            'dataset':dataset,'candidate_id':'CAND_TEST',
            'source_fingerprint_sha256':'a'*64}}
        dummy_config=SimpleNamespace(model='offline-test-model',base_url='http://localhost/unused')
        def fill_slots(outline,llm,batch=6):
            for question in outline.questions():
                for slot in question.required_slots:
                    slot.source='llm'
            return len(outline.questions())
        with tempfile.TemporaryDirectory() as td, patch('playground.common.model_services.get',return_value=dummy_config), \
             patch.object(prep,'_OUTLINE_CACHE_DIR',Path(td)), \
             patch.object(prep,'enrich_slots',side_effect=fill_slots) as enrich:
            prep.clear()
            for lvl,n in [('level0',4),('level1',10),('level2',10)]:
                report=prep.prepare_public_outline(b'',scenario,1,provider='shubi',llm=object(),outline_level=lvl)
                self.assertTrue(report['ready'])
                self.assertEqual(report['outline_level'],lvl)
                self.assertEqual(report['question_count'],n)
                selected=prep.read_prepared(scenario,1,lvl)
                self.assertTrue(selected['ready'])
                raw=bridge._find_outline(b'',dataset,'CAND_TEST','a'*64,1,lvl)
                self.assertEqual(raw,selected['agent_outline'])
            self.assertEqual(enrich.call_count,3)
            self.assertEqual(sum(len(q.probes) for q in Outline.from_dict(
                prep.read_prepared(scenario,1,'level2')['agent_outline']).questions()),7)
            # Same model/guide cache hit is isolated from the other two levels.
            again=prep.prepare_public_outline(b'',scenario,1,provider='shubi',llm=object(),outline_level='level2')
            self.assertTrue(again['cached'])
            self.assertEqual(enrich.call_count,3)
            prep.clear()

    def test_scene_binds_three_levels_and_interview_selects(self):
        root=Path(__file__).resolve().parents[1]
        html=(root/'index.html').read_text('utf-8')
        js=(root/'assets'/'batch_experiments.js').read_text('utf-8')
        server=(root/'server.py').read_text('utf-8')
        projection=(root/'playground/research/guide_level_projection.py').read_text('utf-8')
        self.assertNotIn('id="builderGuideLevel"',html)
        self.assertIn('id="interviewGuideLevel"',html)
        self.assertIn('id="batchGuideLevel"',html)
        self.assertIn('/api/v3/scenario/select-guide',html)
        self.assertIn('attach_all_levels(scenario, body.get("guide_category"))',server)
        self.assertIn('select_attached_level(updated,level)',server)
        self.assertIn('def attach_all_levels',projection)
        self.assertIn("guide_level:$('batchGuideLevel').value",js)


if __name__ == '__main__':
    unittest.main()
