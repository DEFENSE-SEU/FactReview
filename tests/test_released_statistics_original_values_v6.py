"""One offline public v6 control; model semantics and execution are mocks."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from assessment.rules import assess_claim
from common.config import Settings
from common.run_stats import run_scope
from fact_generation.execution.runner import ExecutionConfig, Observation, RunOutcome, execute_plans
from llm.client import LLMConfig
from schemas.claim import Claim
from tests.experiment_authority_fixtures import authority_scope_fixture
from tests.test_released_statistics_public_v5 import _materials, _selection
from verification.contracts import BranchResult
from verification.dispatch import verify_claims
from verification.execution_projection import digest


def _raw_selection(candidate, text):
    canonical = _selection(candidate, text)
    proposal = canonical['proposal']
    canonical['version'] = 'released-statistics-originals-choice-v1'
    proposal['version'] = 'released-statistics-originals-v1'
    for field in proposal['field_bindings']:
        value = candidate['field_inventory'][field['path']]
        field.pop('value_digest')
        field['original_value'] = copy.deepcopy(value)
        for fragment in field['fragments']:
            fragment.pop('text_digest')
            fragment['original_text'] = value[fragment['start']:fragment['end']]
        if isinstance(value, str) and value.endswith('\u2003'):
            field['fragments'][0].update(end=len(value)-1, original_text=value[:-1])
            field['fragments'].append({'start': len(value)-1, 'end': len(value),
                'original_text': '\u2003', 'atom_ids': [], 'separator': True})
    ids = proposal['claim_bindings'][0]['atom_ids']
    proposal['claim_bindings'] = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line.rstrip('\n'))
        proposal['claim_bindings'].append({'start': start, 'end': end,
            'original_text': text[start:end], 'atom_ids': ids, 'separator': False})
        if end < start + len(line):
            proposal['claim_bindings'].append({'start': end, 'end': end+1,
                'original_text': '\n', 'atom_ids': [], 'separator': True})
        start += len(line)
    return canonical


def test_public_v6_exact_original_values_fragments_and_independent_composition(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unmocked external boundary')

    for name in ('requests.sessions.Session.request', 'httpx.Client.send', 'httpx.AsyncClient.send',
                 'subprocess.run', 'screening.checks.llm_json'):
        monkeypatch.setattr(name, forbidden)
    for name in ('screening.checks.resolve_llm_config', 'verification.literature.resolve_llm_config'):
        monkeypatch.setattr(name, lambda: LLMConfig('mock', 'mock', None, None))
    monkeypatch.setattr('common.config.get_settings', lambda: Settings(_env_file=None))
    modes = ('healthy', 'bad_fragment', 'bad_value', 'bad_value_type', 'bad_bool_type', 'bad_unicode_offset',
             'missing_path', 'duplicate_path', 'bad_composition', 'stale_v5')
    for mode in modes:
        case = tmp_path / mode
        materials, text = _materials(case, 75)
        block = next(b for b in materials.blocks if b.id == 'block_10')
        claim = Claim.model_validate({'id': 'claim_001', 'text': text, 'loc': block.loc,
            'source_block_id': block.id, 'source_quote': text,
            'source_refs': [{'source_block_id': 'block_11',
                'source_quote': next(b.text for b in materials.blocks if b.id == 'block_11'),
                'loc': next(b.loc for b in materials.blocks if b.id == 'block_11'), 'covered': ['c1']}],
            'conditions': [{'id': 'c1', 'dataset': 'MiniSet', 'metric': 'accuracy', 'settings': {
                'model': 'ExactMatch', 'accuracy': 0.75, 'examples': 100, 'correct_predictions': 75,
                'incorrect_predictions': 25,
                'accuracy_definition': 'fraction computed by exact equality of each prediction and label',
                'qualifiers': ['No repeated-run uncertainty or population-performance conclusion claimed',
                               'No ranking against other models'],
                'empty_list': [], 'empty_object': {}, 'empty_value': None},
                'description': 'ExactMatch measured test accuracy on MiniSet test.\U0001f9ea\u2003'}],
            'needs': ['Experiments'], 'importance': 'core'})
        original = claim.model_dump(mode='json')
        raw, calls, runs = [], [], []

        def model(**kwargs):
            calls.append(kwargs['module'])
            payload = json.loads(kwargs['prompt'])
            if kwargs['module'] == 'verification.experiments':
                assert payload['output_schema']['properties']['schema_version']['const'] == 'experiments-execution-v6'
                registry = payload['released_statistics_context']
                assert not registry['previous_registry']['candidates']
                assert registry['route_state']['enumeration_complete'] is True
                assert registry['budget']['checks_used'] > registry['previous_registry']['budget']['checks_used']
                candidate = next(iter(registry['candidates'].values()))
                assert 'numerator' not in candidate['recipe_facts']
                selection = _raw_selection(candidate, text)
                proposal = selection['proposal']
                if mode == 'bad_fragment':
                    proposal['claim_bindings'][0]['original_text'] += 'changed'
                elif mode == 'bad_value':
                    next(f for f in proposal['field_bindings'] if f['path'] == '/description')['original_value'] += 'changed'
                elif mode == 'bad_value_type':
                    next(f for f in proposal['field_bindings'] if f['path'] == '/settings/examples')['original_value'] = 100.0
                elif mode == 'bad_bool_type':
                    next(f for f in proposal['field_bindings'] if f['path'] == '/settings/examples')['original_value'] = True
                elif mode == 'bad_unicode_offset':
                    value = proposal['field_bindings'][-1]['original_value']
                    proposal['field_bindings'][-1]['fragments'][-1]['end'] = len(value.encode('utf-16-le')) // 2
                elif mode == 'missing_path':
                    proposal['field_bindings'].pop()
                elif mode == 'duplicate_path':
                    proposal['field_bindings'].append(copy.deepcopy(proposal['field_bindings'][0]))
                if mode == 'stale_v5':
                    selection = _selection(candidate, text)
                response = {'schema_version': 'experiments-execution-v5' if mode == 'stale_v5' else 'experiments-execution-v6',
                    'execution_domain': registry['domain'], 'execution_context_digest': registry['context_digest'],
                    'checked_aspects': ['correspondence', 'fairness', 'isolation', 'stability', 'consistency'], 'items': [],
                    'execution': {'kind': 'statistics_choice', 'selection': selection}}
            else:
                assert kwargs['module'] == 'verification.experiments.scope'
                context = payload['released_statistics_context']
                selection = context['selection']
                assert selection['proposal']['version'] == 'released-predictions-v3'
                assert all('value_digest' in f and 'original_value' not in f for f in selection['proposal']['field_bindings'])
                assert any(f['separator'] and f['text_digest'] == digest('\n') for f in selection['proposal']['claim_bindings'])
                assert any(o['kind'] == 'composition' and o['path'] == '/text' for o in context['obligations'])
                reviews = [{'obligation_id': o['obligation_id'], 'decision': 'confirmed',
                    'rationale': 'Independent original field, fragment and complete composition review.'} for o in context['obligations']]
                if mode == 'bad_composition':
                    target = next(i for i, o in enumerate(context['obligations']) if o['kind'] == 'composition' and o['path'] == '/text')
                    reviews[target]['decision'] = 'unresolved'
                condition = {'condition_id': 'c1', 'assertion': 'descriptive', 'matched_controls_required': False,
                    'uncertainty_sensitive': False, 'relation': 'eq', 'rationale': 'Finite fixed released-predictions assertion.'}
                response = authority_scope_fixture(payload, {'conditions': [condition], 'items': []},
                    requirement='unresolved', basis='unresolved', rationale='No independent paper support is asserted.')
                response.update(execution_scope_domain=context['domain'], released_statistics_reviews=[{
                    'version': 'statistics-review-v1', 'condition_id': selection['condition_id'],
                    'candidate_id': selection['candidate_id'], 'proposal_sha256': context['proposal_sha256'],
                    'context_digest': context['context_digest'], 'decision': 'confirmed', 'reviews': reviews,
                    'rationale': 'Independent complete original composition review.'}])
            raw.append(copy.deepcopy(response))
            return response

        async def literature(*args, **kwargs):
            return BranchResult()

        with run_scope(case / 'stats.json'):
            verified = asyncio.run(verify_claims([claim], materials, case / 'verification', call=model, global_literature=literature))
        assert claim.model_dump(mode='json') == original
        assert verified.claims[0].text == text and verified.claims[0].conditions == claim.conditions
        if mode != 'healthy':
            assert not verified.plans, mode
            assert assess_claim(verified.claims[0]).status.value == 'unverified'
            assert verified.issues
            assert len(calls) == (2 if mode == 'bad_composition' else 1)
            continue
        assert calls == ['verification.experiments', 'verification.experiments.scope']
        assert len(verified.plans) == 1, verified.issues
        binding = verified.plans[0].target_bindings['c1']
        retained = json.loads(Path(binding.projection.scope_audit).read_text(encoding='utf-8'))
        assert retained['first_pass_response'] == raw[0]
        assert retained['first_pass_canonical_selection'] == binding.projection.selection.model_dump(mode='json')
        assert retained['execution_wire']['canonical_selection_sha256'] == digest(retained['first_pass_canonical_selection'])
        assert retained['execution_wire']['raw_response_sha256'] == binding.projection.first_response_sha256 == digest(raw[0])
        assert retained['execution_wire']['normalized_response_sha256'] == binding.projection.normalized_response_sha256
        assert retained['execution_wire']['adapter_sha256'] == binding.projection.adapter_sha256
        assert '"value_digest":' not in json.dumps(raw[0]) and '"text_digest":' not in json.dumps(raw[0])

        def runner(request):
            runs.append(request)
            observed = Observation(dataset='MiniSet', metric='accuracy', settings={'split': 'test', 'model': 'ExactMatch'}, value=0.75)
            return RunOutcome(returncode=0, observations=[observed], stdout=json.dumps({'observations': [observed.model_dump(mode='json')]}))

        executed = execute_plans(verified.plans, verified.claims, materials, case / 'execution',
            config=ExecutionConfig(refine_with_llm=False, training_budget=0), runner=runner)
        assert len(runs) == 1, executed.issues
        assert len(executed.claims[0].evidence) == 1, executed.issues
        assert assess_claim(executed.claims[0]).status.value == 'supported'
        alignment = executed.ledger[0]['alignment'][0]
        measurement = json.loads(Path(alignment['measurement']).read_text(encoding='utf-8'))
        assert measurement['statistics'] == {'total': 100, 'correct': 75, 'incorrect': 25, 'accuracy': 0.75}
        assert measurement['all_records'] is True and measurement['training_performed'] is False
        assert alignment['released_statistics']['consistent'] is True
        assert alignment['tolerance'] == 0.02
        assert executed.claims[0].conditions == claim.conditions
