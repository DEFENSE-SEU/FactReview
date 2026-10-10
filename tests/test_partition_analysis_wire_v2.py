"""Quote-only generation is local; all services and execution are mocked."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from common import run_stats
from fact_generation.execution import v2
from schemas.runtime_science import ROLES
from tests.test_partition_analysis_runtime_v2 import fixture, transport


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Quote wire controls forbid external services and processes')
    for name in ('requests.sessions.Session.request', 'httpx.Client.send',
                 'httpx.AsyncClient.send', 'subprocess.Popen', 'subprocess.run',
                 'llm.client.llm_json'):
        monkeypatch.setattr(name, forbidden)
    for name in ('docker_runner', 'run_command', 'docker_ensure_paper_image'):
        monkeypatch.setattr(v2, name, forbidden)


def candidate(original):
    result = copy.deepcopy(original)
    result['version'] = 'partition-runtime-candidate-v2'
    result['analysis']['version'] = 'released-partition-analysis-candidate-v2'
    for group in ('paper', 'sources'):
        for row in result['analysis'][group].values():
            row.pop('start')
            row.pop('end')
    return result


def test_candidate_real_refinement_preserves_legacy_and_original_judge(tmp_path, monkeypatch):
    plan, claim, materials, original = fixture(tmp_path)
    original['analysis'].update(rank_selector=None, config_settings_selector={})
    wire = candidate(original)
    before = claim.model_dump(mode='json'), plan.model_dump(mode='json')
    calls = []
    def llm(prompt, system, cfg, *, module):
        assert module == 'execution'
        payload = json.loads(prompt)
        calls.append(payload)
        run_stats.record_llm_call(module=module, provider='mock', model='mock',
            usage={'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14})
        if 'plan' in payload:
            assert payload['science_proposal_schema']['properties']['version']['const'] == 'partition-runtime-v1'
            schema = payload['science_proposal_candidate_schema']
            assert schema['properties']['version']['const'] == 'partition-runtime-candidate-v2'
            for name in ('CandidatePaperRoles', 'CandidateSourceRoles'):
                assert set(schema['$defs'][name]['required']) == set(ROLES)
                assert schema['$defs'][name]['additionalProperties'] is False
            assert 'output_mapping null' in system
            return {'command': list(plan.task.command), 'metric_output': None, 'output_mapping': None,
                'source_sites': [{'path': 'eval.py', 'qualname': 'driver', 'firstlineno': 2}],
                'science_proposal': copy.deepcopy(wire)}
        context = payload['context']
        return {'version': 'analysis-v1', 'context_digest': context['context_digest'],
            'condition_id': 'c1', 'measurement_scope': 'released_statistics',
            'obligations': [{'id': key, 'decision': 'confirmed', 'source_ids': values,
                            'rationale': 'Complete released statistics and located definitions agree.'}
                            for key, values in context['obligations'].items()], 'unresolved': []}
    monkeypatch.setattr('llm.client.llm_json', llm)
    monkeypatch.setattr('llm.client.resolve_llm_config', lambda: None)
    result = v2.execute_plans([plan], [claim], materials, tmp_path / 'out', config={'max_attempts': 0},
        runner=transport)
    assert len(calls) == 2
    entry = result.ledger[0]
    refinement = entry['refinement']
    assert refinement['science_proposal_wire'] == wire
    assert refinement['science_proposal'] == original
    assert refinement['science_proposal_localization']['status'] == 'localized'
    assert refinement['science_proposal_localization']['snapshot_timing'] == 'before_completion'
    assert entry['attempts'][0]['observations'] == []
    assert json.loads(entry['attempts'][0]['stdout']) == {'metrics': {'mse': 2/3}}
    assert entry['alignment'][0]['aligned'] and len(result.claims[0].evidence) == 1
    assert before == (claim.model_dump(mode='json'), plan.model_dump(mode='json'))


def test_exact_localization_snapshot_rejections_and_version_isolation(tmp_path, monkeypatch):
    from fact_generation.execution.partition_analysis_wire import freeze_sources, localize_candidate
    from fact_generation.execution.runtime_science import read_science_sources
    plan, claim, materials, original = fixture(tmp_path)  # noqa: RUF059
    original['analysis'].update(rank_selector=None, config_settings_selector={})
    root = Path(materials.repository.root)
    files, _ = read_science_sources(plan, materials, root)
    paper = [{'id': b.id, 'text': b.text, 'loc': b.loc.model_dump(mode='json')} for b in materials.blocks]
    def freeze():
        return freeze_sources(workspace=root, supplied_files=files, paper_sources=paper, materials=materials)
    snapshot = freeze()
    wire = candidate(original)
    def local(value, frozen=snapshot):
        return localize_candidate(value, snapshot=frozen, workspace=root,
            supplied_files=files, paper_sources=paper, materials=materials)
    assert local(wire)['canonical'] == original
    for modify in (lambda w: w['analysis']['paper'].pop('metric'),
                   lambda w: w['analysis']['sources']['model'].update(verified=True),
                   lambda w: w['analysis']['sources']['dataset'].update(path='../README.md'),
                   lambda w: w['analysis']['sources']['metric'].update(quote='absent'),
                   lambda w: w.update(raw_value_selector=[True])):
        bad = copy.deepcopy(wire)
        modify(bad)
        rejected = local(bad)
        assert rejected['status'] == 'unresolved'
        assert rejected['wire'] == bad and rejected['wire_digest'] == hashlib.sha256(
            json.dumps(bad, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert local(original)['canonical'] is None  # canonical-v1 never auto-repaired
    wrong_legacy = copy.deepcopy(original)
    wrong_legacy['analysis']['paper']['dataset']['start'] += 1
    for untouched in (wrong_legacy, {'version': 'unknown', 'analysis': {}}, None):
        record = local(untouched)
        assert record['status'] == 'unresolved' and record['canonical'] is None
        assert record['wire'] == untouched and record['wire_digest'] == hashlib.sha256(
            json.dumps(untouched, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    saved = (root / 'README.md').read_bytes()
    (root / 'README.md').write_bytes(saved + b' changed')
    assert local(wire)['status'] == 'unresolved'
    (root / 'README.md').write_bytes(saved)
    materials.blocks[0].text += ' changed'
    assert local(wire)['status'] == 'unresolved'
    materials.blocks[0].text = paper[0]['text']
    # Complete UTF-8 source with raw CRLF/CR and non-ASCII codepoints; no fuzzy normalization.
    from preprocessing.materials import index_repository
    raw = 'Ω first\r\nsecond\rthird\naaa'.encode('utf-8')  # noqa: UP012
    (root / 'README.md').write_bytes(raw)
    materials.repository = index_repository(root)
    files['README.md'] = raw.decode().replace('\r\n', '\n').replace('\r', '\n')
    mixed = copy.deepcopy(wire)
    for row in mixed['analysis']['sources'].values():
        row['quote'] = 'Ω first\nsecond\nthird'
    localized = local(mixed, freeze())
    assert localized['status'] == 'localized'
    for row in localized['canonical']['analysis']['sources'].values():
        assert raw.decode()[row['start']:row['end']] == row['quote'] == 'Ω first\r\nsecond\rthird'
    for row in mixed['analysis']['sources'].values():
        row['quote'] = 'aa'
    assert local(mixed, freeze())['status'] == 'unresolved'  # overlap: aaa has two aa
    (root / 'README.md').write_bytes(b'\xff')
    materials.repository = index_repository(root)
    assert freeze()['status'] == 'unresolved'
    (root / 'README.md').write_bytes(b'x' * 65537)
    materials.repository = index_repository(root)
    assert freeze()['status'] == 'unresolved'
    from fact_generation.execution import partition_analysis_wire as helper
    def blocked(path):
        raise ValueError('linked/reparse source')
    monkeypatch.setattr(helper, 'plain_path', blocked)
    assert freeze()['status'] == 'unresolved'
