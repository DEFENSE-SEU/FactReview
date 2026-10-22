"""Single isolated accounting control; scientific context and delegate are mocks."""
import json
from pathlib import Path
from types import SimpleNamespace

from common import run_stats
from fact_generation.execution import runtime_science


def test_invalid_inherited_stats_must_stop_before_semantic_delegate(tmp_path, monkeypatch):
    stats = tmp_path / 'invalid-existing-stats.json'
    stats.write_text('{malformed previous accounting', encoding='utf-8')
    original = stats.read_bytes()
    calls = []
    monkeypatch.setattr(run_stats, 'stats_path', lambda: stats)
    monkeypatch.setattr(runtime_science, 'build_consumption_context', lambda *args, **kwargs: {'context_digest': 'a'*64})

    def fake_semantic(context, proposal, kwargs):
        calls.append('delegate')
        run_stats.record_llm_call(module='execution', provider='offline-mock', model='offline-mock',
                                 usage={'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14})
        return {'status': 'qualified', 'scientific_qualification': True, 'alignment': False,
                'support': False, 'model_calls': 1, 'derived_observations': [{'value': 1}],
                'context': context}

    monkeypatch.setattr(runtime_science, '_review_context', fake_semantic)
    result = runtime_science.qualify_consumption({}, plan=None, claim=None, materials=None,
        request=SimpleNamespace(run_dir=str(tmp_path), repair_round=0), outcome=None, source_files={})
    outcome = {'calls': calls, 'status': result['status'], 'scientific_qualification': result['scientific_qualification'],
               'usage': result.get('usage'), 'prior_stats_overwritten': stats.read_bytes()!=original,
               'scope': 'isolated inherited-accounting only; fake semantic callback, no scientific/source claims',
               'source_sha256': __import__('hashlib').sha256(Path(runtime_science.__file__).read_bytes()).hexdigest()}
    (tmp_path/'inherited-stats-result.json').write_text(json.dumps(outcome, indent=2)+'\n', encoding='utf-8')
    assert calls == [], 'invalid inherited accounting must stop before any semantic provider admission'
    assert result['status']=='failed' and result['scientific_qualification'] is False
    assert stats.read_bytes()==original, 'unreadable previous usage must be preserved'
