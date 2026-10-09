"""An already frozen appendix heading accompanies its selected proof body."""

import copy
import json

import pytest

from assessment import assess_claim
from tests.test_theory_visual_heading_v2 import heading_case
from tests.test_theory_visual_recheck_v2 import offline as offline
from tests.test_theory_visual_recheck_v2 import recovered
from verification.theory import verify_theory


@pytest.mark.parametrize("mode", ["healthy", "duplicate", "changed_before_visual"])
def test_selected_appendix_body_retains_only_unchanged_unambiguous_heading(tmp_path, mode):
    claim, materials, partial, heading = heading_case(
        tmp_path, "duplicate" if mode == "duplicate" else "healthy"
    )
    calls = []

    def model(**kwargs):
        calls.append(kwargs["module"])
        if kwargs["module"] == "verification.theory":
            if mode == "changed_before_visual":
                heading.text += " changed"
            return {
                "schema_version": "theory-derivation-v1",
                "items": [],
                "derivations": [],
                "appendix_block_ids": ["b2"],
            }
        if kwargs["module"] == "verification.theory.appendix":
            return copy.deepcopy(partial)
        assert kwargs["module"] == "verification.theory.visual_recheck"
        payload = json.loads(kwargs["prompt"])
        response = recovered(payload)
        response["visual_sources"][0]["printed_anchor"] = heading.text
        return response

    if mode in {"changed_before_visual", "duplicate"}:
        error = "changed" if mode == "changed_before_visual" else "unknown appendix block"
        with pytest.raises(ValueError, match=error):
            verify_theory(
                claim, materials, call=model, visual_recheck_rounds=1, output_dir=tmp_path / "audit"
            )
        assert calls == ["verification.theory"]
        return
    result = verify_theory(
        claim, materials, call=model, visual_recheck_rounds=1, output_dir=tmp_path / "audit"
    )
    assert calls == [
        "verification.theory",
        "verification.theory.appendix",
        "verification.theory.visual_recheck",
    ]
    assert result.theory_derivations[0].trace.outcome == "partial"
    assert result.theory_derivations[-1].state == ("validated" if mode == "healthy" else "invalid")
    claim.evidence.extend(result.evidence)
    assert assess_claim(claim).status == ("supported" if mode == "healthy" else "unverified")
