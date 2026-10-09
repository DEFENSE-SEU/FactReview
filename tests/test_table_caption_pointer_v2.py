"""Caption evidence points at the complete original pixels/text it cites."""

import fitz
import pytest

from tests.test_table_context_v2 import decision, paper, run
from tests.test_table_context_v2 import offline as offline


@pytest.mark.parametrize(
    "mode,above,wrap",
    [
        ("native", False, False),
        ("adjacent", False, False),
        ("adjacent", True, False),
        ("adjacent", False, True),
    ],
)
def test_caption_outside_crop_points_to_complete_original_pdf(tmp_path, mode, above, wrap):
    materials = paper(tmp_path, caption_mode=mode, above=above, wrap=wrap)
    before = materials.model_dump(mode="json")
    findings, issues, records, calls = run(
        materials,
        lambda r, p: decision(p, "manuscript_issue"),
        raw={
            "findings": [
                {
                    "category": "text_table_consistency",
                    "disposition": "issue",
                    "text": "Original contradiction candidate",
                }
            ]
        },
    )
    assert len(findings) == 1 and records[0].context_status == "checked", issues
    assert len(calls) == 2
    pointer = findings[0].evidence[0].pointer
    assert pointer.locator == materials.source_pdf
    assert pointer.page == 1 and pointer.key == materials.tables[0].id
    with fitz.open(materials.source_pdf) as pdf:
        assert pointer.quote in pdf[0].get_text()
    assert "M has six" in pointer.quote
    assert materials.tables[0].printed_crop_path in findings[0].evidence[0].note
    assert materials.model_dump(mode="json") == before
