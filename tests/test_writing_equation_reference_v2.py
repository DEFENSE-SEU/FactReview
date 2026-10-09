import json

import pytest

from screening.checks import check_writing
from tests.test_writing_sections_v2 import candidate, confirmations, make_materials


@pytest.mark.parametrize(
    "reference,kind,valid",
    [
        ("Equation 1", "equation", True),
        ("Eq. (1)", "equation", True),
        ("Equation (1)", "equation", True),
        ("Eq. ( 1 )", "equation", True),
        ("Eq. (2)", "equation", False),
        ("Eq. (1", "equation", False),
        ("Eq. 1)", "equation", False),
        ("Eq. ((1))", "equation", False),
        ("Eq. (1))", "equation", False),
        ("Eq. (1.2)", "equation", False),
        ("Table (1)", "equation", False),
        ("Eq. (1)", "table", False),
    ],
)
def test_equation_labels_reach_original_page_confirmation_only_when_bound(tmp_path, reference, kind, valid):
    materials = make_materials(
        tmp_path,
        [
            ("Methods", f"See {reference} for the positive bound."),
            ("Results", r"Negative bound: x < 0. \tag{1}"),
        ],
    )
    materials.blocks[1].kind = "equation"
    records, validations = [], []

    def model(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing.validation":
            validations.append(payload)
            assert kwargs["images"] == [page.path for page in materials.pages]
            assert payload["additional_target_pages"] == [2]
            return confirmations(payload, "cross_reference_error")
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="cross_reference",
                    target_kind=kind,
                    target_label="1",
                    reference_problem="inconsistent_target",
                )
            ]
        }

    findings = check_writing(materials, call=model, records=records, recover_errors=True)
    assert len(findings) == int(valid)
    assert len(validations) == int(valid)
    assert records[0].status == ("checked" if valid else "failed")
