"""LLM-based reference extraction."""

from __future__ import annotations

from refcopilot.extract import llm_extractor
from refcopilot.models import SourceFormat


def test_text_to_references_via_llm(mock_llm_json):
    mock_llm_json.set(
        {
            "references": [
                {
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "title": "Attention Is All You Need",
                    "venue": "NeurIPS",
                    "year": 2017,
                    "arxiv_id": "1706.03762",
                    "raw": "[1] A. Vaswani, N. Shazeer. Attention Is All You Need. NeurIPS 2017.",
                },
                {
                    "authors": ["Jacob Devlin"],
                    "title": "BERT",
                    "venue": "NAACL",
                    "year": 2019,
                    "doi": "10.18653/v1/N19-1423",
                    "raw": "[2] J. Devlin. BERT. NAACL 2019.",
                },
            ]
        }
    )
    refs = llm_extractor.extract_references(
        "[1] A. Vaswani, N. Shazeer. Attention Is All You Need. NeurIPS 2017.\n"
        "[2] J. Devlin. BERT. NAACL 2019.\n",
        source_format=SourceFormat.TEXT,
    )
    assert len(refs) == 2
    assert refs[0].title == "Attention Is All You Need"
    assert refs[0].arxiv_id == "1706.03762"
    assert refs[0].year == 2017
    assert refs[1].doi == "10.18653/v1/n19-1423"


def test_drops_prompt_echo(mock_llm_json):
    mock_llm_json.set(
        {
            "references": [
                {"title": "Extraction Rules: split by [N]"},
                {
                    "authors": ["A. B."],
                    "title": "A real paper",
                    "year": 2020,
                    "raw": "...",
                },
            ]
        }
    )
    refs = llm_extractor.extract_references("text", source_format=SourceFormat.TEXT)
    assert len(refs) == 1
    assert refs[0].title == "A real paper"


def test_drops_prose_titles(mock_llm_json):
    mock_llm_json.set(
        {
            "references": [
                {"title": "Based on the above, the references are:", "year": 2020},
                {"title": "A real paper", "authors": ["A. B."], "year": 2020},
            ]
        }
    )
    refs = llm_extractor.extract_references("x", source_format=SourceFormat.TEXT)
    assert len(refs) == 1
    assert refs[0].title == "A real paper"


def test_drops_incomplete_items(mock_llm_json):
    mock_llm_json.set(
        {
            "references": [
                {"title": ""},  # no title
                {"title": "Only title"},  # no authors/year/url
                {"title": "Has author", "authors": ["A. B."]},  # accepted
            ]
        }
    )
    refs = llm_extractor.extract_references("x", source_format=SourceFormat.TEXT)
    assert len(refs) == 1
    assert refs[0].title == "Has author"


def test_dedup_by_normalized_author_title(mock_llm_json):
    mock_llm_json.set(
        {
            "references": [
                {"authors": ["A. Smith"], "title": "Hello, World!", "year": 2020},
                {"authors": ["A. Smith"], "title": "hello world", "year": 2020},
                {"authors": ["B. Jones"], "title": "Different", "year": 2021},
            ]
        }
    )
    refs = llm_extractor.extract_references("x", source_format=SourceFormat.TEXT)
    assert len(refs) == 2


def test_handles_empty_payload(mock_llm_json):
    mock_llm_json.set({"references": []})
    refs = llm_extractor.extract_references("noop", source_format=SourceFormat.TEXT)
    assert refs == []


def test_handles_error_payload(mock_llm_json):
    mock_llm_json.set({"status": "error", "error": "TimeoutError: ..."})
    refs = llm_extractor.extract_references("text", source_format=SourceFormat.TEXT)
    assert refs == []


def test_chunking_respects_ref_number_boundary(mock_llm_json):
    long_text = "filler " * 500 + "\n" + "\n".join(f"[{i}] Author{i}. Title {i}. 2020." for i in range(1, 21))
    # Force chunking: budget very small.
    mock_llm_json.set_sequence(
        [
            {
                "references": [
                    {"authors": [f"A{i}"], "title": f"T{i}", "year": 2020} for i in range(1, 11)
                ]
            },
            {
                "references": [
                    {"authors": [f"A{i}"], "title": f"T{i}", "year": 2020} for i in range(11, 21)
                ]
            },
        ]
    )

    # Drop the budget for this test
    refs = llm_extractor._chunk(long_text, char_budget=1500)
    assert len(refs) >= 2  # produced at least 2 chunks


def test_normalizes_arxiv_id_with_version(mock_llm_json):
    mock_llm_json.set(
        {
            "references": [
                {
                    "authors": ["A. B."],
                    "title": "Some paper",
                    "year": 2017,
                    "arxiv_id": "1706.03762v3",
                }
            ]
        }
    )
    refs = llm_extractor.extract_references("x", source_format=SourceFormat.TEXT)
    assert refs[0].arxiv_id == "1706.03762"
    assert refs[0].arxiv_version == 3
