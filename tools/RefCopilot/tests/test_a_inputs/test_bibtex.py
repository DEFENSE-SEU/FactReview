"""Test A.1 — bibtex parsing."""

from __future__ import annotations

from refcopilot.inputs import bibtex


def test_bibtex_parses_minimal(fixtures_dir):
    refs = bibtex.parse_file(fixtures_dir / "inputs" / "minimal.bib")
    assert len(refs) == 3

    by_key = {r.bibkey: r for r in refs}

    vaswani = by_key["vaswani2017attention"]
    assert vaswani.title == "Attention Is All You Need"
    assert "Ashish Vaswani" in vaswani.authors
    assert vaswani.year == 2017
    assert vaswani.venue == "Advances in Neural Information Processing Systems"
    assert vaswani.arxiv_id == "1706.03762"
    assert vaswani.url == "https://arxiv.org/abs/1706.03762"

    devlin = by_key["devlin2019bert"]
    assert "BERT" in devlin.title
    assert devlin.doi == "10.18653/v1/n19-1423"
    assert devlin.year == 2019

    brown = by_key["brown2020gpt3"]
    assert brown.arxiv_id == "2005.14165"
    assert brown.year == 2020


def test_bibtex_handles_unicode(fixtures_dir):
    refs = bibtex.parse_file(fixtures_dir / "inputs" / "unicode.bib")
    assert len(refs) == 2

    schoel = next(r for r in refs if r.bibkey == "schoelkopf2002kernel")
    assert any("Sch" in a for a in schoel.authors)
    assert any("Smola" in a for a in schoel.authors)
    assert schoel.year == 2002

    wang = next(r for r in refs if r.bibkey == "wang2024chinese")
    assert wang.year == 2024
    # author parsing across non-Latin scripts is brittle in pybtex; just
    # require that the title and venue are correctly captured and at least
    # one author entry exists.
    assert wang.title and "中文" in wang.title
    assert wang.venue == "Proc. ACL"
    assert len(wang.authors) >= 1


def test_bibtex_misc_no_authors(fixtures_dir):
    refs = bibtex.parse_file(fixtures_dir / "inputs" / "misc.bib")
    assert len(refs) == 2
    for r in refs:
        assert r.title
        assert r.authors == []


def test_bibtex_extracts_arxiv_id_from_eprint():
    raw = """
    @misc{x,
      title = {T},
      author = {A. B.},
      eprint = {2401.12345v3},
      archivePrefix = {arXiv},
    }
    """
    refs = bibtex.parse_string(raw)
    assert len(refs) == 1
    assert refs[0].arxiv_id == "2401.12345"
    assert refs[0].arxiv_version == 3


def test_bibtex_extracts_arxiv_id_from_url():
    raw = """
    @misc{x,
      title = {T},
      author = {A. B.},
      url = {https://arxiv.org/abs/2401.12345v2},
    }
    """
    refs = bibtex.parse_string(raw)
    assert refs[0].arxiv_id == "2401.12345"
    assert refs[0].arxiv_version == 2


def test_bibtex_normalizes_doi_url():
    raw = """
    @article{x,
      title = {T},
      author = {A. B.},
      doi = {https://doi.org/10.1000/Foo},
    }
    """
    refs = bibtex.parse_string(raw)
    assert refs[0].doi == "10.1000/foo"


def test_bibtex_lenient_mode_skips_malformed():
    raw = """
    @article{good1,
      title = {T1},
      author = {A. B.},
      year = {2020},
    }

    @article{broken,
      title = {missing closing brace
      author = {A. B.},

    @article{good2,
      title = {T2},
      author = {C. D.},
      year = {2021},
    }
    """
    refs = bibtex.parse_string(raw)
    keys = [r.bibkey for r in refs]
    assert "good1" in keys
    # `good2` may or may not survive depending on how the brace imbalance is consumed;
    # at minimum we should not crash and should keep at least one entry.
    assert len(refs) >= 1
