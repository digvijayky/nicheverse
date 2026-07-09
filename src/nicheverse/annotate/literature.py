"""Literature grounding via NCBI PubMed E-utilities and the bioRxiv API.

Used to back marker-based cell-type calls with primary references. Network access
and ``requests`` are required (part of the ``nicheverse[llm]`` extra); every call
degrades to an empty result on failure so annotation never hard-crashes offline.
"""

from __future__ import annotations

__all__ = ["pubmed_search", "biorxiv_search", "literature_for_markers"]

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search(query: str, *, max_results: int = 4, api_key: str | None = None) -> list[dict]:
    """Return up to ``max_results`` PubMed hits ``{pmid, title, journal, year, first_author}``."""
    import requests

    try:
        ids = (
            requests.get(
                f"{_EUTILS}/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json",
                        **({"api_key": api_key} if api_key else {})},
                timeout=20,
            )
            .json()
            .get("esearchresult", {})
            .get("idlist", [])
        )
        if not ids:
            return []
        summ = (
            requests.get(
                f"{_EUTILS}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                timeout=20,
            )
            .json()
            .get("result", {})
        )
        out = []
        for i in ids:
            d = summ.get(i, {})
            out.append(
                {
                    "pmid": i,
                    "title": d.get("title", ""),
                    "journal": d.get("fulljournalname", "") or d.get("source", ""),
                    "year": (d.get("pubdate", "") or "")[:4],
                    "first_author": (d.get("authors") or [{}])[0].get("name", ""),
                }
            )
        return out
    except Exception:
        return []


def biorxiv_search(query: str, *, max_results: int = 3) -> list[dict]:
    """Placeholder: the public bioRxiv API has no keyword-search endpoint, so this returns [].

    Use :func:`pubmed_search` for literature grounding (PubMed also indexes many preprints).
    """
    return []


def literature_for_markers(
    markers: list[str], *, context: str = "cell type marker", max_results: int = 3, api_key: str | None = None
) -> dict[str, list[dict]]:
    """Map each marker gene to its top PubMed references for ``context``."""
    import time

    out: dict[str, list[dict]] = {}
    for i, g in enumerate(markers):
        if i:
            time.sleep(0.34)  # NCBI throttles anonymous callers at ~3 req/s
        out[g] = pubmed_search(f"{g} {context}", max_results=max_results, api_key=api_key)
    return out
