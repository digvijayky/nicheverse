"""Shared gene vocabulary across panels, platforms and species.

A foundation model trained on many imaging based spatial transcriptomics panels needs one
token space that every panel maps into. :class:`GeneVocabulary` builds that space as the
union of human gene symbols over all panels: mouse symbols are translated through a one to
one MGI homology table, mouse symbols with no one to one human ortholog are upper cased and
kept as their own entry, and control / blank / negative probes are dropped.

Nothing here changes existing behavior: the module is additive and is only used by the
multi panel dataset and the foundation model.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "CONTROL_PATTERN",
    "GeneVocabulary",
    "is_control_probe",
    "load_mgi_orthologs",
    "normalize_symbol",
]

#: Names that are not biological genes on any supported imaging platform.
CONTROL_PATTERN = re.compile(
    r"(?:blank|negprb|neg_?prb|negcontrol|neg_?control|systemcontrol|falsecode|"
    r"codeword|unassigned|intergenic|deprecated|antisense_|^neg[-_]|^control)",
    re.IGNORECASE,
)

_ENS_VERSION = re.compile(r"^(ENS[A-Z]*[GT]\d+)\.\d+$", re.IGNORECASE)


def normalize_symbol(symbol: str) -> str:
    """Upper case and strip a gene symbol, dropping an Ensembl version suffix.

    Only Ensembl style identifiers are de-versioned (``ENSG00000141510.5`` ->
    ``ENSG00000141510``); an ordinary symbol is left alone apart from case and
    surrounding whitespace, because a trailing ``.1`` can be part of a real name.
    """
    s = str(symbol).strip()
    m = _ENS_VERSION.match(s)
    if m:
        s = m.group(1)
    return s.upper()


def is_control_probe(symbol: str) -> bool:
    """True when a probe name is a control / blank / negative codeword rather than a gene."""
    return bool(CONTROL_PATTERN.search(str(symbol)))


def load_mgi_orthologs(path: str | Path) -> dict[str, str]:
    """Read an MGI homology report into a one to one mouse -> human symbol map.

    The report groups orthologs by ``DB Class Key``. Only classes holding exactly one
    mouse symbol and exactly one human symbol are kept, so a mouse gene is never mapped
    onto a human token it shares with a paralog. Symbols are normalized with
    :func:`normalize_symbol` and control probes are ignored.

    Parameters
    ----------
    path
        Path to the tab separated MGI ``HOM_MouseHumanSequence`` style report. It must
        have a header row containing at least ``DB Class Key``, ``Common Organism Name``
        and ``Symbol``.

    Returns
    -------
    dict
        ``{mouse_symbol_upper: human_symbol_upper}``.
    """
    path = Path(path)
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            k_i = header.index("DB Class Key")
            o_i = header.index("Common Organism Name")
            s_i = header.index("Symbol")
        except ValueError as exc:  # pragma: no cover - malformed report
            raise ValueError(f"{path} is not an MGI homology report: {exc}") from exc
        groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"mouse": [], "human": []})
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(k_i, o_i, s_i):
                continue
            org = parts[o_i].strip().lower()
            side = "mouse" if org.startswith("mouse") else ("human" if org.startswith("human") else None)
            if side is None:
                continue
            sym = normalize_symbol(parts[s_i])
            if not sym or is_control_probe(sym):
                continue
            groups[parts[k_i].strip()][side].append(sym)
    out: dict[str, str] = {}
    for g in groups.values():
        if len(g["mouse"]) == 1 and len(g["human"]) == 1:
            out[g["mouse"][0]] = g["human"][0]
    return out


class GeneVocabulary:
    """Union token space over many gene panels.

    Parameters
    ----------
    symbols
        The vocabulary, in order. Entry ``i`` is token ``i``.
    ortholog_map
        Mouse to human symbol map used when panels are mapped in (kept so that
        :meth:`map_panel` reproduces the mapping used at build time).

    Notes
    -----
    ``map_panel`` returns a per column token id, so two panel columns that collapse onto
    the same token (duplicated symbols, or two mouse genes with the same human ortholog)
    are handled by the caller, which sums their counts.
    """

    def __init__(self, symbols: Sequence[str], ortholog_map: Mapping[str, str] | None = None) -> None:
        self.symbols: tuple[str, ...] = tuple(symbols)
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("vocabulary symbols must be unique")
        self.index: dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self.ortholog_map: dict[str, str] = dict(ortholog_map or {})

    def __len__(self) -> int:
        return len(self.symbols)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"GeneVocabulary({len(self.symbols)} tokens)"

    # -- construction ---------------------------------------------------------------
    def translate(self, symbol: str, species: str) -> str | None:
        """Map one panel symbol to its vocabulary token, or None when it is a control probe."""
        s = normalize_symbol(symbol)
        if not s or is_control_probe(s):
            return None
        if str(species).strip().lower().startswith("mouse"):
            return self.ortholog_map.get(s, s)
        return s

    @classmethod
    def build(
        cls,
        panels: Mapping[str, Sequence[str]],
        species: Mapping[str, str],
        ortholog_map: Mapping[str, str] | None = None,
    ) -> GeneVocabulary:
        """Build the union vocabulary over ``panels``.

        Parameters
        ----------
        panels
            ``{dataset: gene symbols}``.
        species
            ``{dataset: species}``; anything starting with ``"mouse"`` (case insensitive)
            is translated through ``ortholog_map``.
        ortholog_map
            One to one mouse -> human symbol map, e.g. from :func:`load_mgi_orthologs`.

        Returns
        -------
        GeneVocabulary
            Tokens sorted alphabetically so the vocabulary is deterministic.
        """
        helper = cls((), ortholog_map)
        tokens: set[str] = set()
        for ds, genes in panels.items():
            sp = species.get(ds, "human")
            for g in genes:
                t = helper.translate(g, sp)
                if t:
                    tokens.add(t)
        return cls(sorted(tokens), ortholog_map)

    # -- use ------------------------------------------------------------------------
    def map_panel(self, genes: Sequence[str], species: str) -> tuple[np.ndarray, np.ndarray]:
        """Map a panel onto the vocabulary.

        Returns
        -------
        col_to_token : np.ndarray of int32, shape ``(len(genes),)``
            Token id per panel column, ``-1`` for dropped columns (control probes, or a
            symbol absent from the vocabulary).
        measured : np.ndarray of int32
            Sorted unique token ids this panel measures. Every loss for this dataset is
            restricted to these tokens.
        """
        col = np.full(len(genes), -1, dtype=np.int32)
        for i, g in enumerate(genes):
            t = self.translate(g, species)
            if t is not None:
                j = self.index.get(t)
                if j is not None:
                    col[i] = j
        measured = np.unique(col[col >= 0]).astype(np.int32)
        return col, measured

    # -- io ---------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the vocabulary (and its ortholog map) as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"symbols": list(self.symbols), "ortholog_map": self.ortholog_map}))
        return path

    @classmethod
    def load(cls, path: str | Path) -> GeneVocabulary:
        """Read a vocabulary written by :meth:`save`."""
        d = json.loads(Path(path).read_text())
        return cls(d["symbols"], d.get("ortholog_map", {}))
