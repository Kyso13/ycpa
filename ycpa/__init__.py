"""
YCPA-P — Self-Referential Symbolic Regression for Physical-Law Discovery
========================================================================
Discovers closed-form physical laws (such as T = a^(3/2)) from data.

Quick start
-----------
    import ycpa
    import pandas as pd

    df = pd.DataFrame({"a": [...], "T": [...]})
    result = ycpa.fit(df, target="T", features=["a"])
    print(result.formula)        # "T = 1 · a^(3/2)"
    print(result.exponents)      # [("a", "3/2", 1.5)]

If the C++ acceleration core (ycpa_core) is compiled it is used automatically;
otherwise it falls back seamlessly to pure Python. Check: ycpa.cpp_status()
"""
from __future__ import annotations

from . import _engine
from ._engine import (
    fit_power_law,
    scan_atom_space,
    tournament_mcmc,
    mdl_score,
    cpp_status,
)

__version__ = "0.1.0"

__all__ = ["fit", "discover", "list_atoms", "FitResult", "DiscoverResult",
           "fit_power_law", "cpp_status", "__version__"]

# Default atom pool for general symbolic regression (user-overridable)
DEFAULT_ATOMS = ["x", "x2", "x3", "sin", "cos", "exp_x", "ln_x", "sqrt_x", "inv_x"]


def list_atoms(pool_only: bool = False):
    """List available atom names for discover().

    Parameters
    ----------
    pool_only : bool, default False
        If True, return only the default pool used by discover();
        otherwise return every atom name the engine recognizes.

    Returns
    -------
    list[str]
        Atom names you can pass to discover(atoms=[...]).
    """
    if pool_only:
        return list(DEFAULT_ATOMS)
    try:
        return sorted(_engine.ATOM_LIBRARY.keys())
    except Exception:
        return list(DEFAULT_ATOMS)


def _validate_inputs(data, target, features, atoms=None):
    """Validate user inputs early with clear, actionable messages."""
    # pandas DataFrame?
    if not hasattr(data, "columns"):
        raise TypeError(
            "ycpa expects a pandas DataFrame as `data`, got "
            f"{type(data).__name__}. Example: ycpa.fit(df, target='y', features=['x'])."
        )
    cols = list(data.columns)
    if target not in cols:
        raise ValueError(
            f"target column '{target}' not found. "
            f"Available columns: {cols}"
        )
    missing = [f for f in features if f not in cols]
    if missing:
        raise ValueError(
            f"feature column(s) {missing} not found. "
            f"Available columns: {cols}"
        )
    if not features:
        raise ValueError(
            "no feature columns given (and none could be inferred). "
            "Pass features=['x', ...]."
        )
    if len(data) < 4:
        raise ValueError(
            f"need at least 4 data rows to fit a formula, got {len(data)}."
        )
    if atoms is not None:
        known = set(list_atoms())
        bad = [a for a in atoms if a not in known]
        if bad:
            raise ValueError(
                f"unknown atom(s) {bad}. Use ycpa.list_atoms() to see valid names. "
                f"Default pool: {DEFAULT_ATOMS}"
            )


class FitResult:
    """Readable wrapper around a fit() result."""

    def __init__(self, raw: dict, target: str):
        self._raw = raw or {}
        self.target = target
        self.exponents = self._raw.get("exponents", [])
        self.constant = self._raw.get("C", 1)
        self.constant_rational = self._raw.get("C_piH_simple") or self._raw.get("C_piH")
        self.model_type = self._raw.get("model_type")
        self.r2 = self._raw.get("R2") or self._raw.get("r2")
        self.mdl = self._raw.get("mdl") or self._raw.get("MDL")

    @property
    def formula(self) -> str:
        """Human-readable formula, e.g. 'T = 1 · a^(3/2)'."""
        if not self.exponents:
            return f"{self.target} = (no model found)"
        terms = " · ".join(f"{f}^({lab})" for f, lab, _ in self.exponents)
        c = self.constant_rational or self.constant
        return f"{self.target} = {c} · {terms}"

    def __repr__(self) -> str:
        bits = [f"FitResult({self.formula!r}"]
        if self.r2 is not None:
            bits.append(f"R²={self.r2:.6f}")
        if self.mdl is not None:
            bits.append(f"MDL={self.mdl:.4f}")
        return ", ".join(bits) + ")"

    def to_dict(self) -> dict:
        """Return the raw result dictionary (for advanced use)."""
        return dict(self._raw)


def fit(data, target, features=None, *, use_mcmc=True, **kwargs) -> FitResult:
    """Discover a power law Y = C · x₁^a · x₂^b · … from a data frame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table.
    target : str
        Name of the dependent column to explain.
    features : list[str], optional
        Names of the explanatory columns. If omitted, all columns except
        the target are used.
    use_mcmc : bool, default True
        Use the tournament-MCMC search (fast when C++ is available).
    **kwargs :
        Advanced options forwarded to fit_power_law (H, p5_hard_kill, …).

    Returns
    -------
    FitResult
        With .formula, .exponents, .constant, .r2, .mdl and .to_dict().
    """
    if features is None:
        features = [c for c in data.columns if c != target]
    _validate_inputs(data, target, features)
    raw = fit_power_law(data, target, features, use_mcmc=use_mcmc, **kwargs)
    return FitResult(raw, target)


class DiscoverResult:
    """Result of discover() — the discovered formula plus a confidence score.

    Important: confidence (0–1) indicates how much you can trust the formula.
      >=0.8 high   : the correct form was very likely found (clean data)
      0.5–0.8 med  : probably correct, but noise/uncertainty is present
      <0.5 low     : the form may be outside the atom pool — interpret WITH CARE
    Low confidence means the formula may be wrong despite a high R².
    """

    def __init__(self, raw: dict, target: str):
        self._raw = raw or {}
        self.target = target
        self.terms = self._raw.get("feature_atom_pairs", [])
        self.coefficients = self._raw.get("beta_rat", [])
        self.r2 = self._raw.get("R2") or self._raw.get("r2")
        self.mdl = self._raw.get("mdl")
        self.confidence = self._raw.get("confidence")
        self.confidence_label = self._raw.get("confidence_label")

    @property
    def formula(self) -> str:
        if not self.terms:
            return f"{self.target} = (no formula found)"
        # beta_rat layout: [intercept, atom1_coef, atom2_coef, ...] (if intercept present)
        has_intercept = len(self.coefficients) == len(self.terms) + 1
        offset = 1 if has_intercept else 0
        parts = []
        for i, (feat, atom) in enumerate(self.terms):
            idx = i + offset
            lab = self.coefficients[idx][0] if idx < len(self.coefficients) else "?"
            parts.append(f"{lab}·{atom}({feat})")
        expr = " + ".join(parts)
        if has_intercept:
            c0 = self.coefficients[0][0]
            if str(c0) not in ("0", "0.0"):
                expr += f" + {c0}"
        return f"{self.target} = {expr}"

    def __repr__(self) -> str:
        c = f"{self.confidence:.0%}" if self.confidence is not None else "?"
        r = f"{self.r2:.4f}" if self.r2 is not None else "?"
        return f"DiscoverResult({self.formula!r}, confidence={c}, R²={r})"

    def to_dict(self) -> dict:
        return dict(self._raw)


def discover(data, target, features=None, *, atoms=None, max_terms=2,
             use_mcmc=False, **kwargs) -> DiscoverResult:
    """General symbolic regression: discover a free-form formula from data.

    Difference from fit(): fit() searches for a power law (C·x^a); discover()
    finds additive formulas from an atom pool (sin, cos, exp, ln, polynomial, …),
    e.g. y = 3·sin(x) + 2·exp(z).

    Parameters
    ----------
    data : pandas.DataFrame
    target : str — the column to explain
    features : list[str], optional — explanatory columns (default: all but target)
    atoms : list[str], optional — candidate atom pool (default: DEFAULT_ATOMS)
    max_terms : int — maximum number of terms in the formula (default 2)

    Returns
    -------
    DiscoverResult — with .formula, .confidence (0–1), .r2, .terms

    Note: if confidence is low (<0.5) the form may be outside the atom pool and
    may be wrong despite a high R². Always take the confidence score into account.
    """
    if features is None:
        features = [c for c in data.columns if c != target]
    if atoms is None:
        atoms = DEFAULT_ATOMS
    _validate_inputs(data, target, features, atoms=atoms)
    results = scan_atom_space(data, target, features, atoms,
                              max_terms=max_terms, use_mcmc=use_mcmc,
                              verbose=False, fast=not use_mcmc, **kwargs)
    best = results[0] if results else {}
    return DiscoverResult(best, target)
