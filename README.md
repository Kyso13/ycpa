# YCPA-P

**Self-referential symbolic regression for physical-law discovery.**

YCPA-P discovers closed-form laws from data — power laws like Kepler's
`T = a^(3/2)`, and general additive forms like `y = 3·sin(x) + 2·exp(z)`.
It uses a Minimum-Description-Length (MDL) criterion, rational-coefficient
locking (ΠH: `1.4999 → 3/2`), and a tournament-MCMC search. The
performance-critical core is accelerated in C++ (Eigen + pybind11), with a
pure-Python fallback when the extension is not built.

> Developed independently by a first-year undergraduate. It is **not** a
> replacement for mature tools like PySR or AI Feynman — it is a fast,
> easy-to-use tool for a specific niche (see *What it is good at*).

## Installation

```bash
pip install .
```

For maximum performance (stable Eigen + AVX-512 hardware):

```bash
YCPA_NATIVE=1 pip install .
```

If C++ cannot be compiled, the package still installs and runs in pure-Python
fallback mode.

## Quick start

### Power laws — `fit()`

```python
import ycpa
import pandas as pd

# Kepler planetary data
df = pd.DataFrame({
    "a": [0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07],
    "T": [0.2408, 0.6152, 1.0, 1.8809, 11.8618, 29.4567, 84.0107, 164.786],
})

result = ycpa.fit(df, target="T", features=["a"])
print(result.formula)      # T = 1 · a^(3/2)
print(result.exponents)    # [("a", "3/2", 1.5)]
```

### General formulas — `discover()`

```python
import numpy as np, pandas as pd, ycpa

x = np.linspace(0.1, 6, 80)
df = pd.DataFrame({"x": x, "y": 3*np.sin(x) + 2})

r = ycpa.discover(df, "y")
print(r.formula)       # y = 3·sin(x) + 2
print(r.confidence)    # 1.0   (0–1, how much to trust the result)
```

## The confidence score (important)

`discover()` attaches a **confidence score** (0–1) to every result, similar in
spirit to scikit-learn's `predict_proba`. A high R² alone can be misleading:
a form outside the atom pool may still fit with R² ≈ 0.90 while being wrong.
The confidence score catches this.

| Confidence | Meaning |
|-----------|---------|
| ≥ 0.8 (high)   | Correct form very likely found (clean data) |
| 0.5–0.8 (med)  | Probably correct, but noise/uncertainty present |
| < 0.5 (low)    | Form may be outside the atom pool — interpret with care |

**Low confidence means the formula may be wrong despite a high R².** Always
read the confidence score, not just R².

## What it is good at

- Single/double-term forms from the atom pool (sin, cos, exp, ln, polynomials)
- Clean data, robust up to ~5% noise
- **Speed:** results in milliseconds, where heavier search tools take minutes
- **Rational coefficients:** recovers exact fractions (`3/2`, `8/3`) — a genuine
  niche advantage over decimal-only tools

## What it is not good at (be honest)

- **Out-of-pool forms** (e.g. `tanh` when not in the atom list): it will pick
  the nearest wrong atom — but the **confidence score drops**, signalling this.
- **High noise** (>15%): may select the wrong form.
- **Composite forms** (`sin(x)/x`) and **3+ term mixtures**: often incomplete.

This is a **fast first-look tool for small, clean, simple data** — think of it
as "the scikit-learn of symbolic regression": not the strongest, but fast,
easy, and honest about its uncertainty.

## Available atoms

`discover()` searches a pool of atoms. To see valid names:

```python
import ycpa
ycpa.list_atoms(pool_only=True)   # default pool used by discover()
ycpa.list_atoms()                 # every atom the engine recognizes
```

You can pass your own pool: `ycpa.discover(df, "y", atoms=["sin", "exp_x", "x2"])`.

## Clear error messages

Invalid input raises a readable error instead of a cryptic stack trace:

```python
ycpa.fit(df, target="WRONG", features=["a"])
# ValueError: target column 'WRONG' not found. Available columns: ['a', 'T']
```

## C++ acceleration status

```python
import ycpa
print(ycpa.cpp_status())   # {"cpp_active": True/False, ...}
```

`cpp_active: True` means the fast C++ core is in use. To disable it and force
pure Python: set the environment variable `YCPA_NO_CPP=1`.

## Tests

```bash
pip install ".[dev]"
pytest
```

## License

MIT
