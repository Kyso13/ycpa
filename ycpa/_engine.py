"""
YCPA-P v8.5 — Sembolik Regresyon & Fiziksel Yasa Keşfi   [TAM VERSİYON]
Yusuf Said Osmanoglu · Nisan 2026

Protokoller (asal sayı hariç tümü eksiksiz):
  v2  : MDL + OLS + ΠH (Q_H rasyonel arama)
  v3  : Q_H^alg — φ, ψ, 1/√5, √2, √3, π, e cebirsel uzay genişletme
  v4  : MDL* (r^k̂·Δk·ln r kapalı form) + Sigmoid hibrit geçiş
  v5  : Turnuva MCMC (25 zincir→5 elite→2000 derin) + Gibbs adaptif ısınma
  v6.1: SI birim vektörü Hard-Kill (P5) + türev sürekliliği (P15) + eksen simetrisi (Cax)
  v7  : SINDy + SVD prewhitening + STLS + MDL eşik — zaman serisi dt düzeltmeli
  v7.1: SAIM-Y adaptif alan metrigi (TRI pilot→rough/smooth) + LE 11/24
  v8.x: Asimptotik kilitler y* (α=1) ve y** (α=1,β=1)
  v8.5: Kapalı form zinciri erf→κ(p)→LE→MDL*  + 29/48·H^{-7/4}

Atom havuzu (74 atom, 8 grup):
  Temel güç, Log/Üstel, Trigonometrik, Hiperbolik,
  Cebirsel sabitler, Kompozit, Çapraz etkileşim, SINDy/ODE

Kullanım:
    pip install numpy pandas scipy scikit-learn openpyxl
    python ycpa_p.py                                    # Kepler demo
    python ycpa_p.py --data d.csv                       # interaktif
    python ycpa_p.py --data d.csv --target Y --features x1 x2 --model power
    python ycpa_p.py --data d.csv --target Y --features x1 x2 --model atom --atoms x ln_x sin
    python ycpa_p.py --data d.csv --target Y --features x1 x2 --model sindy
    python ycpa_p.py --data d.csv --target Y --features x1 x2 --model both --mcmc
    python ycpa_p.py --data d.csv --target Y --features x1 x2 --units Y=N x1=m --bootstrap --cv
    python ycpa_p.py --data d.csv --multi_target Fy Fx --features fz ap ae --units Fy=N Fx=N
"""

from __future__ import annotations

import argparse, hashlib, itertools, math, os, re, warnings
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import erf as scipy_erf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# OPSİYONEL C++ HIZLANDIRMA KÖPRÜSÜ
# ycpa_core (pybind11/Eigen) derlenmiş ve import edilebilirse, performans-kritik
# fonksiyonlar (tournament_mcmc, ND penalty matrisleri) C++ çekirdeğine delege
# edilir. Edilemezse saf-Python uygulamaya sorunsuz düşülür (graceful fallback).
# Davranış aynıdır; yalnızca hız değişir. Bunu kapatmak için: YCPA_NO_CPP=1
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_USE_CPP = False
_CPP = None
if _os.environ.get("YCPA_NO_CPP") != "1":
    try:
        try:
            from . import ycpa_core as _CPP
        except Exception:
            import ycpa_core as _CPP
        _USE_CPP = True
    except Exception:
        _CPP = None
        _USE_CPP = False

def cpp_status():
    """Return whether C++ acceleration is active (for diagnostics)."""
    return {"cpp_active": _USE_CPP,
            "module": getattr(_CPP, "__name__", None) if _CPP else None}

warnings.filterwarnings("ignore")

# Büyük veri — bellek / UI koruması
MAX_LOAD_ROWS = 120_000
MAX_ANALYSIS_ROWS = 20_000


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — ATOM KÜTÜPHANESİ  (74 atom, 8 grup)
# ═══════════════════════════════════════════════════════════════════════════════
# Her atom: (görünen_isim, tek_değişkenli_fonksiyon, latex)
# Çapraz atomlar iki sütun alır; build_feature_matrix içinde çiftler halinde işlenir.

ATOM_LIBRARY = {
    # ── Temel güç ─────────────────────────────────────────────────────────────
    "x":          ("x",           lambda x: x,                                   "x"),
    "x2":         ("x²",          lambda x: x**2,                                "x^2"),
    "x3":         ("x³",          lambda x: x**3,                                "x^3"),
    "x4":         ("x⁴",          lambda x: x**4,                                "x^4"),
    "sqrt_x":     ("√|x|",        lambda x: np.sqrt(np.abs(x)),                  r"\sqrt{|x|}"),
    "cbrt_x":     ("∛x",          lambda x: np.cbrt(x),                          "x^{1/3}"),
    "x_3_2":      ("x^(3/2)",     lambda x: np.abs(x)**1.5*np.sign(x),           "x^{3/2}"),
    "x_1_4":      ("x^(1/4)",     lambda x: np.abs(x)**0.25,                     "x^{1/4}"),
    "inv_x":      ("1/x",         lambda x: 1.0/(x+1e-300),                      "1/x"),
    "inv_x2":     ("1/x²",        lambda x: 1.0/(x**2+1e-300),                   "1/x^2"),
    "inv_sqrt_x": ("1/√|x|",      lambda x: 1.0/(np.sqrt(np.abs(x))+1e-300),    r"1/\sqrt{|x|}"),
    "x_2_3":      ("x^(2/3)",     lambda x: np.cbrt(x**2),                        "x^{2/3}"),
    "inv_x3":     ("1/x³",        lambda x: 1.0/(x**3+1e-300),                    "1/x^3"),

    # ── Log / Üstel ───────────────────────────────────────────────────────────
    "ln_x":       ("ln|x|",       lambda x: np.log(np.abs(x)+1e-300),            r"\ln|x|"),
    "log2_x":     ("log₂|x|",     lambda x: np.log2(np.abs(x)+1e-300),           r"\log_2|x|"),
    "log10_x":    ("log₁₀|x|",    lambda x: np.log10(np.abs(x)+1e-300),          r"\log_{10}|x|"),
    "exp_x":      ("eˣ",          lambda x: np.exp(np.clip(x,-50,50)),           "e^x"),
    "exp_neg":    ("e^{-x}",      lambda x: np.exp(np.clip(-x,-50,50)),          "e^{-x}"),
    "exp_x2":     ("e^{x²}",      lambda x: np.exp(np.clip(x**2,0,50)),          "e^{x^2}"),
    "ln1px":      ("ln(1+|x|)",   lambda x: np.log1p(np.abs(x)),                 r"\ln(1+|x|)"),
    "gauss_x":    ("e^{-x²}",     lambda x: np.exp(-np.clip(x**2, 0, 500)),       r"e^{-x^2}"),
    "pow2_x":     ("2^x",         lambda x: np.exp(np.clip(x*math.log(2),-500,500)), "2^x"),

    # ── Trigonometrik ─────────────────────────────────────────────────────────
    "sin":        ("sin(x)",      lambda x: np.sin(x),                           r"\sin x"),
    "cos":        ("cos(x)",      lambda x: np.cos(x),                           r"\cos x"),
    "tan":        ("tan(x)",      lambda x: np.tan(np.clip(x,-1.5,1.5)),         r"\tan x"),
    "arcsin":     ("arcsin(x)",   lambda x: np.arcsin(np.clip(x,-1,1)),          r"\arcsin x"),
    "arccos":     ("arccos(x)",   lambda x: np.arccos(np.clip(x,-1,1)),          r"\arccos x"),
    "arctan":     ("arctan(x)",   lambda x: np.arctan(x),                        r"\arctan x"),
    "sin2x":      ("sin(2x)",     lambda x: np.sin(2*x),                          r"\sin 2x"),
    "cos2x":      ("cos(2x)",     lambda x: np.cos(2*x),                          r"\cos 2x"),
    "sinc_x":     ("sinc(x)",     lambda x: np.sinc(x/np.pi),                     r"\text{sinc}(x)"),

    # ── Hiperbolik ────────────────────────────────────────────────────────────
    "sinh":       ("sinh(x)",     lambda x: np.sinh(np.clip(x,-20,20)),          r"\sinh x"),
    "cosh":       ("cosh(x)",     lambda x: np.cosh(np.clip(x,-20,20)),          r"\cosh x"),
    "tanh":       ("tanh(x)",     lambda x: np.tanh(x),                          r"\tanh x"),

    # ── Cebirsel sabitler (v3) ─────────────────────────────────────────────────
    # φ=(1+√5)/2, ψ=(1-√5)/2, 1/√5  ← Fibonacci/Binet
    "phi_x":      ("φ·x",         lambda x: x*1.6180339887,                      r"\varphi x"),
    "psi_x":      ("ψ·x",         lambda x: x*(-0.6180339887),                   r"\psi x"),
    "inv_sqrt5":  ("x/√5",        lambda x: x/2.2360679775,                      r"x/\sqrt{5}"),
    "pi_x":       ("π·x",         lambda x: x*math.pi,                           r"\pi x"),
    "e_x":        ("e·x",         lambda x: x*math.e,                            r"e\cdot x"),
    "sqrt2_x":    ("√2·x",        lambda x: x*math.sqrt(2),                      r"\sqrt{2}\,x"),
    "sqrt3_x":    ("√3·x",        lambda x: x*math.sqrt(3),                      r"\sqrt{3}\,x"),

    # ── Kompozit ─────────────────────────────────────────────────────────────
    "x_ln_x":     ("x·ln|x|",     lambda x: x*np.log(np.abs(x)+1e-300),          r"x\ln|x|"),
    "x_sin":      ("x·sin(x)",    lambda x: x*np.sin(x),                          r"x\sin x"),
    "x_cos":      ("x·cos(x)",    lambda x: x*np.cos(x),                          r"x\cos x"),
    "x_exp":      ("x·eˣ",        lambda x: x*np.exp(np.clip(x,-50,50)),          r"xe^x"),
    "sin2":       ("sin²(x)",     lambda x: np.sin(x)**2,                         r"\sin^2 x"),
    "cos2":       ("cos²(x)",     lambda x: np.cos(x)**2,                         r"\cos^2 x"),
    "sigmoid":    ("σ(x)",        lambda x: 1/(1+np.exp(np.clip(-x,-50,50))),    r"\sigma(x)"),
    "abs_x":      ("|x|",         lambda x: np.abs(x),                           r"|x|"),
    "sign_x":     ("sgn(x)",      lambda x: np.sign(x),                          r"\text{sgn}(x)"),
    "erf_x":      ("erf(x)",      lambda x: scipy_erf(x),                        r"\text{erf}(x)"),
    "x2_ln_x":    ("x²·ln|x|",   lambda x: x**2 * np.log(np.abs(x)+1e-300),      r"x^2\ln|x|"),
    "inv_x_ln_x": ("ln|x|/x",    lambda x: np.log(np.abs(x)+1e-300)/(x+1e-300),  r"\frac{\ln|x|}{x}"),
    "x_arctan_x": ("x·arctan(x)", lambda x: x * np.arctan(x),                     r"x\arctan x"),

    # ── SINDy (v7) — dt ile düzeltilmiş türev ─────────────────────────────────
    # Fonksiyon np.gradient kullanır; dt sütunu varsa _sindy_dt global ile düzeltilir
    "x_dx":       ("x·ẋ",         lambda x: x*np.gradient(x),                   r"x\dot{x}"),
    "x2_dx":      ("x²·ẋ",        lambda x: x**2*np.gradient(x),                r"x^2\dot{x}"),
    "dx":         ("ẋ",           lambda x: np.gradient(x),                      r"\dot{x}"),
}

# Çapraz etkileşim atomları — iki sütun gerektirir; ayrı dict
# Fonksiyon imzası: fn(col_a_values, col_b_values) -> array
CROSS_ATOM_LIBRARY = {
    "x_times_y":  ("x·y",         lambda a,b: a*b,                              "xy"),
    "x_over_y":   ("x/y",         lambda a,b: a/(b+1e-300),                     "x/y"),
    "y_over_x":   ("y/x",         lambda a,b: b/(a+1e-300),                     "y/x"),
    "x2_times_y": ("x²·y",        lambda a,b: a**2*b,                           "x^2 y"),
    "x_times_y2": ("x·y²",        lambda a,b: a*b**2,                           "xy^2"),
    "sqrt_xy":    ("√|xy|",        lambda a,b: np.sqrt(np.abs(a*b)),             r"\sqrt{|xy|}"),
    "x_plus_y":   ("x+y",         lambda a,b: a+b,                              "x+y"),
    "x_minus_y":  ("x-y",         lambda a,b: a-b,                              "x-y"),
    "max_xy":     ("max(x,y)",     lambda a,b: np.maximum(a,b),                  r"\max(x,y)"),
    "min_xy":     ("min(x,y)",     lambda a,b: np.minimum(a,b),                  r"\min(x,y)"),
    "x_minus_y_sq": ("(x-y)²",    lambda a,b: (a-b)**2,                           "(x-y)^2"),
    "x_plus_y_sq":  ("(x+y)²",    lambda a,b: (a+b)**2,                           "(x+y)^2"),
    "sqrt_x2_y2":   ("√(x²+y²)",  lambda a,b: np.sqrt(a**2+b**2),                r"\sqrt{x^2+y^2}"),
    "x2_plus_y2":   ("x²+y²",     lambda a,b: a**2+b**2,                          "x^2+y^2"),
    "x_ln_y":       ("x·ln|y|",   lambda a,b: a*np.log(np.abs(b)+1e-300),         r"x\ln|y|"),
    "x_over_x_plus_y": ("x/(x+y)", lambda a,b: a/(a+b+1e-300),                   r"\frac{x}{x+y}"),
    "x_minus_y_over_x_plus_y": ("(x-y)/(x+y)", lambda a,b: (a-b)/(a+b+1e-300),  r"\frac{x-y}{x+y}"),
    "hypot_norm":   ("x/√(x²+y²)", lambda a,b: a/(np.sqrt(a**2+b**2)+1e-300),    r"\frac{x}{\sqrt{x^2+y^2}}"),
}

ATOM_GROUPS = {
    "Temel güç":           ["x","x2","x3","x4","sqrt_x","cbrt_x","x_3_2","x_1_4",
                            "inv_x","inv_x2","inv_sqrt_x",
                            "x_2_3","inv_x3"],
    "Log / Üstel":         ["ln_x","log2_x","log10_x","exp_x","exp_neg","exp_x2","ln1px",
                            "gauss_x","pow2_x"],
    "Trigonometrik":       ["sin","cos","tan","arcsin","arccos","arctan",
                            "sin2x","cos2x","sinc_x"],
    "Hiperbolik":          ["sinh","cosh","tanh"],
    "Cebirsel sabitler":   ["phi_x","psi_x","inv_sqrt5","pi_x","e_x","sqrt2_x","sqrt3_x"],
    "Kompozit":            ["x_ln_x","x_sin","x_cos","x_exp","sin2","cos2",
                            "sigmoid","abs_x","sign_x","erf_x",
                            "x2_ln_x","inv_x_ln_x","x_arctan_x"],
    "Çapraz etkileşim":    list(CROSS_ATOM_LIBRARY.keys()),
    "SINDy (ODE)":         ["x_dx","x2_dx","dx"],
}

# AtomPicker'da varsayılan olarak kapalı gelen gruplar
OPTIONAL_ATOM_GROUPS: set[str] = {"Cebirsel sabitler"}


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — ΠH PROJEKSİYONU  (v2 + v3)
# Q_H^alg = Q_H ∪ {p/q·α : α ∈ {φ,ψ,1/√5,√2,√3,π,e}}
# ═══════════════════════════════════════════════════════════════════════════════

_ALG_CONSTS = {
    "φ":    1.6180339887,
    "ψ":   -0.6180339887,
    "1/√5": 0.4472135955,
    "√2":   1.4142135624,
    "√3":   1.7320508076,
    "π":    3.1415926536,
    "e":    2.7182818285,
}

def pi_H(c: float, H: int = 20, algebraic: bool = True) -> tuple:
    """
    Float → (etiket, değer, hata)
    Q_H^alg arama: saf rasyonel + cebirsel çarpanlı
    Ampirik üst sınır: 29/48·H^{-7/4}
    """
    H = max(int(H), 1)
    best_lbl, best_val, best_err = "0", 0.0, float("inf")
    for q in range(1, H+1):
        p = round(c*q)
        for dp in (-1,0,1):
            val = (p+dp)/q
            err = abs(c-val)
            if err < best_err:
                best_err = err; best_val = val
                best_lbl = str(Fraction(p+dp,q))
    if algebraic:
        for sym,alpha in _ALG_CONSTS.items():
            if abs(alpha) < 1e-12: continue
            for q in range(1, H+1):
                p = round(c/alpha*q)
                for dp in (-1,0,1):
                    val = ((p+dp)/q)*alpha
                    err = abs(c-val)
                    if err < best_err:
                        best_err = err; best_val = val
                        best_lbl = f"{Fraction(p+dp,q)}·{sym}"
    return best_lbl, best_val, best_err

def pi_H_error_bound(H: int) -> float:
    """29/48·H^{-7/4}  — v8.5 empirical upper bound"""
    return (29/48)*H**(-7/4)


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — TEMEL İSTATİSTİK VE MDL  (v2)
# ═══════════════════════════════════════════════════════════════════════════════

def ols(X, y):
    b,_,_,_ = np.linalg.lstsq(X, y, rcond=None); return b

def _med(a):
    s = np.sort(np.ravel(a)); n = len(s)
    return float(s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2)

def mad(y):
    m = _med(y); return float(_med(np.abs(y-m)))

def compute_SR(yp, yt, eps):
    n = len(yt)
    return max(float(np.sum(np.abs(yp-yt)<eps))/n, 1/(2*n))

def sr_closed_form(yp, yt):
    """E[SR] = erf(ε/σ√2)  — v8.5 closed form"""
    m = mad(yt); eps = 0.4*(m if m>0 else 1.0)
    sig = float(np.std(yt-yp)) or 1.0
    return float(scipy_erf(eps/(sig*math.sqrt(2))))

def mdl_score(yp, yt, k, H=20, *,
              beta=None, R1=None, R2=None, gamma1=0.0, gamma2=0.0):
    """L(f,D) = -n·ln(SR_clip) + (ln|Hk|/n)·k   [+ γ1·S1 + γ2·S2]

    YCPA-P2 düzgünlük regülarizasyonu (companion makale, Eq. 4) — PURELY ADDITIVE.
    Opsiyonel kuadratik-form cezaları:
        S1 = βᵀ R1 β  (total-variation/Dirichlet),  S2 = βᵀ R2 β  (curvature/Hessian).
    GERİYE-DÖNÜK UYUM: γ1=γ2=0 (varsayılan) iken ek terim tam sıfırdır ve skor
    orijinal implementasyonla bit-for-bit aynıdır (makale §4.5).
    """
    n = len(yt); m = mad(yt); eps = 0.4*(m if m>0 else 1.0)
    sr = compute_SR(yp, yt, eps); Hk = max((2*H)**k,1)
    L = -n*math.log(sr) + (math.log(Hk)/n)*k
    if beta is not None and (gamma1 or gamma2):
        if gamma1 and R1 is not None: L += gamma1*float(beta@R1@beta)
        if gamma2 and R2 is not None: L += gamma2*float(beta@R2@beta)
    return L


# ── YCPA-P2: kapalı-form eğrilik/TV ceza matrisleri (companion makale §2.4, Eq.5)
def smoothness_penalty_matrices_1d(deg, x_min, x_max, n_grid=600):
    """Monomial baz x^p için R1=∫(φ')(φ')ᵀ, R2=∫(φ'')(φ'')ᵀ kuadratik formları.
    S = cᵀ R c.  φ''_p = p(p-1)x^(p-2)."""
    xs = np.linspace(x_min, x_max, n_grid)
    dx = (x_max - x_min) / (n_grid - 1)
    P = np.arange(deg+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = P*np.where(P-1>=0, xs[:,None]**np.clip(P-1,0,None), 0.0)
        d2 = P*(P-1)*np.where(P-2>=0, xs[:,None]**np.clip(P-2,0,None), 0.0)
    d1 = np.nan_to_num(d1); d2 = np.nan_to_num(d2)
    return d1.T@d1*dx, d2.T@d2*dx


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3B — YCPA-P2 DÜZGÜNLÜK REGÜLARİZASYONU  (companion makale, tüm prosedürler)
# §2.1 S1/S2 (1-B) · §2.2 Dirichlet+Hessian-Frobenius+Laplacian (D-B) ·
# §2.4 kapalı-form R + Gauss-divergence sınır indirgemesi · Eq.1-2 varyasyonel
# fonksiyoneller · §5 ridge-tipi regülarize çözüm.
# (mdl_score / tournament_mcmc entegrasyonu yukarıda; bu blok matris/araç katmanı.)
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# §2.4  KAPALI-FORM CEZA MATRİSLERİ   (Eq. 5)
# ──────────────────────────────────────────────────────────────────────────────
# Monomial baz φ_p(x) = x^p,  p = 0..deg
#   φ'_p(x)  = p · x^(p-1)
#   φ''_p(x) = p(p-1) · x^(p-2)
# R = Σ_grid φ'' φ''ᵀ Δx   (curvature, S2);   R1 benzeri φ' için (TV proxy, S1)

def _monomial_basis_grid(deg: int, xs: np.ndarray):
    """Design matrices [φ, φ', φ''] on grid xs (each (len(xs), deg+1))."""
    P = np.arange(deg + 1)                      # 0,1,...,deg
    Phi   = xs[:, None] ** P[None, :]                                   # x^p
    # türevler — negatif kuvvetlerde 0 katsayısı sıfırladığı için güvenli
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = P * np.where(P - 1 >= 0, xs[:, None] ** np.clip(P - 1, 0, None), 0.0)
        d2 = P * (P - 1) * np.where(P - 2 >= 0, xs[:, None] ** np.clip(P - 2, 0, None), 0.0)
    d1 = np.nan_to_num(d1); d2 = np.nan_to_num(d2)
    return Phi, d1, d2


def build_penalty_matrices_1d(deg: int, x_min: float, x_max: float,
                              n_grid: int = 400):
    """
    Tek değişken için R1 (TV-benzeri, S1) ve R2 (curvature, S2) kuadratik-form
    matrislerini döndürür.  S = cᵀ R c  (Eq. 5).

    R2 = Σ φ'' φ''ᵀ Δx   →  ∫(f'')² dx
    R1 = Σ φ'  φ' ᵀ Δx   →  ∫(f')²  dx   (∫|f'| yerine konveks kuadratik vekil)
    """
    xs = np.linspace(x_min, x_max, n_grid)
    dx = (x_max - x_min) / (n_grid - 1)
    _, d1, d2 = _monomial_basis_grid(deg, xs)
    R1 = d1.T @ d1 * dx
    R2 = d2.T @ d2 * dx
    return R1, R2


def build_laplacian_penalty_nd(degrees, bounds, n_grid_per_dim: int = 30):
    """
    §2.2'deki eşdeğer ikinci-mertebe varyant: kareli-Laplacian cezası
        S2^Δ = ∫_Ω (∇²f)² dx ,    ∇²f = Σ_d ∂²f/∂x_d² .
    R_Δ = Σ_grid (Σ_d φ''_dd)(Σ_d φ''_dd)ᵀ ΔV   →  S = cᵀ R_Δ c.
    Hessian-Frobenius'tan farkı: çapraz türevleri içermez, izleri toplar.
    """
    from itertools import product
    D = len(degrees)
    axes = [np.linspace(b[0], b[1], n_grid_per_dim) for b in bounds]
    dV = np.prod([(b[1]-b[0])/(n_grid_per_dim-1) for b in bounds])
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [m.ravel() for m in mesh]
    G = flat[0].size
    multi = list(product(*[range(d+1) for d in degrees]))
    M = len(multi)

    def col_deriv(vals, p, order):
        if order == 0: return vals ** p
        if order == 1:
            return (p*np.where(p-1>=0, vals**max(p-1,0), 0.0) if p>=1 else np.zeros_like(vals))
        return (p*(p-1)*np.where(p-2>=0, vals**max(p-2,0), 0.0) if p>=2 else np.zeros_like(vals))

    # Laplacian kolonu: Σ_d ∂²φ/∂x_d²  (her baz fonksiyonu için)
    Lap = np.zeros((G, M))
    for d in range(D):
        block = np.ones((G, M))
        for dd in range(D):
            order = 2 if dd == d else 0
            tab = np.column_stack([col_deriv(flat[dd], mi[dd], order) for mi in multi])
            block = block * tab
        Lap = Lap + block
    R_lap = Lap.T @ Lap * dV
    return R_lap, multi


def boundary_reduce_1d(deg, x_min, x_max):
    """
    §2.4: Gauss-divergence teoremiyle hacim integralini SINIR integraline indirgeme.
    1-B'de ∫(f'')² dx kısmî integrasyonla
        ∫_a^b (f'')² dx = [f'' f']_a^b - ∫_a^b f''' f' dx
    şeklinde yazılır; sınır terimi B = f''(b)f'(b) - f''(a)f'(a) kuadratik formdur:
        B = cᵀ R_bnd c,   R_bnd = φ''(b)φ'(b)ᵀ - φ''(a)φ'(a)ᵀ  (simetrikleştirilir).
    Bu, çerçevenin kapalı-form skorlama felsefesine uygun O(1) sınır değerlendirmesidir.
    (Tam hacim R2'nin ucuz boundary-proxy'si; ızgara kuadratürünü gerektirmez.)
    """
    P = np.arange(deg+1)
    def d1(x): 
        return P*np.where(P-1>=0, x**np.clip(P-1,0,None), 0.0)
    def d2(x):
        return P*(P-1)*np.where(P-2>=0, x**np.clip(P-2,0,None), 0.0)
    a1, a2 = np.nan_to_num(d1(x_min)), np.nan_to_num(d2(x_min))
    b1, b2 = np.nan_to_num(d1(x_max)), np.nan_to_num(d2(x_max))
    Rb = np.outer(b2, b1) - np.outer(a2, a1)
    return 0.5*(Rb + Rb.T)            # simetrik kuadratik form


def build_penalty_matrices_nd(degrees, bounds, n_grid_per_dim: int = 30):
    """
    D-boyut için tensör-monomial baz üzerinde Hessian-Frobenius (thin-plate, S2)
    ve Dirichlet (S1) ceza matrisleri.   (makale §2.2, Eq. 3)

    degrees : her boyut için maksimum derece, ör. (deg_x, deg_y)
    bounds  : [(x_min,x_max), (y_min,y_max), ...]
    Baz: φ_{p,q,...}(x) = Π_d x_d^{p_d}.   R = Σ_grid (Σ_ij ∂²φ/∂x_i∂x_j ⊗ ...) ΔV
    """
    D = len(degrees)
    axes = [np.linspace(b[0], b[1], n_grid_per_dim) for b in bounds]
    dV = np.prod([(b[1] - b[0]) / (n_grid_per_dim - 1) for b in bounds])
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [m.ravel() for m in mesh]                       # her boyut düzleştirilmiş
    G = flat[0].size

    # çok-indeksli üs listesi
    from itertools import product
    multi = list(product(*[range(d + 1) for d in degrees]))   # (p0,p1,...)
    M = len(multi)

    # her boyut için 0/1/2. türev tek-değişkenli tablolar
    def col_deriv(vals, p, order):
        if order == 0:
            return vals ** p
        if order == 1:
            return (p * np.where(p - 1 >= 0, vals ** max(p - 1, 0), 0.0)
                    if p >= 1 else np.zeros_like(vals))
        # order == 2
        return (p * (p - 1) * np.where(p - 2 >= 0, vals ** max(p - 2, 0), 0.0)
                if p >= 2 else np.zeros_like(vals))

    # Hessian-Frobenius:  Σ_{i,j} (∂²φ/∂x_i∂x_j)
    # her baz fonksiyonu için tüm (i,j) ikinci türev kolonlarını topla
    Hess_cols = np.zeros((G, M))           # ‖H‖_F için: kareler toplamı baz başına
    Grad_cols = np.zeros((G, M))           # Dirichlet için
    # Tam Frobenius normu kuadratik form olarak: R2 = Σ_{i<=j} w_ij (h_ij)(h_ij)ᵀ
    R2 = np.zeros((M, M))
    R1 = np.zeros((M, M))
    for i in range(D):
        for j in range(D):
            # ∂²/∂x_i∂x_j φ  kolon matrisi
            Hcol = np.ones((G, M))
            for d in range(D):
                order = (1 if d == i else 0) + (1 if d == j else 0)
                vals = flat[d]
                tab = np.column_stack([col_deriv(vals, mi[d], order) for mi in multi])
                Hcol = Hcol * tab
            R2 += Hcol.T @ Hcol * dV       # Frobenius: Σ_ij (∂²φ)²
        # gradyan bileşeni i
        Gcol = np.ones((G, M))
        for d in range(D):
            order = 1 if d == i else 0
            vals = flat[d]
            tab = np.column_stack([col_deriv(vals, mi[d], order) for mi in multi])
            Gcol = Gcol * tab
        R1 += Gcol.T @ Gcol * dV           # Dirichlet: Σ_i (∂φ)²
    return R1, R2, multi


def design_matrix_1d(deg: int, x: np.ndarray):
    """Monomial design matrix Φ at data points (n, deg+1)."""
    P = np.arange(deg + 1)
    return x[:, None] ** P[None, :]


def design_matrix_nd(multi, Xcols):
    """Tensor-monomial design for multi-index list and Xcols=[x0,x1,...]."""
    n = len(Xcols[0])
    Phi = np.ones((n, len(multi)))
    for j, mi in enumerate(multi):
        col = np.ones(n)
        for d, p in enumerate(mi):
            col = col * (Xcols[d] ** p)
        Phi[:, j] = col
    return Phi



# ──────────────────────────────────────────────────────────────────────────────
# REGÜLARİZE EDİLMİŞ LİNEER ÇÖZÜM  (ridge-tipi normal denklem; makale §5)
#   (Φᵀ Φ + n·γ2·R2 + n·γ1·R1) c = Φᵀ y
# Bu, βᵀRβ cezalı en-küçük-kareler çözümüdür; penalty'nin etkisini şeffaf kılar.
# ──────────────────────────────────────────────────────────────────────────────
def fit_regularized_linear(Phi, y, R1=None, R2=None, gamma1=0.0, gamma2=0.0,
                           ridge=1e-10):
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    n = len(y)
    if gamma1 and R1 is not None:
        A = A + n * gamma1 * R1
    if gamma2 and R2 is not None:
        A = A + n * gamma2 * R2
    b = Phi.T @ y
    return np.linalg.solve(A, b)


# ──────────────────────────────────────────────────────────────────────────────
# GERÇEK VARYASYONEL FONKSİYONELLER  (makale Eq. 1-2, doğrudan integral)
#   S1 = ∫|f'|dx  (total variation, mutlak değer — kuadratik DEĞİL)
#   S2 = ∫(f'')²dx
# Kuadratik R1/R2 matrisleri optimizasyon içindir; bunlar raporlama/teşhis içindir.
# ──────────────────────────────────────────────────────────────────────────────
def total_variation_1d(coeffs, deg, x_min, x_max, n_grid=2000):
    """S1(f) = ∫|f'(x)|dx  — paper Eq. 1, for a monomial coefficient vector."""
    xs = np.linspace(x_min, x_max, n_grid); dx = (x_max-x_min)/(n_grid-1)
    P = np.arange(deg+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = P*np.where(P-1>=0, xs[:,None]**np.clip(P-1,0,None), 0.0)
    fp = np.nan_to_num(d1) @ coeffs
    return float(np.sum(np.abs(fp))*dx)


def curvature_energy_1d(coeffs, deg, x_min, x_max, n_grid=2000):
    """S2(f) = ∫(f''(x))² dx  — makale Eq. 2."""
    xs = np.linspace(x_min, x_max, n_grid); dx = (x_max-x_min)/(n_grid-1)
    P = np.arange(deg+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = P*(P-1)*np.where(P-2>=0, xs[:,None]**np.clip(P-2,0,None), 0.0)



def r2(yp,yt):
    sst = np.sum((yt-yt.mean())**2)
    val = float(1-np.sum((yt-yp)**2)/(sst+1e-300))
    return float(np.clip(val,-1e6,1.0))   # overflow koruma

def rmse(yp, yt):
    yp, yt = np.asarray(yp, dtype=float), np.asarray(yt, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def smape(yp, yt):
    """sMAPE — denominator (|y| + |ŷ|) / 2."""
    yp, yt = np.asarray(yp, dtype=float), np.asarray(yt, dtype=float)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0 + 1e-300
    raw = float(np.mean(np.abs(yt - yp) / denom)) * 100.0
    return float(np.clip(raw, 0, 1e6))


def nrmse_pct(yp, yt):
    """nRMSE = RMSE / mean(|y|) · 100%."""
    yp, yt = np.asarray(yp, dtype=float), np.asarray(yt, dtype=float)
    m = float(np.mean(np.abs(yt))) or 1e-300
    return float(np.clip(rmse(yp, yt) / m * 100.0, 0, 1e6))


def mape(yp, yt):
    """Backward compatibility — returns sMAPE."""
    return smape(yp, yt)


def error_metrics(yp, yt) -> dict:
    """
    Hata ölçütleri: sMAPE veya (çok sıfıra yakın y varsa) nRMSE.
    >%5 örnekte |y| < 0.1·std(y) ise nRMSE kullanılır.
    """
    yp, yt = np.asarray(yp, dtype=float), np.asarray(yt, dtype=float)
    rmse_v = rmse(yp, yt)
    smape_v = smape(yp, yt)
    nrmse_v = nrmse_pct(yp, yt)
    std_y = float(np.std(yt)) or 1e-300
    near_zero_frac = float(np.mean(np.abs(yt) < 0.1 * std_y))
    use_nrmse = near_zero_frac > 0.05
    if use_nrmse:
        return {
            "rmse": rmse_v,
            "mape": nrmse_v,
            "smape": smape_v,
            "nrmse": nrmse_v,
            "mape_label": "nRMSE%",
            "mape_note": "MAPE unreliable, showing nRMSE",
            "use_nrmse": True,
            "near_zero_frac": near_zero_frac,
        }
    return {
        "rmse": rmse_v,
        "mape": smape_v,
        "smape": smape_v,
        "nrmse": nrmse_v,
        "mape_label": "sMAPE%",
        "mape_note": None,
        "use_nrmse": False,
        "near_zero_frac": near_zero_frac,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — MDL*  (v4) — kapalı form dahil
# L*(f) = λ·r^k̂·max(0, r^{Δk}-1)
# 1. derece: L* ≈ λ·r^k̂·Δk·ln r   (ΠH: katsayı=1, 1/2 — tam rasyonel)
# ═══════════════════════════════════════════════════════════════════════════════

def mdl_star(yp_tr, yt_tr, yp_te, yt_te, k_hat=None, lam=1.0) -> dict:
    n = len(yt_tr)+len(yt_te)
    sr_tr = compute_SR(yp_tr, yt_tr, 0.4*(mad(yt_tr) or 1.0))
    sr_te = compute_SR(yp_te, yt_te, 0.4*(mad(yt_te) or 1.0))
    gap   = max(0.0, sr_tr-sr_te)
    L_sr  = lam*gap*n                    # SR gap versiyonu

    # Kapalı form: r^k̂ · Δk · ln r  (n_large/n_small oranı)
    r = len(yt_te)/(len(yt_tr)+1e-300)
    kh = k_hat if k_hat else 1
    dk = gap * kh                        # Δk yaklaşımı: gap × k̂
    L_closed_1 = lam * (r**kh) * dk * math.log(r+1e-300)   # 1. derece
    L_closed_2 = lam * (r**kh) * (dk*math.log(r+1e-300)
                 + 0.5*(dk*math.log(r+1e-300))**2)          # 2. derece (1/2 ΠH)
    return {
        "SR_train": sr_tr, "SR_test": sr_te, "gap": gap,
        "L_star": L_sr, "L_closed_1": L_closed_1, "L_closed_2": L_closed_2,
        "r": r, "k_hat": kh, "delta_k": dk,
        "overfit": gap > 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — SİGMOİD HİBRİT GEÇİŞ  (v4)
# f̂(n)=(1-w)·f_eski+w·f_yeni,  w=1/(1+exp(-3(n-n*)/δ))
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid_hybrid(y_old, y_new, n_star=None, delta=None):
    n = len(y_old)
    if n_star is None: n_star = n/2
    if delta  is None: delta  = max(n/6, 1)
    w = 1.0/(1.0+np.exp(-3*(np.arange(n,dtype=float)-n_star)/delta))
    return (1-w)*y_old + w*y_new


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — GİBBS ENTROPİSİ + TURNUVA MCMC  (v5)
# ═══════════════════════════════════════════════════════════════════════════════

def gibbs_entropy(energies, T=1.0) -> dict:
    """
    H=-Σpᵢln pᵢ,  H̃=H/ln N
    Büyük-T: H(T)≈ln N - Var(E)/(2NT²)
    Isınma: r(H)=r_min+(r_max-r_min)·H̃  [tam lineer, katsayı=1]
    Normalize: H̃∈[0,1] — keşfet(→1) / yakınsadı(→0)
    """
    e = np.asarray(energies,float); e = e-e.min()
    w = np.exp(-e/(T+1e-300)); p = w/w.sum()
    H = float(-np.sum(p*np.log(p+1e-300)))
    N = len(e); Hmax = math.log(max(N,2))
    He = H/(Hmax+1e-300)
    # Büyük-T asimptotu: H(T→∞) ≈ ln N - Var(E)/(2NT²)
    var_E  = float(np.var(energies))
    H_asymp = Hmax - var_E/(2*max(N,1)*T**2+1e-300)
    He_asymp = H_asymp/(Hmax+1e-300)
    return {"H":H,"Hmax":Hmax,"He":He,"probs":p.tolist(),"r_heat":0.01+0.99*He,
            "H_asymp":H_asymp,"He_asymp":He_asymp,"var_E":var_E,"T":T}

def tournament_mcmc(X, y, n_chains=25, n_warmup=200, n_deep=2000,
                    n_elite=5, step_size=0.05, H_piH=20, verbose=False,
                    R1=None, R2=None, gamma1=0.0, gamma2=0.0) -> dict:
    """Turnuva MCMC dispatcher.

    C++ çekirdeği (ycpa_core) varsa hızlı yola delege eder; yoksa saf-Python
    uygulamaya düşer. İkisi de aynı dict'i döndürür:
    {beta_mcmc, energy, gibbs, n_chains, n_elite}.

    Not: C++ ve Python bağımsız RNG kullandığından sonuçlar istatistiksel olarak
    eşdeğerdir (aynı enerji minimumuna yakınsar) ama bit-bit aynı değildir.
    Bit-bit Python davranışı gerekiyorsa YCPA_NO_CPP=1 ile C++ devre dışı bırakılır.
    """
    if _USE_CPP:
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        yc = np.ascontiguousarray(y, dtype=np.float64).ravel()
        mc = _CPP.TournamentMCMC(n_chains=n_chains, n_warmup=n_warmup,
                                 n_deep=n_deep, n_elite=n_elite,
                                 step_size=step_size, H_piH=H_piH, seed=42)
        res = mc.fit(Xc, yc,
                     R1=(None if R1 is None else np.ascontiguousarray(R1, dtype=np.float64)),
                     R2=(None if R2 is None else np.ascontiguousarray(R2, dtype=np.float64)),
                     gamma1=gamma1, gamma2=gamma2)
        g = res.gibbs
        gibbs = {"H": g.H, "Hmax": g.Hmax, "He": g.He,
                 "r_heat": g.r_heat, "var_E": g.var_E, "T": g.T}
        return {"beta_mcmc": np.asarray(res.beta_mcmc), "energy": res.energy,
                "gibbs": gibbs, "n_chains": res.n_chains, "n_elite": res.n_elite}
    return _tournament_mcmc_py(X, y, n_chains, n_warmup, n_deep, n_elite,
                               step_size, H_piH, verbose, R1, R2, gamma1, gamma2)


def _tournament_mcmc_py(X, y, n_chains=25, n_warmup=200, n_deep=2000,
                    n_elite=5, step_size=0.05, H_piH=20, verbose=False,
                    R1=None, R2=None, gamma1=0.0, gamma2=0.0) -> dict:
    """
    Turnuva MCMC — v5 tam implementasyonu (saf-Python referans)
    Aşama 1 : n_chains bağımsız zincir, n_warmup ısınma (Gibbs adaptif T)
    Aşama 2 : MDL + çeşitlilik filtresi → n_elite seçimi
    Aşama 3 : n_deep derin arama (Gibbs He'den türetilen başlangıç T)

    YCPA-P2 (opsiyonel): R1/R2 + gamma1/gamma2 verilirse enerji, eğrilik/TV
    cezalı MDL'dir (Eq. 4). Varsayılan (gamma=0) tam geriye-dönük uyumludur.
    """
    rng = np.random.default_rng(42); k = X.shape[1]
    def energy(b): return mdl_score(X@b, y, k, H_piH,
                                    beta=b, R1=R1, R2=R2,
                                    gamma1=gamma1, gamma2=gamma2)

    # Aşama 1
    chains = []
    for _ in range(n_chains):
        b = rng.normal(0,0.5,k); e = energy(b); T = 2.0
        for _ in range(n_warmup):
            bc = b+rng.normal(0,step_size,k); ec = energy(bc)
            if ec<e or rng.random()<math.exp(min(0,(e-ec)/T)):
                b,e = bc,ec
            T *= 0.995
        chains.append({"beta":b.copy(),"energy":e})

    gb = gibbs_entropy(np.array([c["energy"] for c in chains]))
    if verbose:
        print(f"    [MCMC Stage1] H̃={gb['He']:.4f}  "
              f"({'explore' if gb['He']>0.7 else 'converged'})")

    # Aşama 2: elite seçimi
    chains.sort(key=lambda c: c["energy"])
    elite = [chains[0]]
    for c in chains[1:]:
        if len(elite)>=n_elite: break
        if all(np.linalg.norm(c["beta"]-e["beta"])>0.1 for e in elite):
            elite.append(c)
    while len(elite)<n_elite and len(elite)<len(chains):
        elite.append(chains[len(elite)])

    # Aşama 3: derin arama
    best_b = elite[0]["beta"].copy(); best_e = elite[0]["energy"]
    T = gb["r_heat"]*0.5
    for ec in elite:
        b,e = ec["beta"].copy(),ec["energy"]
        for _ in range(n_deep):
            bc = b+rng.normal(0,step_size*0.3,k); ec2 = energy(bc)
            if ec2<e or rng.random()<math.exp(min(0,(e-ec2)/T)):
                b,e = bc,ec2
            if e<best_e: best_e,best_b = e,b.copy()
            T *= 0.9997
    if verbose: print(f"    [MCMC Stage3] MDL={best_e:.4f}")
    return {"beta_mcmc":best_b,"energy":best_e,"gibbs":gb,
            "n_chains":n_chains,"n_elite":n_elite}


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — v6.1 FİZİK PROTOKOLLER
# P5  : SI boyut Hard-Kill — boyutsal tutarsız adaylar elenur
# P15 : Türev sürekliliği — ani sıçrama penaltisi (Cdiff)
# Cax : Eksen simetrisi cezası
# ═══════════════════════════════════════════════════════════════════════════════

# SI temel birimler: [M, L, T, I, Θ, N, J]
KNOWN_DIMENSIONS = {
    "kuvvet":    [ 1, 1,-2, 0, 0, 0, 0],
    "basinc":    [ 1,-1,-2, 0, 0, 0, 0],
    "enerji":    [ 1, 2,-2, 0, 0, 0, 0],
    "guc":       [ 1, 2,-3, 0, 0, 0, 0],
    "hiz":       [ 0, 1,-1, 0, 0, 0, 0],
    "ivme":      [ 0, 1,-2, 0, 0, 0, 0],
    "moment":    [ 1, 2,-2, 0, 0, 0, 0],
    "yogunluk":  [ 1,-3, 0, 0, 0, 0, 0],
    "sicaklik":  [ 0, 0, 0, 0, 1, 0, 0],
    "entropi":   [ 1, 2,-2, 0,-1, 0, 0],
    "boyutsuz":  [ 0, 0, 0, 0, 0, 0, 0],
}

def p5_hard_kill(lhs_dim: list, rhs_dim: list) -> bool:
    """
    P5 — SI Hard-Kill: lhs ve rhs boyut vektörleri uyuşmuyorsa True (elensin).
    Boyutsal tutarsızlık oranı tipik fiziksel sistemlerde ~%85 → arama uzayını 6× daraltır.
    """
    return not all(abs(a-b)<1e-9 for a,b in zip(lhs_dim,rhs_dim))

def p15_diff_penalty(yp, yt, lambda_diff=1.0) -> float:
    """
    P15 — Türev sürekliliği: birinci fark dizisindeki ani sıçramalar
    Cdiff = Σ max(0, |Δf'| - median(|Δf'|)·threshold)
    """
    res = np.asarray(yt)-np.asarray(yp)
    d1  = np.diff(res)
    med = float(np.median(np.abs(d1))) or 1.0
    return float(lambda_diff*np.sum(np.maximum(0, np.abs(d1)-3*med)))

def cax_penalty(yp, lambda_ax=1.0) -> float:
    """
    Cax — Eksen simetrisi: f(x) ≈ f(-x) beklentisi için simetri kırılması penaltisi
    Cax = |mean(f(x))| / (std(f(x))+ε)  — tahmin ortalaması sıfırdan uzaklaşırsa ceza
    """
    yp = np.asarray(yp,float)
    return float(lambda_ax*abs(np.mean(yp))/(np.std(yp)+1e-300))

# ─── v6.1 Ek Protokoller P2–P4, P7–P14, P16 ──────────────────────────────────

def p2_monotonicity(yp, yt, lam=1.0) -> float:
    """P2 — Monotonluk: verinin yönü ile tahmin yönü ters ise ceza.
    C_mono = Σ max(0, -sign(Δy)·sign(Δf))  / (n-1)
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    dy = np.diff(yt); df = np.diff(yp)
    violations = np.sum(np.maximum(0, -np.sign(dy)*np.sign(df)))
    return float(lam * violations / max(len(dy),1))

def p3_zero_crossing(yp, yt, lam=1.0) -> float:
    """P3 — Sıfır geçişi: f(0)≈0 beklentisi.
    Veri sıfıra yakın bir x varsa, tahminin de sıfıra yakın olması gerekir.
    C_zero = |f(x_min)| / (range(y)+ε)  — x en küçük değerde tahmin sapması
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    yrange = float(np.ptp(yt)) or 1.0
    return float(lam * abs(yp[0]) / yrange)

def p4_asymptotic_bound(yp, yt, lam=1.0) -> float:
    """P4 — Asimptotik sınır: uç noktalarda tahmin veride sınırlı mı?
    C_asym = max(0, |f_son - y_son| - 3·std(y)) / std(y)
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    std_y = float(np.std(yt)) or 1.0
    tail_err = abs(float(yp[-1]) - float(yt[-1]))
    return float(lam * max(0, tail_err - 3*std_y) / std_y)

def p7_coeff_sign(beta_vals: list, expected_signs: list = None, lam=1.0) -> float:
    """P7 — Katsayı işareti: beklenen fiziksel işaretlerden sapma.
    expected_signs: [+1,-1,0,...] — 0=fark etmez
    C_sign = Σ max(0, -sign_beklenen · sign_gerçek)
    """
    if not expected_signs: return 0.0
    violations = 0
    for b, es in zip(beta_vals, expected_signs):
        if es != 0 and np.sign(b) != np.sign(es):
            violations += 1
    return float(lam * violations)

def p8_output_bounds(yp, yt, lam=1.0) -> float:
    """P8 — Çıkış sınırları: tahmin veri aralığının dışına çıkıyor mu?
    C_bounds = (aşım miktarı) / range(y)
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    ymin,ymax = float(yt.min()),float(yt.max()); yrange = (ymax-ymin) or 1.0
    over  = np.sum(np.maximum(0, yp-ymax))
    under = np.sum(np.maximum(0, ymin-yp))
    return float(lam*(over+under)/yrange/len(yp))

def p9_continuity(yp, lam=1.0) -> float:
    """P9 — Süreklilik: komşu tahminler arası ani sıçrama.
    C_cont = std(|Δf|) / mean(|Δf|+ε)  — varyasyon katsayısı
    """
    yp = np.asarray(yp,float)
    df = np.abs(np.diff(yp))
    return float(lam * np.std(df) / (np.mean(df)+1e-300))

def p10_periodicity(yp, yt, lam=1.0) -> tuple:
    """P10 — Periyodiklik: verinin dominant frekansı ile tahmin frekansı.
    FFT ile dominant freq tespit; eşleşmiyorsa ceza.
    Döndürür: (ceza, dominant_frek_veri, dominant_frek_tahmin)
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    n = len(yt)
    if n < 8:
        return 0.0, 0.0, 0.0
    fft_y = np.abs(np.fft.rfft(yt - yt.mean()))
    fft_f = np.abs(np.fft.rfft(yp - yp.mean()))
    dom_y = int(np.argmax(fft_y[1:])+1)
    dom_f = int(np.argmax(fft_f[1:])+1)
    penalty = float(lam * abs(dom_y - dom_f) / max(dom_y,1))
    return penalty, dom_y, dom_f

def p11_scale_invariance(yp, yt, X_vals, lam=1.0) -> float:
    """P11 — Skala değişmezliği: f(λx)≈λ^α·f(x) testi.
    λ=2 ile test: C_scale = |f(2x_mid)/f(x_mid) - 2^α_tahmin| / |f(x_mid)|
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    n = len(yp)
    if n < 4: return 0.0
    mid = n//2
    ratio_y = abs(float(yt[mid+1]))/(abs(float(yt[mid]))+1e-300)
    ratio_f = abs(float(yp[mid+1]))/(abs(float(yp[mid]))+1e-300)
    return float(lam * abs(ratio_y - ratio_f) / (abs(ratio_y)+1e-300))

def p12_even_odd(yp, yt, lam=1.0) -> dict:
    """P12 — Çift/tek fonksiyon testi: f(-x)≈±f(x).
    Veri simetrik mi test eder; en az n//2 negatif x gerekir.
    Simetri skoru: 1=tam çift, -1=tam tek, 0=ne
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    n = len(yp)
    if n < 4:
        return {"penalty":0.0,"symmetry":"bilinmiyor"}
    half = n//2
    # Tek sayılı n'de ortadaki noktayı atla (simetri için sol/sağ eşit uzunlukta olmalı)
    left = yp[:half]
    right = yp[n - half:][::-1]
    even_err = float(np.mean(np.abs(left + right)))
    odd_err  = float(np.mean(np.abs(left - right)))
    yrange   = float(np.ptp(yp)) or 1.0
    if even_err < odd_err:
        sym = "çift (f(-x)=f(x))"; pen = lam*even_err/yrange
    else:
        sym = "tek (f(-x)=-f(x))"; pen = lam*odd_err/yrange
    return {"penalty":pen,"symmetry":sym}

def p13_positivity(yp, yt, lam=1.0) -> float:
    """P13 — Pozitif tanımlılık: fiziksel büyüklük negatif olamaz.
    Yalnızca yt>0 ise aktif: C_pos = Σ max(0,-f) / n
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    if np.any(yt < 0): return 0.0   # negatif yt varsa kısıt geçersiz
    neg = np.sum(np.maximum(0, -yp))
    return float(lam * neg / len(yp))

def p14_interaction_order(beta_vals: list, lam=1.0) -> float:
    """P14 — Etkileşim sırası: yüksek dereceli terimler küçük dereceli yokken var.
    Basit proxy: katsayı büyüklük sırası bozulmuşsa ceza.
    C_ord = #{i: |β_i| > |β_{i-1}|} / k
    """
    bv = [abs(b) for b in beta_vals[1:]]  # intercept hariç
    if len(bv) < 2: return 0.0
    violations = sum(1 for i in range(1,len(bv)) if bv[i] > bv[i-1])
    return float(lam * violations / len(bv))

def energy_conservation_deep(yp, yt, x_vals=None, lam=1.0, tol_rel: float = 0.15) -> dict:
    """
    P16 genişletilmiş — kümülatif toplam, birinci ve ikinci moment korunumu.
    Üç kontrolden ikisi başarısızsa uyarı.
    """
    yp = np.asarray(yp, dtype=float).ravel()
    yt = np.asarray(yt, dtype=float).ravel()
    n = len(yt)
    out = {
        "penalty": 0.0,
        "max_cumulative_drift": 0.0,
        "moment1_rel": 0.0,
        "moment2_rel": 0.0,
        "ok_cumulative": True,
        "ok_moment1": True,
        "ok_moment2": True,
        "warning": False,
        "warning_message": "",
    }
    if n < 3:
        return out

    cum_f = np.cumsum(yp)
    cum_y = np.cumsum(yt)
    drift = float(np.max(np.abs(cum_f - cum_y)) / (np.max(np.abs(cum_y)) + 1e-300))
    out["max_cumulative_drift"] = drift
    out["ok_cumulative"] = drift < tol_rel

    sum_pen = abs(yp.sum() - yt.sum()) / (abs(yt.sum()) + 1e-300)
    fails = 0 if out["ok_cumulative"] else 1

    if x_vals is not None:
        X = np.asarray(x_vals, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] == n:
            m1f = np.sum(X * yp.reshape(-1, 1), axis=0)
            m1y = np.sum(X * yt.reshape(-1, 1), axis=0)
            rel1 = np.max(np.abs(m1f - m1y) / (np.abs(m1y) + 1e-300))
            out["moment1_rel"] = float(rel1)
            out["ok_moment1"] = rel1 < tol_rel
            if not out["ok_moment1"]:
                fails += 1

            m2f = np.sum((X ** 2) * yp.reshape(-1, 1), axis=0)
            m2y = np.sum((X ** 2) * yt.reshape(-1, 1), axis=0)
            rel2 = np.max(np.abs(m2f - m2y) / (np.abs(m2y) + 1e-300))
            out["moment2_rel"] = float(rel2)
            out["ok_moment2"] = rel2 < tol_rel
            if not out["ok_moment2"]:
                fails += 1
    else:
        out["ok_moment1"] = out["ok_moment2"] = True

    out["penalty"] = float(lam * (drift + sum_pen + out["moment1_rel"] + out["moment2_rel"]))
    if fails >= 2:
        out["warning"] = True
        out["warning_message"] = "⚠ P16: Enerji momenti korunmuyor"
    return out


def p16_conservation(yp, yt, x_vals=None, lam=1.0) -> float:
    """P16 — energy conservation (deep-check penalty)."""
    return energy_conservation_deep(yp, yt, x_vals, lam=lam)["penalty"]


def symbolic_derivative_check(
    yp,
    yt,
    X_vals,
    feature_names,
    data=None,
    res=None,
    target=None,
    rel_bump: float = 0.01,
) -> dict:
    """
    ∂f/∂xᵢ işaretini %1 pertürbasyonla tahmin et; veri korelasyon işaretiyle karşılaştır.
    """
    yp = np.asarray(yp, dtype=float).ravel()
    yt = np.asarray(yt, dtype=float).ravel()
    X = np.asarray(X_vals, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(yt)
    warnings: list[str] = []
    details: list[dict] = []

    for j, name in enumerate(feature_names):
        if j >= X.shape[1]:
            break
        xi = X[:, j]
        if np.std(xi) < 1e-14:
            continue
        try:
            corr = float(np.corrcoef(xi, yt)[0, 1])
        except Exception:
            corr = 0.0
        expected = int(np.sign(corr)) if np.isfinite(corr) and abs(corr) > 1e-6 else 0

        numeric_sign = 0
        if (
            res is not None
            and data is not None
            and target is not None
            and res.get("model_type") == "power_law"
            and name in res.get("features", [])
        ):
            try:
                y0 = _power_predict(data, target, res["features"], res["beta_ols"])
                d_p = data.copy()
                bump = rel_bump * np.maximum(np.abs(xi), np.std(xi), 1e-8)
                d_p[name] = xi + bump
                y1 = _power_predict(d_p, target, res["features"], res["beta_ols"])
                med_d = float(np.median(y1 - y0))
                numeric_sign = int(np.sign(med_d)) if abs(med_d) > 1e-14 * np.max(np.abs(y0)) else 0
            except Exception:
                numeric_sign = 0
        if numeric_sign == 0:
            dx = np.gradient(xi)
            df = np.gradient(yp)
            mask = np.abs(dx) > 1e-14
            if np.any(mask):
                local = df[mask] / dx[mask]
                numeric_sign = int(np.sign(np.median(local)))

        conflict = (
            expected != 0
            and numeric_sign != 0
            and expected != numeric_sign
        )
        rec = {
            "feature": name,
            "expected_sign": expected,
            "numeric_sign": numeric_sign,
            "corr": corr,
            "conflict": conflict,
        }
        details.append(rec)
        if conflict:
            if expected > 0:
                msg = (
                    f"⚠ P_türev: {name} artarken f azalıyor ama veri artış gösteriyor"
                )
            else:
                msg = (
                    f"⚠ P_türev: {name} azalırken f artıyor ama veri azalış gösteriyor"
                )
            warnings.append(msg)

    return {"warnings": warnings, "details": details, "n_conflicts": len(warnings)}


def asymptotic_behavior_check(
    yp, yt, X_vals, feature_names: list | None = None,
    q_low: float = 0.10, q_high: float = 0.90,
) -> dict:
    """x→min ve x→max dilimlerinde tahmin davranışı."""
    yp = np.asarray(yp, dtype=float).ravel()
    yt = np.asarray(yt, dtype=float).ravel()
    X = np.asarray(X_vals, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    warnings: list[str] = []
    details: list[dict] = []
    sig_y = float(np.std(yt)) or 1e-300

    for j in range(X.shape[1]):
        xi = X[:, j]
        if np.std(xi) < 1e-14:
            continue
        lo, hi = np.quantile(xi, [q_low, q_high])
        mask_lo = xi <= lo
        mask_hi = xi >= hi
        cname = (
            feature_names[j]
            if feature_names and j < len(feature_names)
            else f"x{j}"
        )
        rec = {"feature": cname, "low_n": int(mask_lo.sum()), "high_n": int(mask_hi.sum())}

        if mask_lo.sum() >= 2:
            yp_lo = yp[mask_lo]
            yt_lo = yt[mask_lo]
            if np.mean(yt_lo > 0) > 0.5 and np.any(yp_lo < 0):
                w = f"⚠ P_asym: {cname} x→min: negatif tahmin, pozitif veri"
                warnings.append(w)
                rec["low_neg_pred"] = True
            rec["low_mean_err"] = float(np.mean(np.abs(yp_lo - yt_lo)))

        if mask_hi.sum() >= 2:
            yp_hi = yp[mask_hi]
            yt_hi = yt[mask_hi]
            err_hi = np.abs(yp_hi - yt_hi)
            if np.any(err_hi > 3.0 * sig_y):
                w = f"⚠ P_asym: {cname} x→max: Asimptotik sapma (>3σ)"
                warnings.append(w)
                rec["high_drift"] = True
            rec["high_max_err"] = float(np.max(err_hi))

        details.append(rec)

    return {"warnings": warnings, "details": details, "n_warnings": len(warnings)}


def auto_dimension_inference(
    data: pd.DataFrame,
    feature_names: list,
    target: str | None = None,
) -> dict:
    """
    Birim tahmini (kullanıcı --units vermediyse).
    UYARI: Dönen dim_map doğrulanamaz; P5 hard-kill için kullanılmamalıdır.
    """
    cols = list(feature_names)
    if target and target not in cols:
        cols = cols + [target]
    suggestions: dict[str, str] = {}
    dim_map: dict[str, list] = {}

    for col in cols:
        if col not in data.columns:
            continue
        v = data[col].values.astype(float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            suggestions[col] = "[Tahmini birim: boyutsuz?]"
            dim_map[col] = list(SI_UNITS["1"])
            continue

        vmin, vmax = float(np.min(v)), float(np.max(v))
        all_pos = bool(np.all(v > 0))
        in_01 = bool(vmin >= 0 and vmax <= 1.0 + 1e-9)
        has_neg = bool(np.any(v < 0))
        all_int = bool(np.allclose(v, np.round(v)))

        if has_neg:
            unit, hint = "1", "diferansiyel veya hata sinyali (boyutsuz?)"
        elif all_int and vmax < 1e6:
            unit, hint = "1", "indeks veya sayı (boyutsuz)"
        elif in_01:
            unit, hint = "1", "boyutsuz veya açı"
        elif all_pos and vmax > 100:
            if vmax > 1e4:
                unit, hint = "N", "kuvvet (N?)"
            else:
                unit, hint = "m/s", "hız veya akış (m/s?)"
        elif all_pos:
            unit, hint = "m", "uzunluk veya genlik (m?)"
        else:
            unit, hint = "1", "boyutsuz?"

        suggestions[col] = f"[Tahmini birim: {hint}]"
        dim_map[col] = list(SI_UNITS.get(unit, SI_UNITS["1"]))

    return {
        "suggestions": suggestions,
        "dim_map": dim_map,
        "inferred": True,
        "warn": "⚠ Tahmini birim — doğrulayın",
    }


def merge_extended_protocols(
    res: dict,
    yp,
    yt,
    data: pd.DataFrame,
    features: list,
    target: str,
) -> None:
    """Sembolik türev, asimptotik ve P16 derin kontrollerini sonuca yaz."""
    if not features:
        return
    Xm = np.column_stack([data[f].values.astype(float) for f in features])
    deriv = symbolic_derivative_check(
        yp, yt, Xm, features, data=data, res=res, target=target,
    )
    asym = asymptotic_behavior_check(yp, yt, Xm, feature_names=features)
    p16d = energy_conservation_deep(yp, yt, Xm)
    res["ext_protocols"] = {
        "P_deriv": deriv,
        "P_asym_ext": asym,
        "P16_deep": p16d,
    }
    v61 = res.get("v61") or {}
    v61["P16_cons"] = p16d["penalty"]
    v61["P_deriv_pen"] = float(deriv["n_conflicts"])
    v61["P_asym_pen"] = float(asym["n_warnings"])
    v61["P_deriv_warns"] = deriv["warnings"]
    v61["P_asym_warns"] = asym["warnings"]
    if p16d.get("warning"):
        v61["P16_warn"] = p16d["warning_message"]
    res["v61"] = v61
    pw = list(res.get("protocol_warnings") or [])
    pw.extend(deriv["warnings"])
    pw.extend(asym["warnings"])
    if p16d.get("warning_message"):
        pw.append(p16d["warning_message"])
    if pw:
        res["protocol_warnings"] = pw


def compute_all_v61_penalties(yp, yt, beta_vals=None, X_vals=None,
                               lam=1.0) -> dict:
    """Tüm v6.1 protokol cezalarını tek sözlükte toplar.
    J_full = MDL + Σ λ_i·C_i
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    bv = beta_vals if beta_vals is not None else []
    p10_pen, p10_fy, p10_ff = p10_periodicity(yp, yt, lam)
    p12_res = p12_even_odd(yp, yt, lam)
    penalties = {
        "P2_mono":    p2_monotonicity(yp, yt, lam),
        "P3_zero":    p3_zero_crossing(yp, yt, lam),
        "P4_asym":    p4_asymptotic_bound(yp, yt, lam),
        "P5_dim":     0.0,   # Hard-Kill — ayrıca p5_hard_kill ile kontrol
        "P7_sign":    p7_coeff_sign(bv, None, lam),
        "P8_bounds":  p8_output_bounds(yp, yt, lam),
        "P9_cont":    p9_continuity(yp, lam),
        "P10_period": p10_pen,
        "P10_freq_y": p10_fy,
        "P10_freq_f": p10_ff,
        "P11_scale":  p11_scale_invariance(yp, yt, X_vals or [], lam),
        "P12_sym":    p12_res["penalty"],
        "P12_type":   p12_res["symmetry"],
        "P13_pos":    p13_positivity(yp, yt, lam),
        "P14_order":  p14_interaction_order(bv, lam),
        "P15_diff":   0.0,   # p15_diff_penalty ile hesaplı
        "P16_cons":   p16_conservation(yp, yt, X_vals, lam),
    }
    # J_full: tüm sayısal cezaların toplamı
    num_pen = [v for k,v in penalties.items()
               if isinstance(v,float) and k not in ("P10_freq_y","P10_freq_f")]
    penalties["J_full"] = sum(num_pen)
    return penalties


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — SAIM-Y ADAPTİF ALAN METRİĞİ  (v7.1)
# TRI pilot analiz → rough/smooth dallanması → adaptif örnekleme
# A(f,D) = (n-1)·MAE·κ(p),  κ(p)=1-3/8·p  [ΠH: 3/8 tam rasyonel]
# LE = ln(MAE_f/MAE_fc) - 11/24·(p_f-p_fc)  [ΠH: 11/24 tam rasyonel]
# ═══════════════════════════════════════════════════════════════════════════════

def _saim_integrate_segment(e0, e1, x0, x1, n_sub=8):
    """Adaptif alt-segment entegrasyonu: |e| eğrisi altındaki alan."""
    xs = np.linspace(x0,x1,n_sub)
    es = np.linspace(e0,e1,n_sub)
    _trapfn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(_trapfn(np.abs(es),xs))

def saim_y_area(yp, yt, x_vals=None, tau_TRI=0.01, verbose=False) -> dict:
    """
    SAIM-Y v6.0 adaptif alan metrigi:
    1. Pilot (N_pilot=51): TRI = Σ|eᵢ| / (f_range·(b-a))
    2. rough (TRI>τ): ağırlıklı örnekleme w(x)=|f''(x)|+ε
    3. smooth: Gauss-Legendre / trapez
    Kapalı form: A ≈ (n-1)·MAE·κ(p),  κ(p)=1-3/8·p
    """
    yp = np.asarray(yp,float); yt = np.asarray(yt,float)
    n  = len(yt)
    if x_vals is None:
        x_vals = np.linspace(0,1,n)
    x_vals = np.asarray(x_vals,float)
    e = yt-yp

    # Pilot TRI
    f_range = float(np.ptp(yp)) or 1.0
    span    = float(x_vals[-1]-x_vals[0]) or 1.0
    TRI     = float(np.sum(np.abs(e)))/(f_range*span)
    rough   = TRI > tau_TRI

    # Adaptif alan hesabı
    A_trap = 0.0; cross = 0
    for i in range(n-1):
        e0,e1 = float(e[i]),float(e[i+1])
        x0,x1 = float(x_vals[i]),float(x_vals[i+1])
        if rough:
            # Ağırlıklı: ikinci türev tahmini
            f2 = abs(e1-e0)/(x1-x0+1e-300)
            n_sub = max(4, min(20, int(f2*10)+4))
            A_trap += _saim_integrate_segment(e0,e1,x0,x1,n_sub)
        else:
            # Gauss-Legendre 5-noktalı kuadratur (smooth segment)
            # ∫_{x0}^{x1} |e(x)| dx  —  e(x) lineer interpolasyon
            # GL-5 düğümleri ve ağırlıkları [-1,1] üzerinde
            _gl_nodes = np.array([-0.9061798459, -0.5384693101, 0.0,
                                    0.5384693101,  0.9061798459])
            _gl_w     = np.array([ 0.2369268851,  0.4786286705, 0.5688888889,
                                    0.4786286705,  0.2369268851])
            half = (x1-x0)/2; mid = (x0+x1)/2
            x_gl = mid + half*_gl_nodes
            # e(x) lineer: e(x) = e0 + (e1-e0)*(x-x0)/(x1-x0)
            e_gl = e0 + (e1-e0)*(x_gl-x0)/(x1-x0+1e-300)
            A_trap += half * float(np.dot(_gl_w, np.abs(e_gl)))
        if e0*e1 < 0: cross += 1

    p     = cross/(n-1) if n>1 else 0.0
    kappa = 1.0-(3/8)*p                    # ΠH kilidi
    mae_v = float(np.mean(np.abs(e)))
    A_approx = (n-1)*mae_v*kappa*(span/(n-1) if n>1 else 1.0)

    if verbose:
        mode = "rough" if rough else "smooth"
        print(f"    [SAIM-Y] TRI={TRI:.4f} ({mode})  κ={kappa:.4f}  A={A_trap:.4g}")

    return {
        "A": A_trap, "A_approx": A_approx,
        "p": p, "kappa": kappa, "mae": mae_v,
        "TRI": TRI, "rough": rough,
    }

def le_metric(yp, yt, yb) -> float:
    """LE = ln(MAE_f/MAE_fc) - 11/24·(p_f-p_fc)  — ΠH: 11/24"""
    def cr(yp_):
        e = np.asarray(yt)-np.asarray(yp_); n = len(e)
        return sum(1 for i in range(n-1) if e[i]*e[i+1]<0)/(n-1+1e-300)
    mae_f  = float(np.mean(np.abs(np.asarray(yt)-np.asarray(yp))))
    mae_fc = float(np.mean(np.abs(np.asarray(yt)-np.asarray(yb)))) or 1e-300
    return math.log(mae_f/mae_fc)-(11/24)*(cr(yp)-cr(yb))


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — SINDy + SVD PREWHİTENİNG  (v7)
# Ẋ = Θ(X)·Ξ  →  STLS + MDL eşik + ΠH
# Zaman sütunu varsa dt ile doğru türev hesabı
# ═══════════════════════════════════════════════════════════════════════════════

def svd_prewhiten(Theta):
    """Θ=U·S·Vᵀ → Θ_w=U·S (κ(Θ_w)=1),  geri: ξ=Vᵀ·ξ_w"""
    U,S,Vt = np.linalg.svd(Theta,full_matrices=False)
    return U*S[np.newaxis,:], Vt, S

def _lad_solve(A, b, max_iter=200, tol=1e-6) -> np.ndarray:
    """
    ST-LAD: L1 (Least Absolute Deviations) çözücü — IRLS yöntemi.
    Outlier'lara karşı dayanıklı; OLS'nin L2 yerine L1 minimizasyonu.
    min_ξ Σ|Aᵢ·ξ - bᵢ|  ←→  IRLS: W = diag(1/(|r|+ε)),  ξ = (AᵀWA)⁻¹Aᵀ Wb
    """
    xi = ols(A, b)   # başlangıç: OLS tahmini
    for _ in range(max_iter):
        r   = b - A @ xi
        w   = 1.0 / (np.abs(r) + 1e-6)          # IRLS ağırlıkları
        W   = np.diag(w)
        AW  = A.T @ W
        try:
            xi_new, _, _, _ = np.linalg.lstsq(AW @ A, AW @ b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(xi_new - xi) < tol:
            xi = xi_new; break
        xi = xi_new
    return xi


def _stls_once(Theta, xd, lam, max_iter=20, use_lad=True):
    """
    STLS bir adımı.
    use_lad=True  → ST-LAD (L1/IRLS) — aykırı değere dayanıklı
    use_lad=False → OLS (L2)          — temiz veri
    """
    xi = _lad_solve(Theta, xd) if use_lad else ols(Theta, xd)
    for _ in range(max_iter):
        mask = np.abs(xi) >= lam
        if not mask.any(): break
        xi_new = np.zeros_like(xi)
        sub    = Theta[:, mask]
        if sub.shape[1]:
            xi_new[mask] = _lad_solve(sub, xd) if use_lad else ols(sub, xd)
        if np.allclose(xi, xi_new): break
        xi = xi_new
    return xi


def stls(Theta_w, xd, lam=None, H=20, max_iter=20, use_lad=True):
    """
    Optimal λ* = argmin_λ MDL(Ξ_λ).
    use_lad: True → ST-LAD (outlier dayanıklı), False → klasik OLS-STLS
    """
    if lam is None:
        best_mdl, best_lam = float("inf"), 1e-3
        for lt in np.logspace(-3, 1, 30):
            xi = _stls_once(Theta_w, xd, lt, max_iter, use_lad)
            s  = mdl_score(Theta_w @ xi, xd, max(1, int(np.sum(xi != 0))), H)
            if s < best_mdl: best_mdl, best_lam = s, lt
        lam = best_lam
    return _stls_once(Theta_w, xd, lam, max_iter, use_lad)

def sindy_discover(data, state_cols, selected_atoms=None,
                   time_col=None, H=20, verbose=True) -> dict:
    """
    SINDy ODE keşfi — v7 tam:
      Zaman sütunu varsa: ẋ = np.gradient(x, t)  (dt düzeltmeli)
      Yoksa: ẋ = np.gradient(x)  (eşit aralık varsayımı)
    SVD prewhitening + STLS (MDL eşik) + ΠH projeksiyon
    """
    if selected_atoms is None:
        selected_atoms = ["x","x2","x3","sin","cos","x_ln_x"]
    n = len(data); equations = {}

    for state in state_cols:
        xv = data[state].values.astype(float)
        if time_col and time_col in data.columns:
            t  = data[time_col].values.astype(float)
            xd = np.gradient(xv, t)          # dt düzeltmeli türev
        else:
            xd = np.gradient(xv)

        tcols,tnames = [],[]
        for col in state_cols:
            v = data[col].values.astype(float)
            for ak in selected_atoms:
                if ak in ATOM_LIBRARY:
                    _,fn,_ = ATOM_LIBRARY[ak]
                    tv = np.nan_to_num(fn(v),nan=0,posinf=1e10,neginf=-1e10)
                    tcols.append(tv); tnames.append(f"{ak}({col})")

        Theta = np.column_stack([np.ones(n)]+tcols)
        cond_b = np.linalg.cond(Theta)
        Tw,Vt,_ = svd_prewhiten(Theta)
        cond_a  = np.linalg.cond(Tw)
        xi_w = stls(Tw, xd, H=H, use_lad=True)   # ST-LAD: outlier dayanıklı
        xi   = Vt.T@xi_w
        xi_rat = [pi_H(v,H) for v in xi]
        xi_rv  = np.array([v for _,v,_ in xi_rat])
        yp = Theta@xi_rv
        all_names = ["intercept"]+tnames
        active = [{"name":all_names[i],"coeff_ols":float(xi[i]),
                   "coeff_piH":xi_rat[i][0],"coeff_val":xi_rat[i][1],
                   "piH_err":xi_rat[i][2]}
                  for i in range(len(xi)) if abs(xi_rv[i])>1e-10]
        xd_range = float(np.ptp(xd)) or 1.0
        nrmse_val = min(float(np.sqrt(np.mean((xd-yp)**2)))/xd_range*100, 1e6)
        # R² için SS_tot sıfıra yakınsa güvenli fallback
        sst_xd = float(np.sum((xd - xd.mean())**2))
        if sst_xd < 1e-12:
            r2_sindy = 0.0
        else:
            r2_sindy = float(np.clip(1 - np.sum((xd-yp)**2)/sst_xd, -10.0, 1.0))
        equations[state] = {
            "x_dot":xd,"y_pred":yp,"active_terms":active,
            "r2":r2_sindy,
            "mape":nrmse_val,"mape_label":"nRMSE%",
            "mdl":mdl_score(yp,xd,max(1,len(active)),H),
            "kappa_before":cond_b,"kappa_after":cond_a,
        }
        if verbose:
            terms = " + ".join(f"({t['coeff_piH']})·{t['name']}" for t in active)
            print(f"  d{state}/dt = {terms or '≈ 0'}")
    return equations


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — ASİMPTOTİK KİLİTLER  (v8.x)
# y* = ln(y/n)        [α=1 kilidi]
# y** = ln(y/(n·ln n)) [α=1, β=1 kilidi]
# ═══════════════════════════════════════════════════════════════════════════════

def asymptotic_transform(n_vals, y_vals, mode="y_star"):
    n   = np.asarray(n_vals,float)
    y   = np.asarray(y_vals,float)
    ln_n = np.log(np.clip(n,2,None))
    cond_b = np.linalg.cond(np.column_stack([np.ones_like(n),np.log(n+1e-300)]))
    if mode=="y_star":
        yt = np.log(np.abs(y)/(n+1e-300)+1e-300)
        X  = np.column_stack([np.ones_like(n), 1.0/ln_n])
    else:  # y_star2: α=1,β=1
        yt = np.log(np.abs(y)/(n*ln_n+1e-300)+1e-300)
        X  = np.column_stack([np.ones_like(n), 1.0/ln_n,
                               np.log(np.log(np.clip(n,3,None)))/ln_n, 1.0/ln_n**2])
    return yt, X, cond_b, np.linalg.cond(X)


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — ÖZELLİK MATRİSİ OLUŞTURMA
# Tekli atomlar + çapraz etkileşim atomları
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(data, feature_atom_pairs) -> tuple:
    """
    feature_atom_pairs:
      Tekli  : [(col, atom_key), ...]          atom_key ∈ ATOM_LIBRARY
      Çapraz : [(col_a, col_b, cross_key), ...]  cross_key ∈ CROSS_ATOM_LIBRARY
    Döndürür: (X matris, sütun adları)
    """
    n = len(data)
    cols  = [np.ones(n)]
    names = ["intercept"]
    for item in feature_atom_pairs:
        if len(item)==2:
            col,ak = item
            if ak in ATOM_LIBRARY:
                _,fn,_ = ATOM_LIBRARY[ak]
                v = fn(data[col].values.astype(float))
                cols.append(np.nan_to_num(v,nan=0,posinf=1e10,neginf=-1e10))
                names.append(f"{ak}({col})")
            elif ak in CROSS_ATOM_LIBRARY:
                # Tek sütunlu çapraz: col ile col kendisi (kare vb.)
                _,fn,_ = CROSS_ATOM_LIBRARY[ak]
                v = fn(data[col].values.astype(float),
                       data[col].values.astype(float))
                cols.append(np.nan_to_num(v,nan=0,posinf=1e10,neginf=-1e10))
                names.append(f"{ak}({col},{col})")
        elif len(item)==3:
            col_a,col_b,ck = item
            if ck in CROSS_ATOM_LIBRARY:
                _,fn,_ = CROSS_ATOM_LIBRARY[ck]
                va = data[col_a].values.astype(float)
                vb = data[col_b].values.astype(float)
                v  = fn(va,vb)
                cols.append(np.nan_to_num(v,nan=0,posinf=1e10,neginf=-1e10))
                names.append(f"{ck}({col_a},{col_b})")
    return np.column_stack(cols), names


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12 — ANA MODEL FİT  (tüm protokoller entegre)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_piH(beta_ols, H):
    ls,vs,es = [],[],[]
    for b in beta_ols:
        l,v,e = pi_H(b,H,algebraic=True)
        ls.append(l); vs.append(v); es.append(e)
    return ls, np.array(vs), es

def _base_metrics_fast(yp, yt, k, H):
    """GUI / hızlı mod — yalnızca temel uyum ölçütleri."""
    yp, yt = np.asarray(yp), np.asarray(yt)
    err = error_metrics(yp, yt)
    return {
        "r2": r2(yp, yt), **err,
        "mdl": mdl_score(yp, yt, k, H), "sr_cf": sr_closed_form(yp, yt),
        "sr_taylor": None, "area": {}, "A_inf": None, "le": None,
        "p15": 0.0, "cax": 0.0, "v61": {}, "J_score": None,
        "J_full": None, "mdlstar": None, "_fast": True,
    }


def _base_metrics(yp, yt, k, H, x_vals=None):
    yp,yt = np.asarray(yp),np.asarray(yt)
    yb = np.full(len(yt),yt.mean())
    am  = saim_y_area(yp, yt, x_vals)
    le  = le_metric(yp, yt, yb)
    sr  = sr_closed_form(yp, yt)
    mdl = mdl_score(yp, yt, k, H)
    # P15 + Cax + tüm v6.1 protokoller
    p15  = p15_diff_penalty(yp,yt)
    cax  = cax_penalty(yp)
    v61  = compute_all_v61_penalties(yp, yt, beta_vals=None, X_vals=x_vals)
    v61["P15_diff"] = p15
    v61["Cax"]      = cax
    ms  = None
    if len(yt)>=6:
        h = len(yt)//2
        ms = mdl_star(yp[:h],yt[:h],yp[h:],yt[h:],k_hat=k)
    # SR Taylor asimptotu (ε/σ→0): SR ≈ 39/49·(ε/σ)   [ΠH: 39/49=√(2/π)]
    m   = mad(yt); eps = 0.4*(m if m>0 else 1.0)
    sig = float(np.std(yt-yp)) or 1.0
    sr_taylor = (39/49)*(eps/sig)          # 1. derece Taylor; 39/49≈ΠH(√(2/π))
    sr_taylor = float(np.clip(sr_taylor,0,1))

    # A∞ büyük-n asimptotu: Normal dağılım, p→1/2 → A∞=(n-1)·Δx·MAE·13/16
    mae_v = am["mae"]
    n_    = len(yt)
    A_inf = (n_-1)*mae_v*(13/16)           # ΠH kilidi: 13/16

    # J(f) birleşik enerji skoru (v6.1)
    # J(f) = L(f) + λax·Cax + λdiff·Cdiff + λdim·Cdim
    lam_ax=1.0; lam_diff=1.0
    J_score = mdl + lam_ax*cax + lam_diff*p15

    # J_full: MDL + tüm v6.1 cezaları
    J_full = mdl + v61["J_full"]
    err = error_metrics(yp, yt)
    return {"r2": r2(yp, yt), **err, "mdl": mdl, "sr_cf": sr, "sr_taylor": sr_taylor,
            "area":am,"A_inf":A_inf,"le":le,
            "p15":p15,"cax":cax,"J_score":J_score,
            "v61":v61,"J_full":J_full,"mdlstar":ms}

def compute_vif(data, features: list, max_rows: int = MAX_ANALYSIS_ROWS) -> dict:
    """
    Variance Inflation Factor — log-uzayında (güç yasası ile uyumlu).
    VIF_i = 1 / (1 - R²_i),  R²_i: X_i ~ diğer X'ler.
    """
    if not features:
        return {}
    if len(features) == 1:
        return {features[0]: 1.0}
    if len(features) > 40:
        features = features[:40]
    data = analysis_sample(data, max_rows)
    n = len(data)
    log_cols = {
        f: np.log(np.abs(data[f].values.astype(float)) + 1e-300)
        for f in features
    }
    vifs: dict = {}
    for f in features:
        y_col = log_cols[f]
        others = [log_cols[o] for o in features if o != f]
        X = np.column_stack([np.ones(n)] + others)
        try:
            b = ols(X, y_col)
            yp = X @ b
            ss_res = float(np.sum((y_col - yp) ** 2))
            ss_tot = float(np.sum((y_col - y_col.mean()) ** 2))
            if ss_tot < 1e-300:
                r2 = 0.0
            else:
                r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0 - 1e-12))
            vif = 1.0 / max(1.0 - r2, 1e-12)
            vifs[f] = float(vif) if np.isfinite(vif) else float("inf")
        except Exception:
            vifs[f] = float("inf")
    return vifs


def validate_power_law_result(res: dict, exp_limit: float = 50.0) -> str | None:
    """inf/nan veya aşırı üs varsa hata mesajı döner."""
    if res.get("model_type") != "power_law":
        return None
    msg = "Model patladı — değişken sayısını azalt veya bağımlı değişkenleri çıkar"
    for key in ("beta_ols", "y_pred"):
        for x in res.get(key, []):
            if not np.isfinite(float(x)):
                return msg
    for _, _, exp in res.get("exponents", []):
        if not np.isfinite(float(exp)) or abs(float(exp)) > exp_limit:
            return msg
    for _, val, _ in res.get("beta_rat", []):
        if not np.isfinite(float(val)):
            return msg
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12b — FİZİK SABİTLERİ, BOYUT, BOOTSTRAP, CV, ÇOKLU HEDEF
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICAL_CONSTANTS: dict[str, float] = {
    "g": 9.807,
    "2π/√g": 2.006,
    "1/g": 0.102,
    "G": 6.674e-11,
    "c": 3.0e8,
    "h": 6.626e-34,
    "k_B": 1.38e-23,
    "R": 8.314,
    "N_A": 6.022e23,
    "π": math.pi,
    "2π": 2.0 * math.pi,
    "√2": math.sqrt(2),
    "√3": math.sqrt(3),
    "φ": 1.6180339887,
    "e": math.e,
}

PHYS_CONST_DISPLAY: dict[str, str] = {
    "g": "g (yerçekimi)",
    "2π/√g": "2π/√g",
    "1/g": "1/g",
    "G": "G (Newton çekim sabiti)",
    "c": "c (ışık hızı)",
    "h": "h (Planck sabiti)",
    "k_B": "k_B (Boltzmann)",
    "R": "R (evrensel gaz sabiti)",
    "N_A": "N_A (Avogadro)",
    "π": "π",
    "2π": "2π",
    "√2": "√2",
    "√3": "√3",
    "φ": "φ (altın oran)",
    "e": "e (Euler)",
}

# SI [M, L, T, I, Θ, N, J] — ycpa_p P5 vektörü
SI_UNITS: dict[str, list] = {
    "m": [0, 1, 0, 0, 0, 0, 0],
    "kg": [1, 0, 0, 0, 0, 0, 0],
    "s": [0, 0, 1, 0, 0, 0, 0],
    "N": [1, 1, -2, 0, 0, 0, 0],
    "Pa": [1, -1, -2, 0, 0, 0, 0],
    "J": [1, 2, -2, 0, 0, 0, 0],
    "W": [1, 2, -3, 0, 0, 0, 0],
    "A": [0, 0, 0, 1, 0, 0, 0],
    "K": [0, 0, 0, 0, 1, 0, 0],
    "m/s": [0, 1, -1, 0, 0, 0, 0],
    "m/s2": [0, 1, -2, 0, 0, 0, 0],
    "m/s^2": [0, 1, -2, 0, 0, 0, 0],
    "1": [0, 0, 0, 0, 0, 0, 0],
}


def identify_physical_constant(C: float, tol: float = 0.02) -> str | None:
    """Compare C against PHYSICAL_CONSTANTS; return symbol if relative diff < tol."""
    if not np.isfinite(C):
        return None
    best_sym, best_rel = None, float("inf")
    for sym, ref in PHYSICAL_CONSTANTS.items():
        if ref == 0:
            continue
        rel = abs(float(C) - ref) / max(abs(ref), 1e-30)
        if rel < best_rel:
            best_rel, best_sym = rel, sym
    return best_sym if best_rel <= tol else None


def phys_const_note(sym: str | None) -> str:
    if not sym:
        return ""
    return f" → {PHYS_CONST_DISPLAY.get(sym, sym)}"


def _form_complexity(lbl: str) -> int:
    """Düşük = daha sade (p+q + cebirsel çarpan cezası)."""
    s = str(lbl)
    c = 2 if "·" in s else 0
    for ch in ("π", "φ", "ψ", "√", "e"):
        if ch in s:
            c += 2
    try:
        fr = Fraction(s.split("·")[0])
        c += fr.numerator + fr.denominator
    except Exception:
        c += len(s.replace("·", ""))
    return c


def simplify_equivalent(C: float, tol: float = 0.001, H: int = 20) -> dict:
    """
  Bulunan katsayının eşdeğer cebirsel formlarını üret; en sade olanı seç.
    """
    if not np.isfinite(C):
        return {"best": str(C), "alts": [], "value": C}
    candidates: list[tuple[str, float, float, int]] = []

    def _add(lbl: str, val: float):
        err = abs(C - val)
        if err <= max(tol, abs(C) * tol, 1e-12):
            candidates.append((lbl, val, err, _form_complexity(lbl)))

    lbl0, v0, _ = pi_H(C, H, algebraic=False)
    _add(lbl0, v0)
    for sym, alpha in _ALG_CONSTS.items():
        if abs(alpha) < 1e-12:
            continue
        for q in range(1, H + 1):
            p = round((C / alpha) * q)
            for dp in (-1, 0, 1):
                val = ((p + dp) / q) * alpha
                _add(f"{Fraction(p + dp, q)}·{sym}", val)

    if not candidates:
        return {"best": f"{C:.6g}", "alts": [], "value": C}
    candidates.sort(key=lambda x: (x[3], x[2]))
    best = candidates[0]
    alts = [c[0] for c in candidates[1:6] if c[0] != best[0]]
    return {"best": best[0], "alts": alts, "value": best[1], "err": best[2]}


def parse_units_cli(spec: str) -> tuple[list | None, dict]:
    """
    CLI: --units T=s a=m  veya  T=s,L=m
    Sütun adı → SI vektörü.
    """
    if not spec or not spec.strip():
        return None, {}
    dim_map: dict = {}
    dim_lhs = None
    for token in re.split(r"[,;\s]+", spec.strip()):
        if "=" not in token:
            continue
        col, u = token.split("=", 1)
        col, u = col.strip(), u.strip()
        vec = SI_UNITS.get(u) or SI_UNITS.get(u.replace("²", "2").replace("^2", "2"))
        if vec is None:
            print(f"  ⚠  Unknown unit '{u}' — skipped ({col})")
            continue
        dim_map[col] = list(vec)
    return dim_lhs, dim_map


def p5_power_law_dimension_check(
    dim_lhs: list | None,
    features: list,
    exponents: list,
    dim_map: dict | None,
    atol: float = 1e-5,
) -> tuple[bool, str | None]:
    """
    Güç yasası: Σ(βᵢ·birim_i) == hedef_birimi  (C boyutsuz varsayımı).
    """
    if not dim_lhs or not dim_map:
        return True, None
    lhs = np.array(dim_lhs, dtype=float)
    rhs = np.zeros_like(lhs)
    missing = []
    for feat, _, beta in exponents:
        if abs(float(beta)) < 1e-10:
            continue
        if feat not in dim_map:
            missing.append(feat)
            continue
        rhs += float(beta) * np.array(dim_map[feat], dtype=float)
    if missing:
        return True, None
    if np.allclose(rhs, lhs, atol=atol):
        return True, None
    return False, "⚠ P5: Boyutsal tutarsızlık — Σ(üs×birim) ≠ hedef birimi"


def residual_analysis(yp, yt) -> dict:
    """Shapiro-Wilk, Breusch-Pagan benzeri, Durbin-Watson."""
    yp = np.asarray(yp, dtype=float).ravel()
    yt = np.asarray(yt, dtype=float).ravel()
    e = yt - yp
    n = len(e)
    out: dict = {"warnings": [], "shapiro_p": None, "bp_r": None, "durbin_watson": None}
    if n < 4:
        return out
    try:
        if n <= 5000:
            _, p_sw = sp_stats.shapiro(e)
            out["shapiro_p"] = float(p_sw)
            if p_sw < 0.05:
                out["warnings"].append("⚠ Hata normal değil (Shapiro-Wilk p<0.05)")
    except Exception:
        pass
    if n >= 3:
        e2 = e ** 2
        r_bp = float(np.corrcoef(yp, e2)[0, 1]) if np.std(yp) > 1e-14 else 0.0
        out["bp_r"] = r_bp
        if abs(r_bp) > 0.3:
            out["warnings"].append("⚠ Heteroskedastisite (|r(ŷ,e²)|>0.3)")
    if n >= 2:
        de = np.diff(e)
        num = float(np.sum(de ** 2))
        den = float(np.sum(e ** 2)) or 1.0
        d = num / den
        out["durbin_watson"] = d
        if d < 1.5:
            out["warnings"].append("⚠ Pozitif otokorelasyon (Durbin-Watson<1.5)")
    return out


def model_uncertainty(results: list) -> str:
    """MDL farkına göre model güven etiketi."""
    if not results:
        return ""
    if len(results) < 2:
        return "✓ Güvenilir (tek model)"
    d = float(results[1].get("mdl", 0)) - float(results[0].get("mdl", 0))
    if d < 0.5:
        return "⚠ Belirsiz — iki model neredeyse eşit (ΔMDL<0.5)"
    if d < 2.0:
        return "△ Orta güven (ΔMDL<2)"
    return "✓ Güvenilir (ΔMDL>2)"


def _power_predict(data, target, features, beta) -> np.ndarray:
    yr = data[target].values.astype(float)
    X = np.column_stack(
        [np.ones(len(yr))]
        + [np.log(np.abs(data[f].values.astype(float)) + 1e-300) for f in features]
    )
    return np.exp(X @ np.asarray(beta, dtype=float))


def noise_quality_check(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    warnings = []

    signal_power = float(np.var(y_true))
    noise_power = float(np.var(residuals)) or 1e-300
    snr_db = 10 * math.log10(signal_power / noise_power)
    if snr_db < 10:
        warnings.append(
            f"⚠ Düşük SNR ({snr_db:.1f} dB) — ölçüm gürültüsü yüksek olabilir"
        )

    dw = float(np.sum(np.diff(residuals) ** 2)) / (
        float(np.sum(residuals ** 2)) + 1e-300
    )
    if dw < 1.0 or dw > 3.0:
        warnings.append(
            f"⚠ Kalıntı otokorelasyonu (DW={dw:.2f}) — "
            "veri bağımsız değil ya da önemli değişken eksik"
        )

    n = len(residuals)
    shapiro_p = None
    if 4 <= n <= 5000:
        try:
            _, shapiro_p = sp_stats.shapiro(residuals[:5000])
            if shapiro_p < 0.01:
                warnings.append(
                    f"⚠ Kalıntılar normal dağılmıyor "
                    f"(Shapiro p={shapiro_p:.3f})"
                )
        except Exception:
            pass

    ratio = None
    if n >= 10:
        mid = n // 2
        idx = np.argsort(y_pred)
        std_lo = float(np.std(residuals[idx[:mid]]))
        std_hi = float(np.std(residuals[idx[mid:]]))
        ratio = max(std_lo, std_hi) / (min(std_lo, std_hi) + 1e-300)
        if ratio > 3.0:
            warnings.append(
                f"⚠ Heterokedastislik (oran={ratio:.1f}) — "
                "hata varyansı sabit değil"
            )

    return {
        "warnings": warnings,
        "snr_db": snr_db,
        "dw": dw,
        "shapiro_p": shapiro_p,
        "heteroscedasticity_ratio": ratio,
    }


def bootstrap_ci(
    data,
    target,
    features,
    result: dict,
    n_boot: int = 200,
    ci: float = 0.95,
    H: int = 20,
    fast: bool = True,
) -> dict | None:
    """Yerine koyarak bootstrap; katsayı güven aralıkları."""
    mt = result.get("model_type")
    if mt != "power_law":
        return None
    data = analysis_sample(data)
    n = len(data)
    if n < 6:
        return None
    rng = np.random.default_rng(42)
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_b = data.iloc[idx].reset_index(drop=True)
        try:
            r = fit_power_law(
                d_b, target, features, H=H, use_mcmc=False, fast=fast,
                p5_hard_kill=False,
            )
            if r and r.get("beta_ols"):
                coefs.append(r["beta_ols"])
        except Exception:
            pass
    if len(coefs) < max(10, n_boot // 10):
        return None
    arr = np.array(coefs)
    lo_p = (1.0 - ci) / 2.0 * 100.0
    hi_p = (1.0 + ci) / 2.0 * 100.0
    pnames = ["ln(C)"] + [f"exp({f})" for f in features]
    intervals = []
    for i in range(arr.shape[1]):
        intervals.append({
            "name": pnames[i] if i < len(pnames) else f"β{i}",
            "low": float(np.percentile(arr[:, i], lo_p)),
            "high": float(np.percentile(arr[:, i], hi_p)),
        })
    return {"ci": ci, "n_ok": len(coefs), "intervals": intervals}


def kfold_cv(
    data,
    target,
    features,
    model_type: str = "power",
    k: int = 5,
    H: int = 20,
    fast: bool = True,
    unit_lhs=None,
    unit_map=None,
) -> dict | None:
    """k-fold çapraz doğrulama — R², MAPE, RMSE ortalama ± std."""
    data = analysis_sample(data)
    n = len(data)
    if n < k * 3:
        return None
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    r2s, mapes, rmses = [], [], []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        d_tr = data.iloc[tr].reset_index(drop=True)
        d_te = data.iloc[te].reset_index(drop=True)
        yt = d_te[target].values.astype(float)
        try:
            if model_type == "power":
                res = fit_power_law(
                    d_tr, target, features, H=H, use_mcmc=False, fast=fast,
                    dim_lhs=unit_lhs, dim_map=unit_map, p5_hard_kill=False,
                )
                if not res:
                    continue
                yp = _power_predict(d_te, target, features, res["beta_ols"])
            else:
                continue
            err = error_metrics(yp, yt)
            r2s.append(r2(yp, yt))
            mapes.append(err["mape"])
            rmses.append(rmse(yp, yt))
        except Exception:
            continue
    if not r2s:
        return None
    return {
        "k": k,
        "CV_R2_mean": float(np.mean(r2s)),
        "CV_R2_std": float(np.std(r2s)),
        "CV_MAPE_mean": float(np.mean(mapes)),
        "CV_MAPE_std": float(np.std(mapes)),
        "CV_RMSE_mean": float(np.mean(rmses)),
        "CV_RMSE_std": float(np.std(rmses)),
    }


def _exponents_match(ex1: list, ex2: list, tol: float = 0.05) -> bool:
    if len(ex1) != len(ex2):
        return False
    for (f1, _, v1), (f2, _, v2) in zip(ex1, ex2):
        if f1 != f2 or abs(float(v1) - float(v2)) > tol:
            return False
    return True


def fit_multi_target(
    data,
    targets: list,
    features: list,
    H: int = 20,
    use_mcmc: bool = False,
    fast: bool = False,
    dim_map: dict | None = None,
    exp_tol: float = 0.05,
) -> dict:
    """
    Her hedef için güç yasası; ortak üsler varsa paylaştırılmış rapor.
    """
    fits: dict = {}
    for t in targets:
        dim_lhs = dim_map.get(t) if dim_map else None
        umap = {k: v for k, v in (dim_map or {}).items() if k != t}
        fits[t] = fit_power_law(
            data, t, features, H=H, use_mcmc=use_mcmc, fast=fast,
            dim_lhs=dim_lhs, dim_map=umap, p5_hard_kill=False,
        )
    first_fit = fits.get(targets[0])
    if first_fit is None:
        return {
            "model_type": "multi_target",
            "targets": targets,
            "features": features,
            "fits": fits,
            "shared_exponents": False,
            "H": H,
        }
    shared = True
    ref = first_fit.get("exponents", [])
    for tgt in targets[1:]:
        ft = fits.get(tgt)
        if ft is None or not _exponents_match(ref, ft.get("exponents", []), exp_tol):
            shared = False
            break
    return {
        "model_type": "multi_target",
        "targets": targets,
        "features": features,
        "fits": fits,
        "shared_exponents": shared,
        "H": H,
    }


def print_multi_target(mt: dict):
    """Çoklu hedef güç yasası raporu."""
    sep()
    print("  MULTI-TARGET POWER LAW")
    if mt.get("shared_exponents"):
        print("  Shared exponents detected — only C constants differ")
    else:
        print("  ⚠ Exponents differ across targets")
    sep2()
    targets = mt["targets"]
    for t in targets:
        res = mt["fits"].get(t)
        if res is None:
            print(f"  {t} = (fit failed / eliminated by P5)")
            continue
        C = res.get("C", 1)
        idc = res.get("id_phys_const") or identify_physical_constant(C)
        es = " × ".join(f"{f}^({l})" for f, l, _ in res.get("exponents", []))
        print(f"  {t} = [{res.get('C_piH', C)} = {C:.6g}]{phys_const_note(idc)} × {es}")
    print()


def fit_power_law(data, target, features, H=20,
                  use_mcmc=True, verbose=False, fast=False,
                  dim_lhs=None, dim_map=None,
                  p5_hard_kill: bool = True,
                  allow_unit_inference: bool = True) -> dict | None:
    """
    Y = C·x₁^a·x₂^b·...
    dim_lhs  : hedef büyüklüğün SI vektörü (P5 için)
    dim_map  : {sütun: SI_vektör} (P5 için)
    allow_unit_inference : --units yoksa auto_dimension_inference ile P5
    """
    H = max(int(H), 1)
    unit_inferred = False
    unit_suggestions: dict = {}
    if allow_unit_inference and not dim_map:
        inf = auto_dimension_inference(data, list(features), target)
        dim_map = inf["dim_map"]
        dim_lhs = dim_map.get(target)
        unit_inferred = True
        unit_suggestions = inf["suggestions"]
    elif dim_map and dim_lhs is None:
        dim_lhs = dim_map.get(target)

    if len(data) > MAX_ANALYSIS_ROWS:
        data = analysis_sample(data)
    yr = data[target].values.astype(float)
    yl = np.log(np.abs(yr)+1e-300)
    X  = np.column_stack([np.ones(len(yr))]+
                         [np.log(np.abs(data[f].values.astype(float))+1e-300)
                          for f in features])
    bo = ols(X,yl)

    dim_warn = None

    mc = None
    if use_mcmc and not fast and len(yr)>=4:
        mc = tournament_mcmc(X,yl,verbose=verbose)
        bb = (mc["beta_mcmc"]
              if mdl_score(X@mc["beta_mcmc"],yl,len(bo),H) < mdl_score(X@bo,yl,len(bo),H)
              else bo)
    else:
        bb = bo

    labs,br,errs = _apply_piH(bb,H)
    yp_log = X@br; yp = np.exp(yp_log)

    if mc is not None:
        yh = sigmoid_hybrid(np.exp(X@bo), np.exp(X@mc["beta_mcmc"]))
        if mdl_score(yh,yr,len(br),H)<mdl_score(yp,yr,len(br),H): yp=yh

    # Asimptotik kilit (hızlı modda atlanır)
    asym = None
    if not fast:
        for mode in ("y_star","y_star2"):
            try:
                _,_,cb,ca = asymptotic_transform(np.arange(1,len(yr)+1,dtype=float),yr,mode)
                asym = {"mode":mode,"cond_before":cb,"cond_after":ca,"improvement":cb/max(ca,1)}
                break
            except: pass

    C = float(np.exp(br[0]))
    if not np.isfinite(C):
        C = float("nan")

    exponents = [(features[i], labs[i + 1], float(br[i + 1])) for i in range(len(features))]
    ok_dim, p5_msg = p5_power_law_dimension_check(dim_lhs, features, exponents, dim_map)
    if not ok_dim and p5_msg:
        dim_warn = p5_msg
        if unit_inferred:
            dim_warn = f"{p5_msg}  ⚠ Tahmini birim — doğrulayın"
        if p5_hard_kill and not unit_inferred:
            if verbose:
                print(f"  {dim_warn} — model elendi (P5 Hard-Kill)")
            return None
        if p5_hard_kill and unit_inferred:
            dim_warn = (dim_warn or "") + "  ⚠ (tahmini birimle P5 çalışmaz — birim girin)"
    elif unit_inferred:
        dim_warn = "⚠ Tahmini birim — doğrulayın (P5 otomatik tahmin)"

    id_phys = identify_physical_constant(C)
    simp = simplify_equivalent(C, tol=0.001, H=H)
    c_disp = simp["best"]
    if simp.get("alts"):
        c_disp = f"{c_disp}  ({', '.join(simp['alts'][:3])})"

    metrics = _base_metrics_fast(yp, yr, len(br), H) if fast else _base_metrics(yp, yr, len(br), H)
    res = {
        "model_type": "power_law", "C": C, "C_piH": labs[0],
        "C_piH_simple": c_disp, "id_phys_const": id_phys, "simplify": simp,
        "exponents": exponents,
        "features": features, "beta_ols": bo.tolist(),
        "beta_rat": list(zip(labs, br.tolist(), errs)),
        "y_pred": yp.tolist(), "n": len(yr), "k": len(br), "H": H,
        "mcmc": mc, "asym": asym, "dim_warn": dim_warn,
        "unit_inferred": unit_inferred,
        "unit_suggestions": unit_suggestions,
        "dim_map_used": dim_map,
        "vif": compute_vif(data, features) if len(features) >= 1 else {},
        **metrics,
    }
    if not fast:
        merge_extended_protocols(res, yp, yr, data, features, target)
        res["residual"] = residual_analysis(yp, yr)
        try:
            res["bootstrap_ci"] = bootstrap_ci(
                data, target, features, res,
                n_boot=200, H=H, fast=True,
            )
        except Exception:
            res["bootstrap_ci"] = None
        if len(data) >= 30:
            try:
                res["cv_stats"] = kfold_cv(
                    data, target, features,
                    model_type="power", k=5, H=H, fast=True,
                    unit_lhs=dim_lhs, unit_map=dim_map,
                )
            except Exception:
                res["cv_stats"] = None
        try:
            res["noise_check"] = noise_quality_check(yp, yr)
            nw = res["noise_check"].get("warnings", [])
            if nw:
                existing = list(res.get("protocol_warnings") or [])
                res["protocol_warnings"] = existing + nw
        except Exception:
            res["noise_check"] = None
    return res

def fit_atom_model(data, target, feature_atom_pairs,
                   H=20, use_mcmc=True, verbose=False, fast=False) -> dict:
    """Y = β₀ + Σ βᵢ·atomᵢ(xᵢ)  — tekli + çapraz atomlar"""
    if len(data) > MAX_ANALYSIS_ROWS:
        data = analysis_sample(data)
    y = data[target].values.astype(float)
    X,col_names = build_feature_matrix(data, feature_atom_pairs)
    bo = ols(X,y)

    mc = None
    if use_mcmc and not fast and len(y)>=4:
        mc = tournament_mcmc(X,y,verbose=verbose)
        bb = (mc["beta_mcmc"]
              if mdl_score(X@mc["beta_mcmc"],y,len(bo),H) < mdl_score(X@bo,y,len(bo),H)
              else bo)
    else:
        bb = bo

    labs,br,errs = _apply_piH(bb,H)
    yp = X@br

    if mc is not None:
        yh = sigmoid_hybrid(X@bo, X@mc["beta_mcmc"])
        if mdl_score(yh,y,len(br),H)<mdl_score(yp,y,len(br),H): yp=yh

    metrics = _base_metrics_fast(yp, y, len(br), H) if fast else _base_metrics(yp, y, len(br), H)
    b0 = float(bo[0]) if len(bo) else 0.0
    res = {
        "model_type": "atom", "feature_atom_pairs": feature_atom_pairs,
        "col_names": col_names,
        "beta_ols": bo.tolist(), "beta_rat": list(zip(labs, br.tolist(), errs)),
        "y_pred": yp.tolist(), "n": len(y), "k": len(br), "H": H,
        "id_phys_const": identify_physical_constant(b0),
        "mcmc": mc, **metrics,
    }
    if not fast:
        feat_cols: list[str] = []
        for item in feature_atom_pairs:
            if len(item) >= 2:
                feat_cols.append(item[0])
            if len(item) >= 3:
                feat_cols.append(item[1])
        feat_cols = list(dict.fromkeys(feat_cols))
        if feat_cols:
            merge_extended_protocols(res, yp, y, data, feat_cols, target)
        res["residual"] = residual_analysis(yp, y)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 13 — UZAY TARAMASI  (tekli + çapraz)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_atom_space(data, target, features, selected_atoms,
                    selected_cross=None, H=20, max_terms=3,
                    use_mcmc=False, verbose=True, fast=False,
                    max_combos=8000, progress_cb=None) -> list:
    """
    selected_atoms : ATOM_LIBRARY'den seçilenler
    selected_cross : CROSS_ATOM_LIBRARY'den seçilenler (None=hariç)
    Kombinasyon: tekli çiftler + çapraz üçlüler
    """
    H = max(int(H), 1)
    if len(data) > MAX_ANALYSIS_ROWS:
        data = analysis_sample(data)
    y = data[target].values.astype(float)

    # Tekli adaylar
    single_cands = [(f,a) for f in features for a in selected_atoms]

    # Çapraz adaylar
    cross_cands = []
    if selected_cross:
        for i,fa in enumerate(features):
            for fb in features[i:]:
                for ck in selected_cross:
                    cross_cands.append((fa,fb,ck))

    all_cands = single_cands + cross_cands
    if not all_cands: return []

    total = sum(math.comb(len(all_cands),r)
                for r in range(1,min(max_terms,len(all_cands))+1))
    if verbose: print(f"  Aday kombinasyon: ~{total}")
    cap = 3000 if fast else 10000
    if total > cap:
        if verbose: print(f"  ⚠  Large space — reduced to max_terms=2")
        max_terms = min(max_terms, 2)

    results = []; done = 0
    limit = min(max_combos, 1200 if fast else max_combos)
    _seen_hashes: set = set()   # SHA-256 önbellek — ziyaret edilen denklemleri tekrar hesaplama

    def _combo_hash(combo):
        """Kombinasyonun deterministik SHA-256 hash'i."""
        key = str(sorted(str(c) for c in combo)).encode()
        return hashlib.sha256(key).hexdigest()

    for r in range(1,min(max_terms,len(all_cands))+1):
        for combo in itertools.combinations(all_cands,r):
            if len(set(map(tuple,combo)))<len(combo): continue
            h = _combo_hash(combo)
            if h in _seen_hashes: continue   # cache hit — atla
            _seen_hashes.add(h)
            try:
                res = fit_atom_model(
                    data, target, list(combo), H=H, use_mcmc=False, fast=fast
                )
                res["combo_hash"] = h
                results.append(res)
            except: pass
            done += 1
            if progress_cb and done % 25 == 0:
                try:
                    progress_cb(done, min(total, limit))
                except Exception:
                    pass
            if done >= limit:
                if verbose: print(f"  Scan limit ({limit}) — stopped.          ")
                break
            if verbose and done%500==0: print(f"  ... {done} scanned (cache={len(_seen_hashes)})",end="\r")
        if done >= limit:
            break

    if verbose: print(f"  {done} combinations scanned, {len(_seen_hashes)} unique hashes.          ")
    results.sort(key=lambda r: r["mdl"])

    # ── İYİLEŞTİRME: artık-temelli yeniden sıralama (terim kaçırma + biçim karışmasını azaltır)
    # MDL tek başına bazen (a) gerçek çok-terimli formülü tek terime indirger,
    # (b) exp'i yüksek polinomla karıştırır. En iyi adayları R² + parsimony dengesiyle
    # yeniden tartarak daha doğru biçim seçeriz. Mevcut MDL sıralaması taban kalır;
    # bu yalnızca ilk birkaç adayı rafine eder, davranışı geriye-uyumlu tutar.
    if len(results) >= 2:
        def _quality(r):
            r2 = r.get("R2") or r.get("r2") or 0.0
            k = r.get("k", 1) or 1
            n = r.get("n", 0) or 0
            # Düzeltilmiş R² (adjusted R²): fazladan terim, açıkladığı varyanstan
            # daha az katkı sağlıyorsa skoru DÜŞÜRÜR. Gürültüyü açıklamak için
            # eklenen sahte terimler (overfitting) burada kendiliğinden elenir,
            # gerçek terimler (varyansı gerçekten azaltanlar) korunur.
            if n > k + 1:
                adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
            else:
                adj = r2
            return -(math.log(max(1.0 - adj, 1e-12)) - 0.3 * math.log(k + 1))
        top = results[: min(8, len(results))]
        top.sort(key=_quality, reverse=True)
        results = top + results[min(8, len(results)) :]

    # Sıfıra-kilitlenen terimleri (ΠH ile 0'a projekte edilmiş) raporlamadan ele.
    # Motor doğru katsayıyı zaten 0 yapıyor; bu yalnızca sunulan formülü sadeleştirir
    # ("sin + 0·cos" → "sin") ve gerçek sparsity'yi yansıtır.
    for res in results[: min(8, len(results))]:
        br = res.get("beta_rat")
        pairs = res.get("feature_atom_pairs")
        if not br or not pairs:
            continue
        # beta_rat[0] genelde sabit terim (intercept); atomlar onu izler.
        has_intercept = len(br) == len(pairs) + 1
        kept_pairs, kept_br = [], []
        if has_intercept:
            kept_br.append(br[0])
        offset = 1 if has_intercept else 0
        for i, pr in enumerate(pairs):
            val = br[i + offset][1] if i + offset < len(br) else 0.0
            if abs(val) > 1e-9:
                kept_pairs.append(pr)
                kept_br.append(br[i + offset])
        if kept_pairs and len(kept_pairs) < len(pairs):
            res["feature_atom_pairs"] = kept_pairs
            res["beta_rat"] = kept_br

    if use_mcmc and not fast:
        for res in results[:3]:
            try:
                r2_ = fit_atom_model(
                    data, target, res["feature_atom_pairs"], H=H, use_mcmc=True, fast=False
                )
                if r2_["mdl"]<res["mdl"]: res.update(r2_)
            except: pass
        results.sort(key=lambda r: r["mdl"])

    # ── GÜVEN SKORU (0–1): kullanıcıya "bu formüle ne kadar güvenebilirim" sinyali.
    # Sert "güvenme" yerine olasılık verir (scikit-learn predict_proba mantığı).
    # Esas sinyal nRMSE (normalize artık büyüklüğü): doğru biçimde ~0, havuz-dışı
    # veya bileşik formüllerde belirgin yükselir — R²'den daha ayırt edici
    # (R² 0.90–0.99 arası yanıltıcı olabilir; nRMSE bunu yakalar).
    for res in results:
        y_arr = data[target].values.astype(float)
        yp = np.asarray(res.get("y_pred", []), dtype=float)
        if yp.size == len(y_arr) and np.std(y_arr) > 0:
            nrmse = float(np.sqrt(np.mean((y_arr - yp) ** 2)) / (np.std(y_arr) + 1e-12))
        else:
            nrmse = 1.0
        r2 = res.get("R2") or res.get("r2") or 0.0
        # nRMSE→güven: 0 nRMSE → ~1.0 güven; 0.05 → ~0.7; 0.15 → ~0.3; 0.3+ → ~0.1
        conf_nrmse = float(np.exp(-nrmse / 0.07))
        # R² düşükse ek ceza (her iki sinyal birlikte)
        conf_r2 = max(0.0, min(1.0, (r2 - 0.90) / 0.099)) if r2 < 0.999 else 1.0
        confidence = round(max(0.0, min(1.0, 0.7 * conf_nrmse + 0.3 * conf_r2)), 3)
        res["confidence"] = confidence
        # İnsan-okunur etiket (yine de sayıyı ezmez)
        res["confidence_label"] = (
            "high" if confidence >= 0.8 else
            "medium" if confidence >= 0.5 else
            "low"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 14 — RAPOR
# ═══════════════════════════════════════════════════════════════════════════════

W=74
def sep(c="═"): print(c*W)
def sep2():     print("─"*W)

def qflag(r2v,mv):
    if r2v>0.95 and mv<5:   return "✓✓ MÜKEMMEL"
    elif r2v>0.8 and mv<15: return "✓  İYİ"
    elif r2v>0.6:           return "△  ORTA"
    else:                   return "✗  ZAYIF"

def print_result(res, y_true, target, rank=1):
    sep()
    mt  = res.get("model_type","")
    tag = "GÜÇYASASI" if mt=="power_law" else "ATOM-LİNEER"
    r2v = res.get("r2", 0)
    mv = res.get("mape", 0)
    mdlv = res.get("mdl", 0)
    mlbl = res.get("mape_label", "sMAPE%")
    print(f"  #{rank}  {tag}   MDL={mdlv:.4f}   R²={r2v:.4f}   {mlbl}={mv:.2f}%   "
          f"RMSE={res.get('rmse', 0):.4g}   {qflag(r2v, mv)}")
    if res.get("mape_note"):
        print(f"  ℹ  {res['mape_note']}")
    if res.get("dim_warn"):
        print(f"  ⚠  {res['dim_warn']}")
    if res.get("unit_inferred") and res.get("unit_suggestions"):
        print("  ── Otomatik birim tahmini (P5) ─────────────────────────────")
        for col, hint in res["unit_suggestions"].items():
            print(f"    {col}: {hint}")
    for w in res.get("protocol_warnings") or []:
        if w != res.get("dim_warn"):
            print(f"  {w}")
    sep2()

    # Denklem
    if mt == "power_law":
        es = " × ".join(f"{f}^({l})" for f, l, _ in res["exponents"])
        c_show = res.get("C_piH_simple", res.get("C_piH"))
        idc = res.get("id_phys_const") or identify_physical_constant(res.get("C", 0))
        print(f"  {target} = [{c_show} = {res['C']:.6g}]{phys_const_note(idc)} × {es}")
    else:
        cnames = res.get("col_names", [])
        parts = []
        for i, (l, v, e) in enumerate(res["beta_rat"]):
            nm = cnames[i] if i < len(cnames) else f"β{i}"
            if i == 0:
                idc = res.get("id_phys_const") or identify_physical_constant(float(v))
                parts.append(f"[{l}={v:.4g}]{phys_const_note(idc)}")
            else:
                parts.append(f"{'+' if v >= 0 else ''}[{l}]·{nm}")
        print(f"  {target} = " + " ".join(parts))
    print()

    # Kalite
    sr = res.get("sr_cf",0)
    print(f"  {'R²':<18} = {r2v:.6f}")
    print(f"  {mlbl:<18} = {mv:.4f}%")
    print(f"  {'RMSE':<18} = {res.get('rmse', 0):.6g}")
    if res.get("smape") is not None and res.get("use_nrmse"):
        print(f"  {'sMAPE (ref)':<18} = {res.get('smape', 0):.4f}%")
    if res.get("mape_note"):
        print(f"  {'Not':<18}   {res['mape_note']}")
    print(f"  {'MDL':<18} = {mdlv:.4f}  (lower=better)")
    cv = res.get("cv_stats")
    if cv:
        print(f"  {'CV R²':<18} = {cv['CV_R2_mean']:.4f} ± {cv['CV_R2_std']:.4f}")
        print(f"  {'CV MAPE%':<18} = {cv['CV_MAPE_mean']:.2f} ± {cv['CV_MAPE_std']:.2f}")
        print(f"  {'CV RMSE':<18} = {cv['CV_RMSE_mean']:.4g} ± {cv['CV_RMSE_std']:.4g}")
    sr_t = res.get("sr_taylor",None)
    print(f"  {'SR[erf]':<18} = {sr:.4f}  E[SR]=erf(ε/σ√2)  [v8.5 closed form]")
    if sr_t is not None:
        print(f"  {'SR[Taylor]':<18} = {sr_t:.4f}  SR≈39/49·(ε/σ)  [ε/σ→0 asimptot, 39/49=ΠH(√(2/π))]")
    jv = res.get("J_score",None)
    if jv is not None:
        print(f"  {'J(f) combined':<18} = {jv:.4f}  L(f)+λax·Cax+λdiff·Cdiff  [v6.1]")

    # J(f) + v6.1 tam protokol raporu
    jfv  = res.get("J_full", res.get("J_score",0))
    print(f"  {'J_full (v6.1)':<18} = {jfv:.4f}  MDL+ΣCᵢ  (lower=better)")
    v61 = res.get("v61",{})
    if v61:
        print(f"\n  ── v6.1 Fizik Protokolleri (16 kural) ──────────────────────")
        rows = [
            ("P2  Monotonluk",  "P2_mono",  1.0,  "ters yön ihlali"),
            ("P3  Sıfır geçiş", "P3_zero",  0.3,  "f(x_0)≠0"),
            ("P4  Asimptot",    "P4_asym",  1.0,  "kuyruk sapması"),
            ("P7  Katsayı ±",   "P7_sign",  0.5,  "işaret ihlali"),
            ("P8  Çıkış sınırı","P8_bounds",0.1,  "aralık dışı"),
            ("P9  Süreklilik",  "P9_cont",  2.0,  "sıçrama"),
            ("P10 Periyodiklik","P10_period",0.5, "frekans uyuşmazlığı"),
            ("P11 Skala",       "P11_scale",0.5,  "ölçek uyumsuzluğu"),
            ("P12 Çift/Tek",    "P12_sym",  0.5,  v61.get("P12_type","?")),
            ("P13 Pozitiflik",  "P13_pos",  0.5,  "negatif tahmin"),
            ("P14 Etk.sırası",  "P14_order",0.5,  "katsayı sırası"),
            ("P15 Türev sürek.","P15_diff",  1.0, "ani sıçrama"),
            ("P16 Korunumluluk","P16_cons",  0.5, "moment/kümülatif"),
            ("P_türev işaret",  "P_deriv_pen", 0.5, "∂f/∂xᵢ vs veri"),
            ("P_asimptot",      "P_asym_pen", 0.5, "x→min/max sapma"),
            ("Cax Simetri",     "Cax",       0.5, "eksen simetrisi"),
        ]
        for name,key,thresh,desc in rows:
            val = v61.get(key,0)
            if isinstance(val,float):
                flag = "⚠" if val>thresh else "✓"
                print(f"  {name:<20} = {val:8.4f}  {flag}  {desc}")
        if v61.get("P10_freq_y"):
            print(f"  P10 frekans: veri={v61['P10_freq_y']}  tahmin={v61['P10_freq_f']}")
        for w in v61.get("P_deriv_warns") or []:
            print(f"  {w}")
        if v61.get("P16_warn"):
            print(f"  {v61['P16_warn']}")
        for w in v61.get("P_asym_warns") or []:
            print(f"  {w}")

    extp = res.get("ext_protocols") or {}
    p16d = extp.get("P16_deep") or {}
    if p16d and not res.get("_fast"):
        print(f"\n  ── P16 Enerji Korunumu (derin) ───────────────────────────────")
        print(f"  {'Cumulative drift':<18} = {p16d.get('max_cumulative_drift', 0):.4g}")
        print(f"  {'1. moment rel':<18} = {p16d.get('moment1_rel', 0):.4g}")
        print(f"  {'2. moment rel':<18} = {p16d.get('moment2_rel', 0):.4g}")
    asym_ext = extp.get("P_asym_ext") or {}
    if asym_ext.get("warnings") and not v61:
        print(f"\n  ── Asymptotic Behavior ───────────────────────────────────────")
        for w in asym_ext["warnings"]:
            print(f"  {w}")

    # SAIM-Y alan metrigi (v7.1)
    am = res.get("area",{})
    if am:
        print(f"\n  ── SAIM-Y Alan Metrigi (v7.1) ───────────────────────────────")
        mode = "rough" if am.get("rough",False) else "smooth"
        print(f"  {'MAE':<18} = {am['mae']:.6g}  TRI={am.get('TRI',0):.4f} [{mode}]")
        print(f"  {'p (intercept)':<18} = {am['p']:.4f}")
        print(f"  {'κ(p)=1-3/8·p':<18} = {am['kappa']:.4f}  [ΠH kilidi: 3/8]")
        print(f"  {'A(f,D)':<18} = {am['A']:.6g}  A_approx={am['A_approx']:.6g}")
        lev = res.get("le")
        if lev is not None:
            print(f"  {'LE':<18} = {lev:.4f}  ln(MAE/MAE_b)-11/24·Δp  [ΠH: 11/24]")
        ainf = res.get("A_inf")
        if ainf is not None:
            print(f"  {'A∞ (large-n)':<18} = {ainf:.6g}  (n-1)·MAE·13/16  [Normal, p→1/2, ΠH: 13/16]")

    ra = res.get("residual")
    if ra:
        print(f"\n  ── Residual Analysis ───────────────────────────────────")
        if ra.get("shapiro_p") is not None:
            print(f"  {'Shapiro p':<18} = {ra['shapiro_p']:.4f}")
        if ra.get("bp_r") is not None:
            print(f"  {'|r(ŷ,e²)|':<18} = {abs(ra['bp_r']):.4f}")
        if ra.get("durbin_watson") is not None:
            print(f"  {'Durbin-Watson':<18} = {ra['durbin_watson']:.4f}")
        for w in ra.get("warnings", []):
            print(f"  {w}")

    # MDL* (v4) — kapalı form dahil
    ms = res.get("mdlstar")
    if ms:
        flag = "⚠ AŞIRI UYUM" if ms["overfit"] else "✓ İyi genelleme"
        print(f"\n  ── MDL* Penaltisi (v4) ──────────────────────────────────────")
        print(f"  {'L* (SR gap)':<18} = {ms['L_star']:.4f}  {flag}")
        print(f"  {'L* 1.derece':<18} = {ms['L_closed_1']:.4f}  λ·r^k̂·Δk·ln r  [ΠH: 1]")
        print(f"  {'L* 2.derece':<18} = {ms['L_closed_2']:.4f}  +1/2·(Δk·ln r)²  [ΠH: 1/2]")
        print(f"  SR_tr={ms['SR_train']:.4f}  SR_te={ms['SR_test']:.4f}  gap={ms['gap']:.4f}  r={ms['r']:.3f}")

    # Gibbs / MCMC (v5)
    mc = res.get("mcmc")
    if mc and "gibbs" in mc:
        gb = mc["gibbs"]; vd = "keşfet" if gb["He"]>0.7 else "ince ayar" if gb["He"]>0.3 else "yakınsadı"
        print(f"\n  ── Gibbs Entropi / Turnuva MCMC (v5) ───────────────────────")
        print(f"  {'H̃=H/ln N':<18} = {gb['He']:.4f}  [{vd}]  r_heat={gb['r_heat']:.4f}")
        print(f"  {'H / Hmax':<18} = {gb['H']:.4f} / {gb['Hmax']:.4f}")
        if "H_asymp" in gb:
            print(f"  {'H(T→∞) asymptote':<18} = {gb['H_asymp']:.4f}  ln N-Var(E)/(2NT²)  T={gb['T']:.2f}")
            print(f"  {'H̃_asymp':<18} = {gb['He_asymp']:.4f}  Var(E)={gb['var_E']:.4g}")
        print(f"  {'MCMC MDL':<18} = {mc['energy']:.4f}  ({mc['n_chains']} zincir → {mc['n_elite']} elite)")

    # Asimptotik kilit (v8.x)
    asym = res.get("asym")
    if asym:
        print(f"\n  ── Asimptotik Kilit (v8.x) ─────────────────────────────────")
        print(f"  Mod={asym['mode']}  κ: {asym['cond_before']:.2e}→{asym['cond_after']:.2e}  ({asym['improvement']:.1f}×)")

    # ΠH katsayılar (v2+v3)
    H = res.get("H",20)
    bci = res.get("bootstrap_ci")
    print(f"\n  ── ΠH Rational Coefficients (v2+v3, H={H}, Q_H^alg) ────────────")
    if bci:
        print(f"  Bootstrap {bci['ci']*100:.0f}% CI  (n={bci['n_ok']} successful resamples)")
    hdr = f"  {'Parametre':<22} {'OLS':>11} {'ΠH':>16} {'Δ':>10}"
    if bci:
        hdr += f"  {'CI %2.5–%97.5':>22}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    bo = res.get("beta_ols", []); br = res.get("beta_rat", [])
    if mt == "power_law":
        pnames = ["ln(C)"] + [f"exp({f})" for f in res["features"]]
    else:
        pnames = res.get("col_names", [f"β{i}" for i in range(len(br))])
    for i, (l, v, e) in enumerate(br):
        ov = bo[i] if i < len(bo) else 0.0
        pn = pnames[i] if i < len(pnames) else f"β{i}"
        pi_disp = l
        if i == 0 and mt == "power_law" and res.get("C_piH_simple"):
            pi_disp = res["C_piH_simple"]
        line = f"  {pn:<22} {ov:>11.6f} {pi_disp:>16}  {e:>10.4e}"
        if bci and i < len(bci.get("intervals", [])):
            iv = bci["intervals"][i]
            line += f"  [{iv['low']:>9.4g}, {iv['high']:>9.4g}]"
        print(line)
    print(f"\n  ΠH üst sınır: 29/48·H^(-7/4)  H={H} → {pi_H_error_bound(H):.6f}  [13/16 asimptot A∞]")

    # Öz-referanslı kapalı form zinciri (v8.5)
    sr_v  = res.get("sr_cf",0)
    sr_tv = res.get("sr_taylor",None)
    am_v  = res.get("area",{})
    le_v  = res.get("le",None)
    ms_v  = res.get("mdlstar")
    ainf_v= res.get("A_inf",None)
    jv2   = res.get("J_score",None)
    print(f"\n  ── Öz-Referanslı Kapalı Form Zinciri (v8.5) ───────────────────")
    chain = []
    chain.append(f"E[SR]={sr_v:.4f}")
    if sr_tv is not None: chain.append(f"SR_T={sr_tv:.4f}")
    mdl_approx = len(res.get("y_pred",[1]))*(1-sr_v)
    chain.append(f"-n·ln(SR)≈{mdl_approx:.2f}")
    if am_v: chain.append(f"κ(p)={am_v.get('kappa',0):.4f}")
    if ainf_v is not None: chain.append(f"A∞={ainf_v:.4g}")
    if le_v is not None: chain.append(f"LE={le_v:.4f}")
    if ms_v: chain.append(f"L*={ms_v['L_star']:.4f}")
    if jv2 is not None: chain.append(f"J={jv2:.4f}")
    print("  " + "  →  ".join(chain))
    print(f"  ΠH kilitleri: {{1, 39/49, 3/8, 11/24, 29/48, 13/16, 1/2}}")

    # Tahmin
    yp = np.array(res["y_pred"])
    print(f"\n  ── Tahmin vs Gerçek ─────────────────────────────────────────────")
    print(f"  {'#':>4}  {'Gerçek':>13}  {'Tahmin':>13}  {'Hata%':>9}")
    print("  "+"─"*48)
    for i,(yv,ypv) in enumerate(zip(y_true,yp)):
        ep = 100*(ypv-yv)/(abs(yv)+1e-300)
        st = "✓" if abs(ep)<5 else "△" if abs(ep)<15 else "✗"
        print(f"  {i+1:>4}  {yv:>13.4f}  {ypv:>13.4f}  {ep:>+9.2f}%  {st}")
    print()

def print_sindy(equations):
    sep(); print("  SINDy ODE Keşfi  (v7 — SVD prewhitening + STLS + ΠH + dt düzeltme)"); sep2()
    for state,eq in equations.items():
        print(f"\n  d{state}/dt =")
        for t in eq["active_terms"]:
            print(f"    ({t['coeff_piH']:>10}) · {t['name']:<24}  "
                  f"[OLS={t['coeff_ols']:+.4f}  Δ={t['piH_err']:.2e}]")
        if not eq["active_terms"]: print("    ≈ 0")
        print(f"    R²={eq['r2']:.4f}  MAPE={eq['mape']:.2f}%  MDL={eq['mdl']:.4f}")
        print(f"    κ: {eq['kappa_before']:.2e} → {eq['kappa_after']:.2e}  (SVD prewhitening)")


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 15 — İNTERAKTİF ATOM SEÇİMİ
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_atom_selection(include_cross=True):
    sep(); print("  ATOM KÜTÜPHANESİ  —  Arama uzayını seç"); sep2()
    all_atoms = []
    for group,keys in ATOM_GROUPS.items():
        is_cross = (group=="Çapraz etkileşim")
        print(f"\n  [{group}{'  ← iki değişken gerektirir' if is_cross else ''}]")
        for key in keys:
            if is_cross:
                name,_,_ = CROSS_ATOM_LIBRARY[key]; idx = len(all_atoms)+1
            else:
                name,_,_ = ATOM_LIBRARY[key]; idx = len(all_atoms)+1
            all_atoms.append((key, is_cross))
            print(f"    {idx:>3}.  {key:<22}  {name}")
    print(f"\n  Seçim: numara, grup adı, atom adı, 'all', Enter=varsayılan")
    raw = input("  Atomlar > ").strip()

    if not raw:
        sel = ["x","ln_x","sqrt_x","x2","inv_x"]; print(f"  → Varsayılan: {sel}"); return sel, []
    if raw.lower()=="all":
        sa = list(ATOM_LIBRARY.keys()); sc = list(CROSS_ATOM_LIBRARY.keys())
        print(f"  → Tümü: {len(sa)} tekli + {len(sc)} çapraz"); return sa, sc

    sel_single=[]; sel_cross=[]
    for group,keys in ATOM_GROUPS.items():
        is_cross = (group=="Çapraz etkileşim")
        if group.lower() in raw.lower():
            for k in keys:
                lst = sel_cross if is_cross else sel_single
                if k not in lst: lst.append(k)
    for token in raw.split():
        if token.isdigit():
            idx = int(token)-1
            if 0<=idx<len(all_atoms):
                k,is_cross = all_atoms[idx]
                lst = sel_cross if is_cross else sel_single
                if k not in lst: lst.append(k)
        elif token in ATOM_LIBRARY and token not in sel_single: sel_single.append(token)
        elif token in CROSS_ATOM_LIBRARY and token not in sel_cross: sel_cross.append(token)

    if not sel_single and not sel_cross:
        sel_single=["x","ln_x","sqrt_x","x2","inv_x"]; print(f"  → Varsayılan")
    else:
        print(f"  → Tekli ({len(sel_single)}): {sel_single}")
        if sel_cross: print(f"  → Çapraz ({len(sel_cross)}): {sel_cross}")
    return sel_single, sel_cross


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 16 — İNTERAKTİF AKIŞ
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_run(data):
    sep("═"); print("  YCPA-P v8.5  [TAM VERSİYON]  —  Sembolik Regresyon & Fiziksel Yasa Keşfi")
    print("  Yusuf Said Osmanoglu · Nisan 2026"); sep("═")
    print(f"\n  {len(data)} satır, {len(data.columns)} sütun")
    for i,c in enumerate(data.columns): print(f"    {i+1:>3}. {c}")

    tv = input("\n  Hedef değişken (ad veya numara) > ").strip()
    target = data.columns[int(tv)-1] if tv.isdigit() else tv
    if target not in data.columns: print(f"  ✗ '{target}'"); return

    remaining = [c for c in data.columns if c!=target]
    print(f"\n  Bağımsız değişkenler (Enter=hepsi): {remaining}")
    fv = input("  > ").strip()
    if not fv:
        features = remaining
    else:
        features = []
        for t in fv.split():
            if t.isdigit(): features.append(data.columns[int(t)-1])
            elif t in data.columns: features.append(t)
    if not features: features = remaining
    print(f"  → {features}")

    # Zaman sütunu (SINDy için)
    time_col = None
    if any(c.lower() in ("time","t","zaman","sure") for c in data.columns):
        candidates = [c for c in data.columns if c.lower() in ("time","t","zaman","sure")]
        print(f"\n  Zaman sütunu tespit edildi: {candidates[0]}")
        time_col = candidates[0]

    print("\n  Model tipi:")
    print("    1. Güç yasası   Y=C·x₁^a·x₂^b·...")
    print("    2. Atom tarama  Y=Σ β·atom(x)  (çapraz atomlar dahil)")
    print("    3. SINDy/ODE    dX/dt=Θ(X)·Ξ")
    print("    4. Hepsi")
    mtype = input("  [1/2/3/4, Enter=1] > ").strip() or "1"

    hv = input(f"\n  ΠH H değeri [Enter=20] > ").strip()
    H  = int(hv) if hv.isdigit() else 20
    print(f"  → H={H}  üst sınır={pi_H_error_bound(H):.6f}")

    mcmc_a = input("\n  Turnuva MCMC? [e/H, Enter=H] > ").strip().lower()
    use_mcmc = mcmc_a in ("e","evet","y","yes")

    y_true = data[target].values.astype(float); all_results=[]

    if mtype in ("1","4"):
        print("\n  [Güç Yasası] hesaplanıyor...")
        all_results.append(fit_power_law(data,target,features,H=H,use_mcmc=use_mcmc,verbose=True))

    if mtype in ("2","4"):
        sa,sc = interactive_atom_selection()
        mtv = input("\n  Maks terim sayısı [Enter=3] > ").strip()
        mt  = int(mtv) if mtv.isdigit() else 3
        print("\n  [Atom Tarama] taranıyor...")
        sr = scan_atom_space(data,target,features,sa,selected_cross=sc or None,
                             H=H,max_terms=mt,use_mcmc=use_mcmc,verbose=True)
        all_results.extend(sr[:5])

    if mtype in ("3","4"):
        print("\n  [SINDy] ODE keşfi...")
        sa2,_ = interactive_atom_selection() if mtype=="3" else (["x","x2","x3","sin","cos","x_ln_x"],[])
        eqs = sindy_discover(data,features,sa2,time_col=time_col,H=H,verbose=True)
        print_sindy(eqs)

    if all_results:
        all_results.sort(key=lambda r: r["mdl"])
        print("\n"); sep("═"); print("  SONUÇLAR  (MDL sırası)"); sep("═")
        for rank,res in enumerate(all_results[:5],1):
            print_result(res,y_true,target,rank)
        if len(all_results)>1:
            sep2()
            print(f"  {'#':<4} {'Model':<20} {'R²':>8} {'Hata%':>8} {'RMSE':>10} {'MDL':>10}")
            print("  "+"─"*62)
            for i,res in enumerate(all_results[:10],1):
                tag = "GüçYasası" if res["model_type"]=="power_law" else "Atom"
                print(f"  {i:<4} {tag:<20} {res.get('r2',0):>8.4f} {res.get('mape',0):>8.2f} "
                      f"{res.get('rmse',0):>10.4g} {res['mdl']:>10.4f}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 17 — VERİ YÜKLEME & CLI
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_sample(df: pd.DataFrame, max_rows: int = MAX_ANALYSIS_ROWS) -> pd.DataFrame:
    """Korelasyon, VIF ve model fit için alt örneklem."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def load_data(path, max_rows: int = MAX_LOAD_ROWS) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        if ext == ".xls":
            raise ValueError(
                ".xls formatı desteklenmiyor. "
                "Dosyayı Excel'de .xlsx olarak kaydedin."
            )
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Desteklenmeyen: {ext}")
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    dropped = set(df.columns) - set(num)
    if dropped:
        print(f"  Sayısal olmayan atlandı: {dropped}")
    df = df[num]
    if len(df) > max_rows:
        print(f"  ⚠  {len(df)} satır → bellek için {max_rows} rastgele örnek yüklendi")
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df

def main():
    p = argparse.ArgumentParser(description="YCPA-P v8.5 [TAM]")
    p.add_argument("--data"); p.add_argument("--target")
    p.add_argument("--features", nargs="+"); p.add_argument("--atoms", nargs="+")
    p.add_argument("--cross", nargs="+", help="Çapraz atom keyler")
    p.add_argument("--H", type=int, default=20)
    p.add_argument("--model", type=str, default="power",
                     choices=["power", "atom", "sindy", "both"])
    p.add_argument("--max_terms", type=int, default=3)
    p.add_argument("--mcmc", action="store_true")
    p.add_argument("--time_col", type=str, default=None, help="SINDy için zaman sütunu")
    p.add_argument("--bootstrap", action="store_true",
                   help="Bootstrap güven aralığı (güç yasası)")
    p.add_argument("--boot_n", type=int, default=200, help="Bootstrap iterasyon sayısı")
    p.add_argument("--cv", action="store_true", help="k-fold çapraz doğrulama")
    p.add_argument("--cv_k", type=int, default=5, help="CV fold sayısı")
    p.add_argument("--units", type=str, default=None,
                   help="SI birimleri: T=s a=m veya T=s,L=m")
    p.add_argument("--multi_target", nargs="+", default=None,
                   help="Çoklu hedef güç yasası: --multi_target Fy Fx")
    p.add_argument("--no_p5_kill", action="store_true",
                   help="P5 boyutsal Hard-Kill kapat")
    args = p.parse_args()

    if not args.data:
        print("\n  Demo: Kepler T=a^{3/2}")
        a = np.array([0.387,0.723,1.000,1.524,5.203,9.537,19.19,30.07])
        data = pd.DataFrame({"a":a,"T":a**1.5})
        args.target="T"; args.features=["a"]; args.model="power"
    else:
        print(f"\n  Yükleniyor: {args.data}")
        data = load_data(args.data)
        print(f"  {len(data)} satır × {len(data.columns)} sütun: {list(data.columns)}")

    if not args.target:
        interactive_run(data); return

    target = args.target
    features = args.features or [c for c in data.columns if c != target]
    y_true = data[target].values.astype(float)
    results = []

    unit_map: dict = {}
    unit_inferred_cli = False
    if args.units:
        _, unit_map = parse_units_cli(args.units)
    dim_lhs = unit_map.get(target) if unit_map else None
    p5_kill = not args.no_p5_kill

    if args.multi_target:
        mtargets = list(args.multi_target)
        print(f"\n  [Çoklu Hedef Güç Yasası] {mtargets}")
        mt = fit_multi_target(
            data, mtargets, features, H=args.H,
            use_mcmc=args.mcmc, fast=False, dim_map=unit_map,
        )
        print_multi_target(mt)
        return

    if args.model in ("power", "both"):
        print("\n  [Güç Yasası]")
        pw = fit_power_law(
            data, target, features, H=args.H,
            use_mcmc=args.mcmc, verbose=True,
            dim_lhs=dim_lhs, dim_map=unit_map, p5_hard_kill=p5_kill,
            allow_unit_inference=not bool(args.units),
        )
        if pw:
            if pw.get("unit_inferred") and pw.get("unit_suggestions"):
                print("  [P5] Otomatik birim tahmini:")
                for col, hint in pw["unit_suggestions"].items():
                    print(f"    {col}: {hint}")
            if args.bootstrap:
                print("  [Bootstrap CI] hesaplanıyor…")
                pw["bootstrap_ci"] = bootstrap_ci(
                    data, target, features, pw,
                    n_boot=args.boot_n, H=args.H, fast=True,
                )
            if args.cv:
                print("  [k-fold CV] hesaplanıyor…")
                pw["cv_stats"] = kfold_cv(
                    data, target, features, model_type="power",
                    k=args.cv_k, H=args.H, fast=True,
                    unit_lhs=dim_lhs, unit_map=unit_map,
                )
            results.append(pw)
        elif unit_map:
            print("  ✗ Güç yasası: P5 boyutsal tutarsızlık — model elendi")

    if args.model in ("atom", "both"):
        atoms  = args.atoms or ["x","ln_x","sqrt_x","x2","inv_x","sin","cos","phi_x"]
        cross  = args.cross or []
        print(f"\n  [Atom Tarama] tekli={atoms}  çapraz={cross}")
        sr = scan_atom_space(data,target,features,atoms,
                             selected_cross=cross or None,
                             H=args.H,max_terms=args.max_terms,
                             use_mcmc=args.mcmc,verbose=True)
        results.extend(sr[:5])

    if args.model=="sindy":
        atoms = args.atoms or ["x","x2","x3","sin","cos","x_ln_x"]
        eqs   = sindy_discover(data,features,atoms,
                               time_col=args.time_col,H=args.H,verbose=True)
        print_sindy(eqs); return

    if not results:
        print("\n  Sonuç yok.")
        return

    results.sort(key=lambda r: r["mdl"])
    sep("═"); print("  YCPA-P v8.5 [TAM] SONUÇLARI"); sep("═")
    unc = model_uncertainty(results)
    if unc:
        print(f"  {unc}\n")
    for rank, res in enumerate(results[:5], 1):
        print_result(res, y_true, target, rank)

if __name__=="__main__":
    main()
