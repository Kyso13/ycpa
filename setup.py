"""
setup.py — builds the ycpa package and its C++ core (ycpa.ycpa_core).

Kurulum:
    pip install .                  # default: portable AVX2 baseline
    YCPA_NATIVE=1 pip install .    # stable Eigen + AVX-512 (maximum performance)

If C++ cannot be built the package still installs and runs in pure-Python fallback.
"""
import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_ext import build_ext

try:
    from pybind11 import get_include as pybind_include
    import pybind11  # noqa
    _HAVE_PYBIND = True
except Exception:
    _HAVE_PYBIND = False


def _brew_prefix(formula):
    """Return the path of a Homebrew-installed formula (macOS); None otherwise."""
    for cmd in (["brew", "--prefix", formula], ["brew", "--prefix"]):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                          text=True).strip()
            if out and os.path.isdir(out):
                # "brew --prefix libomp" gives the path directly; "brew --prefix" the root
                cand = out if cmd[-1] == formula else os.path.join(out, "opt", formula)
                if os.path.isdir(cand):
                    return cand
        except Exception:
            continue
    return None


def eigen_include():
    env = os.environ.get("EIGEN_INCLUDE_DIR")
    if env and os.path.isdir(env):
        return env
    for cand in ("/usr/include/eigen3", "/usr/local/include/eigen3",
                 "/opt/homebrew/include/eigen3"):
        if os.path.isdir(cand):
            return cand
    return None


class BuildExt(build_ext):
    """Select hardware flags by platform/arch/env; skip on failure."""
    def build_extensions(self):
        import platform as _plat
        ct = self.compiler.compiler_type
        # Cross-compile-safe arch detection: ARCHFLAGS (set by cibuildwheel on macOS)
        # takes priority over platform.machine(), which only reports the HOST cpu and
        # is wrong when an arm64 wheel is cross-built on an x86_64 runner.
        archflags = os.environ.get("ARCHFLAGS", "").lower()
        machine = _plat.machine().lower()
        is_arm = ("arm64" in archflags) or (not archflags and machine in ("arm64", "aarch64"))
        is_x86 = ("x86_64" in archflags) or (not archflags and machine in ("x86_64", "amd64", "i686", "x86"))
        opts, link = [], []

        if ct == "unix":
            opts += ["-O3", "-std=c++17", "-fvisibility=hidden"]

            if sys.platform == "darwin":
                # macOS: do NOT force any SIMD/arch flag. cibuildwheel injects the
                # correct -arch (x86_64 or arm64); Clang + Eigen auto-vectorize
                # (AVX on Intel, NEON on Apple Silicon) at -O3 with zero extra flags.
                # Forcing -mavx2 here leaked into arm64 cross-builds and produced a
                # non-arm64 binary, which made delocate fail. OpenMP is also skipped
                # (Homebrew libomp linkage can break the arm64 slice). Single-threaded
                # is correct; the real speedup is C++/Eigen, not OpenMP.
                pass
            else:
                # Linux (gcc): OpenMP built in.
                opts += ["-fopenmp"]; link += ["-fopenmp"]
                if os.environ.get("YCPA_NATIVE") == "1":
                    opts += ["-mcpu=native"] if is_arm else ["-march=native"]
                elif is_x86:
                    opts += ["-mavx2", "-mfma"]   # portable x86-64-v3 baseline
                # No special SIMD flag needed on ARM Linux (NEON built in).

        elif ct == "msvc":
            # MSVC: /openmp built in, AVX2 via /arch:AVX2 (except ARM Windows).
            opts += ["/O2", "/std:c++17", "/openmp"]
            if is_x86:
                opts += ["/arch:AVX2"]

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link
        try:
            super().build_extensions()
        except Exception as e:
            # If C++ fails to build, the package keeps working in pure Python.
            print(f"[ycpa] C++ extension build failed ({e}); using pure-Python fallback.")


ext_modules = []
_eigen = eigen_include()
if _HAVE_PYBIND and _eigen:
    from setuptools import Extension
    ext_modules = [Extension(
        "ycpa.ycpa_core",
        sources=["ycpa/_native/ycpa_core.cpp"],
        include_dirs=[pybind_include(), _eigen],
        language="c++",
    )]
else:
    print("[ycpa] pybind11 or Eigen not found; skipping C++ extension (pure-Python fallback).")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
