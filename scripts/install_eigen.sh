#!/usr/bin/env bash
# Downloads the Eigen 3.4 (stable) header-only library.
# Runs in the cibuildwheel before-all step on every platform (Linux/macOS/Windows-bash).
# Stable 3.4 chosen: the AVX-512 GEMM crash was master-specific (see dev notes).
set -e

EIGEN_VERSION="3.4.0"
# Target C:/eigen on Windows, /tmp/eigen elsewhere
case "$(uname -s 2>/dev/null || echo Windows)" in
    *NT*|*MINGW*|*MSYS*|Windows) DEST="/c/eigen" ;;
    *) DEST="/tmp/eigen" ;;
esac

mkdir -p "$DEST"
cd "$DEST"

# Stable Eigen 3.4.0 (AVX-512 GEMM crash only in master; 3.4 is safe — verified).
# Primary: eigen-mirror GitHub tag 3.4.0 (tested, real stable release).
# Fallback: official GitLab archive.
URL="https://codeload.github.com/eigen-mirror/eigen/tar.gz/refs/tags/${EIGEN_VERSION}"
ALT="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"

echo "[install_eigen] Downloading Eigen ${EIGEN_VERSION} (stable)..."
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o eigen.tar.gz || curl -fsSL "$ALT" -o eigen.tar.gz
else
    wget -q "$URL" -O eigen.tar.gz || wget -q "$ALT" -O eigen.tar.gz
fi

tar xzf eigen.tar.gz
SRC=$(ls -d eigen-*/ | head -1)

# Place headers under include/eigen3/ (standard location)
mkdir -p include/eigen3
cp -r "${SRC}Eigen" include/eigen3/
cp -r "${SRC}unsupported" include/eigen3/ 2>/dev/null || true

echo "[install_eigen] Eigen kuruldu: $DEST/include/eigen3"
