#!/usr/bin/env bash
# Install the PointNeXt / OpenPoints source environment for training & evaluation.
#
# Default recipe targets MODERN NVIDIA GPUs, including Blackwell (RTX 50-series /
# RTX 5090, compute capability sm_120), using:
#   * uv-managed virtual environment, Python 3.12
#   * PyTorch >= 2.7 built against CUDA 12.8 (cu128 wheels)
#   * the custom CUDA/C++ operators compiled for your GPU architecture
#
# Blackwell (sm_120) REQUIRES CUDA >= 12.8 and PyTorch >= 2.7. Older PyTorch/CUDA
# stacks (e.g. the original CUDA 11.3 recipe, kept commented at the bottom) will
# NOT run on an RTX 5090.
#
# Usage:  source install.sh
set -e

# Make sure submodules are present.
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 0. Toolchain
# ---------------------------------------------------------------------------
# A full CUDA toolkit (nvcc) is needed to compile the operators. Point CUDA_HOME
# at it and put nvcc on PATH. Adjust the version to match your machine.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}

# PyTorch's cpp_extension rejects host compilers newer than what the CUDA toolkit
# officially supports (e.g. CUDA 12.x wants g++ < 14). If your default gcc is too
# new, install an older one (`sudo apt install gcc-13 g++-13`) and select it here.
export CC=${CC:-gcc-13}
export CXX=${CXX:-g++-13}

# GPU architectures to compile the CUDA kernels for. Include 12.0 for Blackwell.
#   Ampere 3090/A100: 8.0/8.6 ; Ada 4090: 8.9 ; Hopper H100: 9.0 ; Blackwell 5090: 12.0
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.6;8.9;9.0;12.0"}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

# ---------------------------------------------------------------------------
# 1. Python environment (uv + Python 3.12)
# ---------------------------------------------------------------------------
# Install uv first if missing:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 2. PyTorch with CUDA 12.8 (Blackwell / sm_120 support)
# ---------------------------------------------------------------------------
# Check https://pytorch.org for the wheel/index matching your CUDA if different.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
uv pip install -r requirements.txt

# torch-scatter, built against the installed torch. Prebuilt wheels often lag new
# torch releases, so compile from source (uses the toolchain configured above).
uv pip install --no-build-isolation torch-scatter

# ---------------------------------------------------------------------------
# 4. Custom CUDA/C++ extensions (the pointnet++ / pointops libraries)
# ---------------------------------------------------------------------------
# pointnet2 batch ops (ball query / grouping / sampling / interpolation)
( cd openpoints/cpp/pointnet2_batch && uv pip install --no-build-isolation -e . )

# grid_subsampling (needed only for S3DIS_sphere)
( cd openpoints/cpp/subsampling && python setup.py build_ext --inplace )

# pointops (Point Transformer / Stratified Transformer)
( cd openpoints/cpp/pointops && uv pip install --no-build-isolation -e . )

# chamfer_dist + emd (reconstruction / completion tasks only)
( cd openpoints/cpp/chamfer_dist && uv pip install --no-build-isolation -e . )
( cd openpoints/cpp/emd && uv pip install --no-build-isolation -e . )

echo ""
echo "Done. Quick check:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\""


# ===========================================================================
# LEGACY recipe (CUDA 11.3 / PyTorch 1.10, conda). Kept for reference only.
# Does NOT support Blackwell (RTX 5090). Uncomment and adapt for older GPUs.
# ===========================================================================
# export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"
# conda deactivate
# conda env remove --name openpoints
# conda create -n openpoints -y python=3.7 numpy=1.20 numba
# conda activate openpoints
# conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
# pip install -r requirements.txt
# cd openpoints/cpp/pointnet2_batch && python setup.py install && cd ../
# cd subsampling && python setup.py build_ext --inplace && cd ..
# cd pointops/ && python setup.py install && cd ..
# cd chamfer_dist && python setup.py install --user && cd ../emd && python setup.py install --user && cd ../../../
