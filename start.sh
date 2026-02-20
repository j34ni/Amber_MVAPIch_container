. /opt/conda/etc/profile.d/conda.sh && conda activate base

export AMBERHOME="${CONDA_PREFIX}/amber24"
export CPATH="${CONDA_PREFIX}/include:${CPATH}"
