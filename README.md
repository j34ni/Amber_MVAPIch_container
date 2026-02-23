
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18744113.svg)](https://doi.org/10.5281/zenodo.18744113)

# Amber24-container

Recipes for Docker and Singularity/Apptainer (for `x86_64` and `aarch64` architectures) to build **Amber24** and **AmberTools25** with MVAPIch-4.1 and CUDA 12 support.

**Important:** The source code is not included in this repository because it requires signing an End User License Agreement (EULA).

### How to build

1. Go to the official Amber website: https://ambermd.org/GetAmber.php
2. Download the following files (non-commercial use):
   - `pmemd24.tar.bz2`
   - `AmberTools25.tar.bz2`
3. Place both files in the same directory as the relevant Dockerfile / definition file and `start.sh`
4. Build the container:

```
# For x86_64 (Docker)
docker build --progress=plain -t amber24_x86_64 -f Dockerfile_x86_64 .

# For aarch64 / Grace Hopper (Singularity)
singularity build amber24_aarch64.sif amber24_aarch64.def
```
### How to use

Start the container and type `source /opt/start.sh` to activate the conda environment, then `source $AMBERHOME/amber.sh` sets up the environment variables needed to use Amber.

In a Slurm script the execution command will be something like:
```
srun -n $SLURM_NTASKS --mpi=pmi2 apptainer exec --bind $PWD:/opt/uio amber24.sif  bash -c "cd /opt/uio/amber/20k-atoms && source /opt/start.sh && source \$AMBERHOME/amber.sh && pmemd.MPI -O -i benchmark.in -o benchmark_cpu.out -p benchmark.top -c benchmark.rst -r benchmark_cpu.rst -x benchmark_cpu.nc -inf benchmark_cpu.inf"
```
for `pmemd.MPI` (for the HPC CPU partition)

or:

```
srun -n $SLURM_NTASKS --mpi=pmi2 apptainer exec --nv --bind $PWD:/opt/uio amber24_aarch64.sif  bash -c "cd /opt/uio/amber/20k-atoms && source /opt/start.sh && source \$AMBERHOME/amber.sh && pmemd.cuda -O -i benchmark.in -o benchmark_olivia_gpu.out -p benchmark.top -c benchmark.rst -r benchmark_olivia_gpu.rst -x benchmark_olivia_gpu.nc -inf benchmark_olivia_gpu.inf"
```
for `pmemd.cuda` (this example is for a HPC GPU partition).

Remember to **bind** the folder containing the data (`$PWD` on the host, and in this example `opt/uio` in the container) and to use **`--nv`** for the GPU.

The container comes with everything it needs, therefore it is not necessary to load any module on the host.


## Benchmark results

Benchmarks were run on:
- Saga HPC
  - Intel Xeon CPUs (32 MPI processes)
  - NVIDIA A100 GPUs (single GPU)
- Olivia HPC
  - AMD EPYC Turin CPUs (32 MPI processes)
  - Neoverse-V2 CPUs (32 MPI processes)
  - NVIDIA Grace Hopper GPUs (single GPU)

using a standard Amber benchmark suite (https://www.hecbiosim.ac.uk/file-store/benchmark-suite/amber.tar.xz) `20k-atoms` and `61k-atoms` examples.

### Performance summary

### Throughput and Speedup

![Performance](amber_performance.png)

- Left panel: Throughput in ns/day (higher is better)
- Right panel: Speedup relative to Saga Intel Xeon CPU (higher is better)

Note: CPU runs use 32 MPI processes; GPU runs use a single GPU.

### Energy & Temperature stability

![Energy & Temperature stability](amber_energy_temp.png)

- Temperature remains stable around 300 K
- Total energy is conserved (no drift)
- Density is stable
- Slight energy offsets between CPU (double precision) and GPU (single precision) are expected and documented in Amber

These results confirm that the container produces correct physical behavior and excellent performance on both architectures, with outstanding acceleration on Grace Hopper GPUs.

Also note how the Neoverse-V2 outperforms the Intel Xeon and AMD EPYC Turin.
