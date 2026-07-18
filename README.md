# LAMMPS-MACE GPU runtime

This repository publishes a neutral GPU runtime for legacy ML-MACE LAMMPS models. It is
independent of any scientific project or input dataset.

The image contains:

- NVIDIA CUDA 12.1.1 runtime;
- LibTorch 2.2.0 for CUDA 12.1;
- ACEsuit/LAMMPS commit `4d222cb3ee2a6b14083c778968497bf9e0efc4b4`;
- `patches/float32-mace-kokkos.patch`, which makes `mace/kk` dtype-aware:
  upstream handles model tensors as float64-only, so float32 LAMMPS exports
  produce NaN forces and a CUDA illegal memory access (cf. ACEsuit/mace#990).
  With the patch, both float32 and float64 exports work; float64 models keep
  the original zero-copy path;
- Kokkos CUDA support compiled for one NVIDIA architecture per tag;
- the `ML-MACE` package and `mace`/`mace/kk` pair styles.

It deliberately does not contain a MACE model, structure, simulation input, or P3P4 code.
Those remain external HTCondor inputs.

The image is compiled on a GPU-less build host using NVIDIA's link-time CUDA driver stub.
That stub is not installed on the runtime library path; CHTC's NVIDIA container runtime
injects the real host driver when a GPU job starts.

## Published tags

| workflow choice | compute capability | tag |
|---|---:|---|
| `ampere80` | 8.0 | `ampere80-cuda12.1` |
| `ada89` | 8.9 | `ada89-cuda12.1` |
| `hopper90` | 9.0 | `hopper90-cuda12.1` |

Example HTCondor setting:

```text
universe = container
container_image = docker://ghcr.io/jacpas1/lammps-mace-gpu:ampere80-cuda12.1
```

Expected single-GPU invocation:

```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.mace
```

ML-MACE requires Newton pair accumulation.  LAMMPS Kokkos defaults to `newton off` and
`neigh full` on GPUs, so both options must be overridden as shown.  If the input file
contains a `newton` command, it must likewise be `newton on`.

Scientific jobs should record the immutable image digest as well as the external model hash.

## mliap-cueq image

`Dockerfile.mliap` publishes `mliap-cueq07-ngc10sep25`: NVIDIA's prebuilt
LAMMPS+PyTorch ML-IAP container (`nvcr.io/nvidia/lammps:10Sep25_pytorch`,
digest-pinned) plus the frozen MACE/cuEquivariance python stack from
`mliap/requirements-frozen.txt` (mace-torch 0.3.16, cuEquivariance 0.7.0 —
the newest release compatible with the base torch 2.8.0a0). This is the
runtime for `pair_style mliap unified` MACE jobs; it removes the per-job
PyPI install that the Phase 0 gates used. Multi-arch (sm80/sm90 selected at
runtime); one tag for all architectures. `USER`/`LOGNAME` are preset because
HTCondor runs containers under a uid absent from /etc/passwd.

Expected invocation (N = number of GPUs):

```bash
mpirun -np N lmp -k on g N -sf kk -pk kokkos newton on neigh half -in in.mliap
```
