# LAMMPS-MACE GPU runtime

This repository publishes a neutral GPU runtime for legacy ML-MACE LAMMPS models. It is
independent of any scientific project or input dataset.

The image contains:

- NVIDIA CUDA 12.1.1 runtime;
- LibTorch 2.2.0 for CUDA 12.1;
- ACEsuit/LAMMPS commit `4d222cb3ee2a6b14083c778968497bf9e0efc4b4`;
- Kokkos CUDA support compiled for one NVIDIA architecture per tag;
- the `ML-MACE` package and `mace`/`mace/kk` pair styles.

It deliberately does not contain a MACE model, structure, simulation input, or P3P4 code.
Those remain external HTCondor inputs.

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
lmp -k on g 1 -sf kk -in in.mace
```

Scientific jobs should record the immutable image digest as well as the external model hash.
