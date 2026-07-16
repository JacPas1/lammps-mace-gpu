# syntax=docker/dockerfile:1.7
# Neutral CUDA/Kokkos/LAMMPS/ML-MACE runtime for CHTC GPU jobs.
ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LAMMPS_COMMIT=4d222cb3ee2a6b14083c778968497bf9e0efc4b4
ARG LIBTORCH_VERSION=2.2.0
ARG KOKKOS_ARCH=AMPERE80

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates cmake g++ git libopenmpi-dev ninja-build \
      openmpi-bin python3 unzip wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q \
      "https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu121.zip" \
      -O /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

RUN git clone --filter=blob:none https://github.com/ACEsuit/lammps.git /src/lammps \
    && git -C /src/lammps checkout --detach "${LAMMPS_COMMIT}"

# A container build host has the CUDA toolkit but no physical NVIDIA driver.
# LibTorch's CUDA libraries have a DT_NEEDED entry for libcuda.so.1, while the
# toolkit intentionally ships only the link-time libcuda.so driver stub.  Make
# the SONAME visible for the final lmp link without baking the stub into /opt.
RUN test -f /usr/local/cuda/lib64/stubs/libcuda.so \
    && ln -sfn libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN case "${KOKKOS_ARCH}" in \
      AMPERE80) export TORCH_CUDA_ARCH_LIST=8.0 ;; \
      ADA89) export TORCH_CUDA_ARCH_LIST=8.9 ;; \
      HOPPER90) export TORCH_CUDA_ARCH_LIST=9.0 ;; \
      *) echo "Unsupported KOKKOS_ARCH=${KOKKOS_ARCH}" >&2; exit 2 ;; \
    esac \
    && cmake -S /src/lammps/cmake -B /build/lammps -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/lammps \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_STANDARD_REQUIRED=ON \
      -D CMAKE_CXX_COMPILER=/src/lammps/lib/kokkos/bin/nvcc_wrapper \
      -D CMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
      -D BUILD_MPI=ON \
      -D BUILD_SHARED_LIBS=ON \
      -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ARCH_${KOKKOS_ARCH}=ON \
      -D CMAKE_PREFIX_PATH=/opt/libtorch \
      -D MKL_INCLUDE_DIR=/opt/libtorch/include \
      -D PKG_ML-MACE=ON \
    && cmake --build /build/lammps --parallel 2 \
    && cmake --install /build/lammps

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates libgomp1 libopenmpi3 openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/lammps /opt/lammps
COPY --from=builder /opt/libtorch /opt/libtorch
# Temporary build-time driver stub for the registration smoke test below.
# It is removed in the same RUN and is not on the final runtime library path.
COPY --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1

ARG IMAGE_REVISION=unknown
ENV PATH=/opt/lammps/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/lammps/lib:${LD_LIBRARY_PATH} \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN LD_LIBRARY_PATH=/tmp/cuda-stubs:${LD_LIBRARY_PATH} \
      lmp -h > /tmp/lammps-help.txt \
    && grep -q "ML-MACE" /tmp/lammps-help.txt \
    && grep -q "mace/kk" /tmp/lammps-help.txt \
    && rm -rf /tmp/cuda-stubs /tmp/lammps-help.txt

LABEL org.opencontainers.image.source="https://github.com/JacPas1/lammps-mace-gpu" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.description="CUDA/Kokkos LAMMPS ML-MACE runtime"

WORKDIR /work
CMD ["bash"]
