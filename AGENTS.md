<!--
SPDX-FileCopyrightText: Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
-->

# AGENTS.md

## What this repo is

- Header-only C++20 library for distributed ranges. Two execution models:
  **MP** (multi-process, MPI) in `include/dr/mp`, **SP** (single-process,
  SYCL) in `include/dr/sp`. Public CMake targets: `DR::mpi`, `DR::sp`.
- `include/` is its own CMake subproject; it defines the interface targets and
  fetches dependencies. Do not add `add_subdirectory` calls for it.
- Upstream Intel project is **discontinued** (see README). This checkout is an
  actively modified fork for CPU-only/OpenMPI experiments; keep the notice.
- Tests: `test/gtest/{mp,sp,serial,common}`. Docs: `doc/developer/testing`,
  `doc/spec`. Examples and benchmarks are split by model under
  `examples/{mp,sp,serial}` and `benchmarks/gbench/{mp,sp}`.

## Setup

- Linux, CMake >= 3.20, MPI, and Intel oneAPI **MKL + TBB** (both are
  `find_package(... REQUIRED)`). Source the environment before any configure,
  build, or test run:
  `source /opt/intel/oneapi/setvars.sh` (or source the MKL and MPI
  `env/vars.sh` files individually).
- Python static checks:
  `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.

## Build

- Upstream/Intel path: `CXX=icpx cmake -B build && cmake --build build -- -j`.
- CPU-only GCC path used by this fork:

  ```sh
  source /opt/intel/oneapi/mkl/latest/env/vars.sh
  source /opt/intel/oneapi/mpi/latest/env/vars.sh
  CXX=g++ CC=gcc MPI_CXX_COMPILER=mpicxx cmake -B build \
      -DCMAKE_BUILD_TYPE=Release -DENABLE_SYCL=OFF
  cmake --build build -j2
  ```

- The real option is `ENABLE_SYCL` (OFF by default, forced ON for
  `IntelLLVM`). `-DDR_ENABLE_SYCL=...` in `exp.sh`/`run_experiments.sh` is a
  no-op.
- Useful targets: `all-tests`, `mp-all-tests`, `sp-all-tests`, `mp-tests`,
  `mp-quick-test`, `mp-bench`, `mp-quick-bench`, `wave_equation`,
  `shallow_water`. SP targets exist only when `ENABLE_SYCL` is on.
- Release uses `-O3 -march=native`. Prefer `-j2`; template-heavy compilation
  can exhaust memory.

## Test

- `ctest --test-dir build --output-on-failure -L TESTLABEL -j4` runs the main
  suites; `-L MP` / `-L SP` select a model. `ctest --test-dir build -N` lists
  all registered tests.
- Every MP test is registered three extra times with `-gdb`, `-log`, and
  `-gdb-log` suffixes. `-R mp-tests` also matches those; target exact names
  such as `ctest --test-dir build -R '^mp-tests-2$' --output-on-failure`.
- Direct gtest runs: `mpirun -n 2 ./build/test/gtest/mp/mp-tests
  --gtest_filter=...`; serial: `./build/test/gtest/serial/serial-tests`.
- `mp-quick-test`, `sp-quick-test`, `mp-quick-bench`, `sp-quick-bench` use
  intentionally short source lists for fast iteration; edit their CMake source
  lists when working on one test/benchmark.

## Benchmarks

- MP: `mpirun -n N ./build/benchmarks/gbench/mp/mp-bench
  --benchmark_filter=... --benchmark_format=csv --benchmark_out=...`.
  `--drhelp` shows DR-specific options such as `--vector-size`, `--reps`,
  `--check`, and `--sycl` (device execution). SP binary exists only with SYCL.
- Google Benchmark also accepts `--benchmark_repetitions` and
  `--benchmark_report_aggregates_only`.
- With OpenMPI on this machine the experiment scripts export
  `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader OMPI_MCA_osc=^ucx` to avoid hangs.
  `exp.sh`, `run_experiments.sh`, `paper_plots.py`, and `results/` are local
  experiment artifacts, not upstream code.

## Style and static checks

- Run `pre-commit run --all` before finishing; CI runs exactly this.
  `SKIP=sphinx pre-commit run --all` skips the slow documentation checks.
- Every new file needs the SPDX header (`//` for C++, `#` for CMake/Python):
  `SPDX-FileCopyrightText: Intel Corporation` plus
  `SPDX-License-Identifier: BSD-3-Clause`.
- `scripts/dr-style.py` enforces DR-specific conventions, mainly under
  `include/`: use `#include <...>`, avoid `using namespace sycl`, use
  `rng::size/begin/end`, and use `namespace __detail` instead of anonymous or
  `internal` namespaces. Run it with:
  `python3 scripts/dr-style.py --Werror include test examples`.
- Pre-commit also applies clang-format 16, black at line length 79, and
  cmake-format.

## Fork-specific changes to preserve

- CPU-only builds guard SYCL-only code with `#ifdef SYCL_LANGUAGE_VERSION`
  (see `csr_*_distribution.hpp`, `benchmarks/gbench/mp/gemv.cpp`).
- `include/dr/detail/communicator.hpp` uses `MPI_SUPPORTS_LARGE_COUNT` to
  select MPI-4 large-count `_c` calls versus classic MPI calls. Keep both
  branches when editing.
- GCC needs the top-level `-include cstdint` and `-Wno-template-body` compile
  options added in `CMakeLists.txt`.
