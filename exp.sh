#!/usr/bin/env bash
set -e

echo "=== [1/3] Cargando variables de entorno e Intel MKL ==="
source /opt/intel/oneapi/mkl/latest/env/vars.sh
source /opt/intel/oneapi/mpi/latest/env/vars.sh
export CPLUS_INCLUDE_PATH=/opt/intel/oneapi/mkl/latest/include:$CPLUS_INCLUDE_PATH
export CPATH=/opt/intel/oneapi/mkl/latest/include:$CPATH

echo "=== [2/3] Compilando binarios en modo Release (-O3) ==="
# No altera el código fuente, solo reconfigura el directorio de compilación /build
CXX=g++ CC=gcc MPI_CXX_COMPILER=mpicxx cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DDR_ENABLE_SYCL=OFF

# Límite seguro de 2 hilos para proteger tu memoria RAM
cmake --build build -j 2

echo "=== [3/3] Ejecutando Barrido de Benchmarks de Evaluación ==="
export MPI_FLAGS="OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader OMPI_MCA_osc=^ucx"
export DR_FILTER="Stream_Triad_DR|DotProduct_DR|Reduce_DR|Inclusive_Scan_DR|GemvEq_DR|BlackScholes_DR|WaveEquation_DR"
export BASELINE_FILTER="Reduce_Serial|Inclusive_Scan_Std|Stream_Triad_DR|Reduce_DR|Inclusive_Scan_DR"

# Escalabilidad Fuerte (Strong Scaling)
for n in 1 2 4 6; do
  echo "--> Ejecutando con $n procesos MPI..."
  eval $MPI_FLAGS mpirun -n $n ./build/benchmarks/gbench/mp/mp-bench \
    --benchmark_filter="$DR_FILTER" \
    --benchmark_format=csv \
    --benchmark_out="strong_scaling_${n}ranks.csv"
done

# Baseline de Costo Cero
echo "--> Ejecutando baseline para costo cero (1 rango)..."
eval $MPI_FLAGS mpirun -n 1 ./build/benchmarks/gbench/mp/mp-bench \
  --benchmark_filter="$BASELINE_FILTER" \
  --benchmark_format=csv \
  --benchmark_out="zero_cost_baseline.csv"

echo "=== Pruebas completadas. Archivos CSV generados con éxito. ==="
