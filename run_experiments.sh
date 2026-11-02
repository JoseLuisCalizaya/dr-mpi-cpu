#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Evaluación experimental de Distributed Ranges (CPU local, OpenMPI)
# Adaptado del paper: "Distributed Ranges: A Model for Distributed Data
#                      Structures, Algorithms, and Views"
#
# Este script es EXTERNO al proyecto: no modifica fuentes C++, cabeceras ni
# CMakeLists.txt. Ejecuta (y opcionalmente compila) el binario y guarda CSV
# en results/.
#
# REQUISITOS:
#   - ./build/benchmarks/gbench/mp/mp-bench compilado en Release
#   - OpenMPI con transporte ob1/vader
#   - Intel MKL disponible en /opt/intel/oneapi/mkl/latest
#
# USO:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Variables de entorno opcionales:
#   RANK_LIST="1 2 4 6"       procesos MPI del barrido fuerte
#   VECTOR_SIZE=10000000      elementos por vector (10M por defecto)
#   REDUCE_VECTOR_SIZE=5000000  tamaño dedicado para Reduce (5M)
#   ROWS=5000 COLUMNS=5000    tamaño de Stencil2D / GEMV
#   STENCIL_STEPS=10          pasos del stencil
#   REPS=5                    repeticiones por benchmark
#   RESULTS_DIR=results       directorio de salida
#   MPI_BIND_ARGS="--map-by core --bind-to core"
#   RUN_COMM=1                añade la fase opcional de comunicación MPI
#   RUN_REDUCE=0              omite la fase dedicada de Reduce
#   BUILD=1                   configura y compila mp-bench en Release con -j 2
#
# Los valores por defecto están reducidos para evitar presión de memoria en
# equipos de 8-16 GB. Para replicar el tamaño del paper, usa:
#   VECTOR_SIZE=100000000 ROWS=10000 COLUMNS=10000 ./run_experiments.sh
# Reduce_DR se mide siempre en su fase dedicada con REDUCE_VECTOR_SIZE, así
# que un tamaño grande en VECTOR_SIZE no afecta su autocomprobación float.
# =============================================================================

set -euo pipefail

# ── Flags de transporte MPI (OBLIGATORIOS en este sistema) ──────────────────
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_osc=^ucx

# Un proceso MPI por núcleo: evita que TBB/OpenMP compitan por los mismos hilos
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Variables de entorno Intel MKL/MPI ──────────────────────────────────────
for env_script in \
    /opt/intel/oneapi/mkl/latest/env/vars.sh \
    /opt/intel/oneapi/mpi/latest/env/vars.sh; do
    if [ -f "$env_script" ]; then
        # shellcheck disable=SC1090
        source "$env_script"
    fi
done

# ── Rutas y configuración ───────────────────────────────────────────────────
BENCH_BIN="${BENCH_BIN:-./build/benchmarks/gbench/mp/mp-bench}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="$RESULTS_DIR/logs"
VECTOR_SIZE="${VECTOR_SIZE:-10000000}"
REDUCE_VECTOR_SIZE="${REDUCE_VECTOR_SIZE:-5000000}"
ROWS="${ROWS:-5000}"
COLUMNS="${COLUMNS:-5000}"
STENCIL_STEPS="${STENCIL_STEPS:-10}"
REPS="${REPS:-5}"
RANK_LIST="${RANK_LIST:-1 2 4 6}"
MPI_BIND_ARGS="${MPI_BIND_ARGS:---map-by core --bind-to core}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
RUN_COMM="${RUN_COMM:-0}"
RUN_REDUCE="${RUN_REDUCE:-1}"
BUILD="${BUILD:-0}"

read -r -a RANKS <<< "$RANK_LIST"

# Mapeo directo a la Sección 5 del paper:
#   STREAM Triad, Dot Product, Reduce, Inclusive Scan, Black-Scholes, GEMV.
#
# Reduce_DR se ejecuta en una fase separada con REDUCE_VECTOR_SIZE (5M por
# defecto). Su autocomprobación float es sensible al orden de reducción y
# falla de forma no determinista con 10M o más (por ejemplo, N=1 con 20M:
# Ref 2000000000 vs Actual 1962950000). Mantenerlo separado evita que un
# fallo de precisión aborte todo el barrido de escalabilidad.
STRONG_FILTER="${STRONG_FILTER:-Stream_Triad_DR|DotProduct_DR|Inclusive_Scan_DR|BlackScholes_DR|GemvEq_DR}"
REDUCE_FILTER="${REDUCE_FILTER:-Reduce_DR}"

# Pares DR vs Serial/Std para el experimento de costo cero (N=1).
# Reduce_Serial se omite: su autocomprobación con float falla incluso con
# tamaños pequeños (suma acumulada fuera de tolerancia) y aborta el proceso.
ZEROCOST_FILTER="${ZEROCOST_FILTER:-Fill_DR|Fill_Serial|Copy_DR|Copy_Serial|TransformIdentity_DR|TransformIdentity_Serial|Mul_DR|Mul_Serial|Inclusive_Scan_DR|Inclusive_Scan_Std|Exclusive_Scan_DR|Exclusive_Scan_Std|Stencil1D_Subrange_DR|Stencil1D_Subrange_Std|Stencil2D_DR|Stencil2D_Loop_Serial|DRSortFixture/Sort_DR|StdSortFixture/Sort_Std}"

COMM_FILTER="${COMM_FILTER:-Barrier|Fence|CopyDist2Local_DR|CopyLocal2Dist_DR}"

# Usa la forma --opcion=valor: esta versión de Google Benchmark/cxxopts
# falla al mezclar varios valores separados por espacio.
BENCH_ARGS=(
    --reps=1
    --benchmark_repetitions="$REPS"
    --benchmark_report_aggregates_only=true
    --benchmark_format=csv
    --benchmark_out_format=csv
    --benchmark_time_unit=ms
)
COMMON_ARGS=(
    --vector-size="$VECTOR_SIZE"
    --rows="$ROWS"
    --columns="$COLUMNS"
    --stencil-steps="$STENCIL_STEPS"
    "${BENCH_ARGS[@]}"
)
REDUCE_ARGS=(
    --vector-size="$REDUCE_VECTOR_SIZE"
    "${BENCH_ARGS[@]}"
)

# ── Utilidades ──────────────────────────────────────────────────────────────
run_mpi() {
    local n="$1"
    shift
    # Los argumentos de binding se expanden intencionalmente sin comillas.
    # shellcheck disable=SC2086
    mpirun --allow-run-as-root ${MPI_BIND_ARGS} ${MPI_EXTRA_ARGS} -n "$n" "$@"
}

require_csv() {
    local csv="$1"
    local log="$2"
    if [ ! -s "$csv" ] || ! grep -q '^name,' "$csv"; then
        echo "ERROR: no se generó un CSV válido: $csv" >&2
        echo "       Últimas líneas del log ($log):" >&2
        tail -n 40 "$log" >&2 || true
        return 1
    fi
}

# =============================================================================
# VERIFICACIONES PREVIAS
# =============================================================================
echo "============================================================"
echo "  Distributed Ranges — Evaluación Experimental"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "$BUILD" = "1" ]; then
    if ! command -v cmake &>/dev/null; then
        echo "ERROR: cmake no encontrado en PATH"
        exit 1
    fi
    echo "▶ Compilando mp-bench en Release (-j 2)..."
    CXX=g++ CC=gcc MPI_CXX_COMPILER=mpicxx cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DDR_ENABLE_SYCL=OFF \
        -DENABLE_SYCL=OFF
    cmake --build build -j 2 --target mp-bench
fi

if [ ! -x "$BENCH_BIN" ]; then
    echo "ERROR: binario no encontrado o no ejecutable: $BENCH_BIN"
    echo "       Compílalo en Release con:"
    echo "         BUILD=1 ./run_experiments.sh"
    echo "       o manualmente:"
    echo "         CXX=g++ CC=gcc MPI_CXX_COMPILER=mpicxx cmake -B build \\"
    echo "             -DCMAKE_BUILD_TYPE=Release -DDR_ENABLE_SYCL=OFF \\"
    echo "             -DENABLE_SYCL=OFF"
    echo "         cmake --build build -j 2 --target mp-bench"
    exit 1
fi

if ! command -v mpirun &>/dev/null; then
    echo "ERROR: mpirun no encontrado en PATH"
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

{
    echo "date: $(date -Iseconds)"
    echo "host: $(uname -a)"
    echo "binary: $BENCH_BIN"
    echo "vector_size: $VECTOR_SIZE"
    echo "reduce_vector_size: $REDUCE_VECTOR_SIZE"
    echo "rows: $ROWS"
    echo "columns: $COLUMNS"
    echo "stencil_steps: $STENCIL_STEPS"
    echo "repetitions: $REPS"
    echo "ranks: $RANK_LIST"
    lscpu
} > "$RESULTS_DIR/env.txt" 2>&1 || true

echo ""
echo "▶ Binario:      $BENCH_BIN"
echo "▶ Resultados:   $RESULTS_DIR/"
echo "▶ Vector size:  $VECTOR_SIZE elementos"
echo "▶ Reduce size:  $REDUCE_VECTOR_SIZE elementos (fase separada)"
echo "▶ Matriz:       ${ROWS}x${COLUMNS}, stencil_steps=$STENCIL_STEPS"
echo "▶ Repeticiones: $REPS"
echo "▶ Rangos MPI:   $RANK_LIST"
if [ "$RUN_REDUCE" = "1" ]; then
    echo "▶ Reduce_DR:    fase dedicada con ${REDUCE_VECTOR_SIZE} elementos"
else
    echo "▶ Reduce_DR:    omitido (RUN_REDUCE=0)"
fi
echo ""

# =============================================================================
# FASE 1 — ESCALABILIDAD FUERTE (Strong Scaling)
# T_N con tamaño de problema fijo; S_p = T_1 / T_N se calcula en paper_plots.py
# =============================================================================
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FASE 1: Escalabilidad Fuerte (N = ${RANK_LIST})        ║"
echo "╚══════════════════════════════════════════════════════════╝"

for N in "${RANKS[@]}"; do
    OUT_FILE="$RESULTS_DIR/scaling_n${N}.csv"
    LOG_FILE="$LOG_DIR/scaling_n${N}.log"
    echo ""
    echo "── N=$N procesos MPI ──"
    echo "   CSV: $OUT_FILE"

    if ! run_mpi "$N" "$BENCH_BIN" "${COMMON_ARGS[@]}" \
            --benchmark_filter="$STRONG_FILTER" \
            --benchmark_out="$OUT_FILE" >"$LOG_FILE" 2>&1; then
        echo "ERROR: falló mpirun con N=$N. Revisa $LOG_FILE" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
        exit 1
    fi
    require_csv "$OUT_FILE" "$LOG_FILE"
    echo "   ✓ CSV válido ($(wc -l < "$OUT_FILE") líneas)"
done

# =============================================================================
# FASE 1b — ESCALABILIDAD FUERTE DE REDUCE (tamaño dedicado)
# Reduce_DR usa float y su check es sensible al orden de reducción. Se mide
# con REDUCE_VECTOR_SIZE para que la autocomprobación sea estable.
# =============================================================================
if [ "$RUN_REDUCE" = "1" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  FASE 1b: Reduce (N = ${RANK_LIST})                      ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    for N in "${RANKS[@]}"; do
        OUT_FILE="$RESULTS_DIR/scaling_reduce_n${N}.csv"
        LOG_FILE="$LOG_DIR/scaling_reduce_n${N}.log"
        echo ""
        echo "── Reduce N=$N procesos MPI ──"
        echo "   CSV: $OUT_FILE"

        if ! run_mpi "$N" "$BENCH_BIN" "${REDUCE_ARGS[@]}" \
                --benchmark_filter="$REDUCE_FILTER" \
                --benchmark_out="$OUT_FILE" >"$LOG_FILE" 2>&1; then
            echo "ERROR: falló Reduce con N=$N. Revisa $LOG_FILE" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            exit 1
        fi
        require_csv "$OUT_FILE" "$LOG_FILE"
        echo "   ✓ CSV válido ($(wc -l < "$OUT_FILE") líneas)"
    done
fi

# =============================================================================
# FASE 2 — COMPARATIVA DE COSTO CERO (Zero-Cost Abstraction)
# DR vs implementación serial/std de referencia con N=1
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FASE 2: Comparativa Costo Cero (DR vs Serial, N=1)     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

OUT_FILE="$RESULTS_DIR/zerocost_comparison.csv"
LOG_FILE="$LOG_DIR/zerocost_comparison.log"
echo "── N=1 proceso MPI ──"
echo "   CSV: $OUT_FILE"

if ! run_mpi 1 "$BENCH_BIN" "${COMMON_ARGS[@]}" \
        --benchmark_filter="$ZEROCOST_FILTER" \
        --benchmark_out="$OUT_FILE" >"$LOG_FILE" 2>&1; then
    echo "ERROR: falló la fase de costo cero. Revisa $LOG_FILE" >&2
    tail -n 40 "$LOG_FILE" >&2 || true
    exit 1
fi
require_csv "$OUT_FILE" "$LOG_FILE"
echo "   ✓ CSV válido ($(wc -l < "$OUT_FILE") líneas)"

# =============================================================================
# FASE 3 (OPCIONAL) — OVERHEAD DE COMUNICACIÓN MPI (Barrier / Fence / copias)
# Activar con RUN_COMM=1 ./run_experiments.sh
# =============================================================================
if [ "$RUN_COMM" = "1" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  FASE 3: Comunicación MPI (N = ${RANK_LIST})            ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    for N in "${RANKS[@]}"; do
        OUT_FILE="$RESULTS_DIR/comm_n${N}.csv"
        LOG_FILE="$LOG_DIR/comm_n${N}.log"
        echo "── N=$N procesos MPI ──"

        if ! run_mpi "$N" "$BENCH_BIN" "${COMMON_ARGS[@]}" \
                --benchmark_filter="$COMM_FILTER" \
                --benchmark_out="$OUT_FILE" >"$LOG_FILE" 2>&1; then
            echo "ERROR: falló la fase de comunicación con N=$N" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            exit 1
        fi
        require_csv "$OUT_FILE" "$LOG_FILE"
        echo "   ✓ $OUT_FILE"
    done
fi

# =============================================================================
# RESUMEN FINAL
# =============================================================================
echo ""
echo "============================================================"
echo "  ✓ Experimentos completados"
echo "  Archivos generados:"
ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null || echo "  (ningún CSV encontrado)"
echo ""
echo "  Siguiente paso: python3 paper_plots.py"
echo "============================================================"
