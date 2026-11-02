#!/usr/bin/env python3
"""
paper_plots.py
==============
Procesa los CSV generados por run_experiments.sh y reproduce las gráficas
del paper "Distributed Ranges: A Model for Distributed Data Structures,
Algorithms, and Views", adaptadas al entorno local (AMD Ryzen 5 5625U, OpenMPI).

Este script es EXTERNO al proyecto: solo lee results/*.csv y escribe PNG/CSV
en el directorio de salida. No modifica fuentes C++, cabeceras ni CMakeLists.

REQUISITOS:
    pip install pandas matplotlib numpy

USO:
    python3 paper_plots.py [--results-dir results] [--out-dir results]

SALIDAS:
    fig2_paper_panels.png             Paneles estilo Figura 2 (speedup + BW)
    fig3_gemv_time_speedup.png        GEMV: tiempo y speedup vs rangos MPI
    fig4_zerocost_comparison.png      mhp vs Serial/Std a N=1
    fig5_parallel_efficiency.png      Eficiencia E_p = S_p / N
    summary_strong_scaling.csv        Datos numéricos del speedup
    summary_bandwidth.csv             Datos numéricos del ancho de banda
    summary_zerocost.csv              Datos numéricos del costo cero
"""

import argparse
import json
import re
import sys
import warnings
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Sin GUI: guarda directamente a PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

# ─── Paleta de colores ──────────────────────────────────────────────────────
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# ─── Conversión de unidades de tiempo de Google Benchmark ───────────────────
TIME_UNIT_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1e3,
}

# Tamaño de vector por defecto del barrido (10M elementos).
# El número real se lee del contexto del CSV (default_vector_size).
DEFAULT_VECTOR_SIZE = 10_000_000

# Límite teórico de la RAM DDR4 dual-channel del Ryzen 5 5625U.
THEORETICAL_BW_GIB_S = 21.3

# Una referencia Serial por debajo de 1 µs suele haber sido optimizada por el
# compilador (resultado no usado). Se marca como no válida para el costo cero.
MIN_VALID_REF_MS = 1e-3

# Benchmarks con ancho de banda reportado en la Figura 2.
# factors = vectores accedidos; elem_bytes = tamaño del tipo del contenedor.
BANDWIDTH_BENCHMARKS = {
    "Stream_Triad_DR": {
        "factors": 3,   # lee A y B, escribe C
        "elem_bytes": 4,  # distributed_vector<float>
        "label": "STREAM Triad",
    },
    "DotProduct_DR": {
        "factors": 2,   # lee A y B, reduce a escalar
        "elem_bytes": 8,  # distributed_vector<double>
        "label": "Dot Product",
    },
}


# Pares DR vs referencia para el experimento de costo cero.
ZEROCOST_PAIRS = [
    ("Fill_DR", "Fill_Serial"),
    ("Copy_DR", "Copy_Serial"),
    # Reduce_Serial se omite: su check float falla para el tamaño por defecto.
    ("TransformIdentity_DR", "TransformIdentity_Serial"),
    ("Mul_DR", "Mul_Serial"),
    ("Inclusive_Scan_DR", "Inclusive_Scan_Std"),
    ("Exclusive_Scan_DR", "Exclusive_Scan_Std"),
    ("Stencil1D_Subrange_DR", "Stencil1D_Subrange_Std"),
    ("Stencil2D_DR", "Stencil2D_Loop_Serial"),
    ("DRSortFixture/Sort_DR", "StdSortFixture/Sort_Std"),
]


# =============================================================================
# UTILIDADES DE LECTURA
# =============================================================================

def _clean_name(name) -> str:
    """Elimina parámetros y sufijos de agregación de Google Benchmark."""
    if not isinstance(name, str):
        return str(name)
    name = re.sub(r"_(mean|median|stddev|cv)$", "", name)
    name = re.sub(
        r"/(min_time|min_warmup_time|iterations|repetitions):[^/]+", "", name
    )
    name = re.sub(r"/(real_time|cpu_time)$", "", name)
    return name.strip()


def _normalise(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Normaliza columnas de tiempo/bytes y limpia nombres de benchmark."""
    if df.empty:
        return df
    if "name" not in df.columns or "real_time" not in df.columns:
        raise ValueError("el archivo no contiene las columnas 'name' y 'real_time'")

    df = df.copy()
    df["benchmark"] = df["name"].map(_clean_name)

    if "time_unit" in df.columns:
        units = df["time_unit"].fillna("ns").astype(str).str.lower()
    else:
        units = pd.Series(["ns"] * len(df), index=df.index)

    real_time = pd.to_numeric(df["real_time"], errors="coerce")
    df["time_ms"] = real_time * units.map(TIME_UNIT_TO_MS).fillna(1.0)

    if "bytes_per_second" in df.columns:
        df["bytes_per_second"] = pd.to_numeric(
            df["bytes_per_second"], errors="coerce"
        )
    else:
        df["bytes_per_second"] = np.nan

    if "error_occurred" in df.columns:
        errors = (
            df["error_occurred"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[~errors]

    df.attrs["context"] = context
    return df


def _load_csv_text(raw: str) -> pd.DataFrame:
    """Lee un CSV de Google Benchmark con cabecera de contexto multilínea."""
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("name,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("no se encontró la cabecera CSV 'name,...'")

    context = {}
    for line in lines[:header_idx]:
        match = re.match(r"^([A-Za-z0-9_.-]+):\s?(.*)$", line)
        if match:
            context[match.group(1)] = match.group(2)

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    return _normalise(df, context)


def _load_json_text(raw: str) -> pd.DataFrame:
    """Fallback para JSON de Google Benchmark (p. ej. CSVs antiguos)."""
    obj = json.loads(raw)
    context = obj.get("context", {})
    df = pd.DataFrame(obj.get("benchmarks", []))
    if df.empty:
        return df
    if "time_unit" not in df.columns:
        df["time_unit"] = "ns"
    return _normalise(df, context)


def load_benchmark_file(path: Path) -> pd.DataFrame:
    """Carga un resultado de Google Benchmark en formato CSV o JSON."""
    raw = path.read_text(errors="replace")
    if raw.lstrip().startswith("{") and '"benchmarks"' in raw:
        return _load_json_text(raw)
    return _load_csv_text(raw)


def _select_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona la fila agregada media (o todas si no hay repeticiones)."""
    if df.empty:
        return df

    if "aggregate_name" in df.columns:
        mask = df["aggregate_name"].astype(str).str.lower() == "mean"
        if mask.any():
            return df[mask]

    if "name" in df.columns:
        mask = df["name"].astype(str).str.endswith("_mean")
        if mask.any():
            return df[mask]

    return df


def benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por benchmark: mediana de tiempo y bytes/s."""
    selected = _select_mean(df)
    if selected.empty:
        return pd.DataFrame(
            columns=["benchmark", "time_ms", "bytes_per_second"]
        )

    aggregations = {
        "time_ms": ("time_ms", "median"),
        "bytes_per_second": ("bytes_per_second", "median"),
    }
    if "vector_size" in selected.columns:
        aggregations["vector_size"] = ("vector_size", "first")

    return (
        selected.groupby("benchmark", as_index=False)
        .agg(**aggregations)
        .sort_values("benchmark")
    )


def _load_scaling_pattern(results_dir: Path, pattern: str) -> dict:
    """Carga un patrón de CSV por número de rangos MPI."""
    data = {}
    for path in sorted(results_dir.glob(pattern)):
        match = re.search(r"_n(\d+)\.csv$", path.name)
        if not match:
            continue
        n = int(match.group(1))
        try:
            df = load_benchmark_file(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  AVISO: no se pudo leer {path}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        df["vector_size"] = _vector_size(df)
        data[n] = df
    return data


def load_scaling_data(results_dir: Path) -> dict:
    """Carga scaling_n*.csv y scaling_reduce_n*.csv y los une por N."""
    main_data = _load_scaling_pattern(results_dir, "scaling_n*.csv")
    reduce_data = _load_scaling_pattern(
        results_dir, "scaling_reduce_n*.csv"
    )

    data = dict(main_data)
    for n, reduce_df in reduce_data.items():
        if n not in data:
            data[n] = reduce_df
            continue
        context = data[n].attrs.get("context", {})
        merged = pd.concat([data[n], reduce_df], ignore_index=True)
        merged.attrs["context"] = context
        data[n] = merged
    return data


def _vector_size(df: pd.DataFrame) -> int:
    """Obtiene default_vector_size del contexto del CSV."""
    context = df.attrs.get("context", {})
    try:
        return int(float(context.get("default_vector_size", DEFAULT_VECTOR_SIZE)))
    except (TypeError, ValueError):
        return DEFAULT_VECTOR_SIZE


# =============================================================================
# CÁLCULOS
# =============================================================================

def build_scaling_summary(data: dict) -> pd.DataFrame:
    """S_p = T_1 / T_N y E_p = S_p / N para cada benchmark y N."""
    if 1 not in data:
        return pd.DataFrame()

    t1_table = benchmark_table(data[1]).set_index("benchmark")["time_ms"]
    rows = []

    for n in sorted(data):
        table = benchmark_table(data[n])
        for _, row in table.iterrows():
            benchmark = row["benchmark"]
            tn = row["time_ms"]
            if benchmark not in t1_table.index:
                continue
            t1 = t1_table.loc[benchmark]
            if pd.isna(t1) or pd.isna(tn) or t1 <= 0 or tn <= 0:
                continue
            speedup = t1 / tn
            vector_size = (
                int(row["vector_size"])
                if "vector_size" in row.index
                and pd.notna(row["vector_size"])
                else None
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "ranks": n,
                    "time_ms": tn,
                    "speedup": speedup,
                    "efficiency": speedup / n,
                    "vector_size": vector_size,
                }
            )

    return pd.DataFrame(rows)


def build_bandwidth_summary(data: dict) -> pd.DataFrame:
    """Ancho de banda efectivo en GiB/s (bytes_per_second o fórmula)."""
    rows = []

    for n in sorted(data):
        df = data[n]
        default_vector_size = _vector_size(df)
        table = benchmark_table(df).set_index("benchmark")

        for benchmark, spec in BANDWIDTH_BENCHMARKS.items():
            if benchmark not in table.index:
                continue
            time_ms = table.loc[benchmark, "time_ms"]
            bps = table.loc[benchmark, "bytes_per_second"]
            if pd.isna(time_ms) or time_ms <= 0:
                continue

            vector_size = default_vector_size
            if "vector_size" in table.columns:
                value = table.loc[benchmark, "vector_size"]
                if pd.notna(value):
                    vector_size = int(value)

            if pd.notna(bps) and bps > 0:
                bandwidth = float(bps) / (1024**3)
                total_bytes = float(bps) * (time_ms / 1000.0)
            else:
                total_bytes = (
                    spec["factors"] * vector_size * spec["elem_bytes"]
                )
                bandwidth = total_bytes / (time_ms / 1000.0) / (1024**3)

            rows.append(
                {
                    "benchmark": benchmark,
                    "label": spec["label"],
                    "ranks": n,
                    "time_ms": float(time_ms),
                    "bytes": float(total_bytes),
                    "bandwidth_GiB_s": float(bandwidth),
                }
            )

    return pd.DataFrame(rows)


def build_zerocost_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tiempo DR vs referencia y overhead relativo a N=1."""
    table = benchmark_table(df).set_index("benchmark")
    rows = []

    for dr_bench, ref_bench in ZEROCOST_PAIRS:
        if dr_bench not in table.index:
            continue
        dr_time = table.loc[dr_bench, "time_ms"]
        if pd.isna(dr_time):
            continue

        ref_time = np.nan
        if ref_bench and ref_bench in table.index:
            ref_time = table.loc[ref_bench, "time_ms"]

        valid = pd.notna(ref_time) and ref_time >= MIN_VALID_REF_MS
        overhead = np.nan
        if valid and ref_time > 0:
            overhead = (dr_time - ref_time) / ref_time * 100.0

        rows.append(
            {
                "benchmark": dr_bench,
                "reference": ref_bench,
                "dr_time_ms": float(dr_time),
                "ref_time_ms": float(ref_time) if pd.notna(ref_time) else np.nan,
                "overhead_pct": float(overhead) if pd.notna(overhead) else np.nan,
                "valid": bool(valid),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# GRÁFICAS
# =============================================================================

def _pretty_label(benchmark: str) -> str:
    return (
        benchmark.replace("DRSortFixture/", "")
        .replace("StdSortFixture/", "")
        .replace("_DR", "")
        .replace("TransformIdentity", "Transform")
    )


def _ranks_from(*tables) -> list:
    """Rangos MPI presentes en uno o más resúmenes."""
    ranks = set()
    for table in tables:
        if table is None or table.empty or "ranks" not in table.columns:
            continue
        ranks.update(int(value) for value in table["ranks"].unique())
    return sorted(ranks) if ranks else [1, 2, 4, 6]


def _speedup_series(summary: pd.DataFrame, benchmark: str):
    if summary.empty:
        return None
    series = summary[summary["benchmark"] == benchmark].sort_values("ranks")
    return series if not series.empty else None


def _plot_speedup_panel(
    ax, summary: pd.DataFrame, benchmark: str, title: str, ranks: list
) -> None:
    ax.plot(
        ranks,
        ranks,
        "k--",
        linewidth=1.4,
        alpha=0.55,
        label="Ideal (lineal)",
    )

    series = _speedup_series(summary, benchmark)
    if series is None:
        ax.text(
            0.5,
            0.5,
            "sin datos",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=11,
        )
        values = [float(r) for r in ranks]
    else:
        ax.plot(
            series["ranks"],
            series["speedup"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=COLORS[0],
            label=_pretty_label(benchmark),
        )
        values = [float(r) for r in ranks] + [
            float(v) for v in series["speedup"].tolist()
        ]

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Rangos MPI (N)", fontsize=10)
    ax.set_ylabel("Speedup $S_p = T_1/T_N$", fontsize=10)
    ax.set_xticks(ranks)
    ax.set_xlim(min(ranks) - 0.3, max(ranks) + 0.3)
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def _plot_bandwidth_panel(
    ax, bandwidth: pd.DataFrame, benchmark: str, title: str, ranks: list
) -> None:
    max_bw = THEORETICAL_BW_GIB_S
    series = None
    if not bandwidth.empty:
        series = bandwidth[bandwidth["benchmark"] == benchmark].sort_values(
            "ranks"
        )

    if series is not None and not series.empty:
        label = (
            str(series["label"].iloc[0])
            if "label" in series.columns
            else _pretty_label(benchmark)
        )
        ax.plot(
            series["ranks"],
            series["bandwidth_GiB_s"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=COLORS[0],
            label=label,
        )
        max_bw = max(max_bw, float(series["bandwidth_GiB_s"].max()))
    else:
        ax.text(
            0.5,
            0.5,
            "sin datos",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=11,
        )

    ax.axhline(
        THEORETICAL_BW_GIB_S,
        color="#d62728",
        linestyle="--",
        linewidth=1.6,
        label=f"Límite DDR4 ≈ {THEORETICAL_BW_GIB_S:.1f} GiB/s",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Rangos MPI (N)", fontsize=10)
    ax.set_ylabel("Ancho de banda (GiB/s)", fontsize=10)
    ax.set_xticks(ranks)
    ax.set_xlim(min(ranks) - 0.3, max(ranks) + 0.3)
    ax.set_ylim(0, max_bw * 1.20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def plot_paper_panels(
    summary: pd.DataFrame, bandwidth: pd.DataFrame, out_dir: Path
) -> None:
    """Figura estilo Figura 2: paneles de ancho de banda y speedup."""
    if summary.empty and bandwidth.empty:
        print("  AVISO: sin datos para los paneles de la Figura 2")
        return

    ranks = _ranks_from(summary, bandwidth)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(
        "Figura 2 (adaptada): Distributed Ranges en AMD Ryzen 5 5625U "
        "(mhp / OpenMPI)",
        fontsize=15,
        fontweight="bold",
    )

    _plot_bandwidth_panel(
        axes[0, 0],
        bandwidth,
        "Stream_Triad_DR",
        "(a) STREAM Triad: ancho de banda",
        ranks,
    )
    _plot_bandwidth_panel(
        axes[0, 1],
        bandwidth,
        "DotProduct_DR",
        "(b) Dot Product: ancho de banda",
        ranks,
    )
    _plot_speedup_panel(
        axes[0, 2],
        summary,
        "DotProduct_DR",
        "(c) Dot Product: speedup",
        ranks,
    )
    _plot_speedup_panel(
        axes[1, 0],
        summary,
        "Reduce_DR",
        "(d) Reduce: speedup",
        ranks,
    )
    _plot_speedup_panel(
        axes[1, 1],
        summary,
        "Inclusive_Scan_DR",
        "(e) Inclusive Scan: speedup",
        ranks,
    )
    _plot_speedup_panel(
        axes[1, 2],
        summary,
        "BlackScholes_DR",
        "(f) Black-Scholes: speedup",
        ranks,
    )

    out_path = out_dir / "fig2_paper_panels.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_gemv(summary: pd.DataFrame, out_dir: Path) -> None:
    """GEMV (GemvEq): tiempo de ejecución y speedup vs rangos MPI."""
    series = _speedup_series(summary, "GemvEq_DR")
    if summary.empty or series is None:
        print("  AVISO: sin datos de GemvEq_DR")
        return

    ranks = _ranks_from(summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        series["ranks"],
        series["time_ms"],
        marker="o",
        linewidth=2.0,
        markersize=7,
        color=COLORS[1],
    )
    axes[0].set_title(
        "(a) GEMV (GemvEq): tiempo de ejecución",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Rangos MPI (N)", fontsize=11)
    axes[0].set_ylabel("Tiempo (ms, media)", fontsize=11)
    axes[0].set_xticks(ranks)
    axes[0].set_xlim(min(ranks) - 0.3, max(ranks) + 0.3)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    _plot_speedup_panel(
        axes[1],
        summary,
        "GemvEq_DR",
        "(b) GEMV (GemvEq): speedup",
        ranks,
    )

    fig.suptitle(
        "GEMV (GemvEq) — Distributed Ranges (mhp / OpenMPI)",
        fontsize=14,
        fontweight="bold",
    )
    out_path = out_dir / "fig3_gemv_time_speedup.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_zerocost(zerocost: pd.DataFrame, out_dir: Path) -> None:
    """mhp vs referencias Serial/Std a N=1: tiempos y ratio normalizado."""
    if zerocost.empty:
        print("  AVISO: sin datos para la comparativa de costo cero")
        return

    if "valid" in zerocost.columns:
        invalid = zerocost[~zerocost["valid"]]
        if not invalid.empty:
            names = ", ".join(invalid["benchmark"].tolist())
            print(
                "  AVISO: referencias no válidas (posible optimización del "
                f"compilador), excluidas: {names}"
            )
        data = zerocost[zerocost["valid"]].copy()
    else:
        data = zerocost.dropna(subset=["ref_time_ms"]).copy()

    if data.empty:
        print("  AVISO: no hay pares DR/referencia válidos")
        return

    labels = [_pretty_label(b) for b in data["benchmark"]]
    x = np.arange(len(labels))
    width = 0.38
    ratios = data["dr_time_ms"].to_numpy() / data["ref_time_ms"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    bars_dr = ax.bar(
        x - width / 2,
        data["dr_time_ms"],
        width,
        label="mhp (Distributed Ranges)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        data["ref_time_ms"],
        width,
        label="Referencia Serial/Std",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax.set_yscale("log")
    positive = data[["dr_time_ms", "ref_time_ms"]].to_numpy().ravel()
    positive = positive[positive > 0]
    if positive.size:
        ax.set_ylim(bottom=float(positive.min()) * 0.5)

    for bar, overhead in zip(bars_dr, data["overhead_pct"]):
        if pd.isna(overhead):
            continue
        sign = "+" if overhead >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            f"{sign}{overhead:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )

    ax.set_title(
        "(a) Tiempo absoluto (escala log)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Tiempo (ms, media)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    colors = ["#2ca02c" if ratio <= 1.0 else "#d62728" for ratio in ratios]
    bars_ratio = ax.bar(x, ratios, width * 1.5, color=colors, alpha=0.85)
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Referencia = 1.0",
    )
    for bar, overhead in zip(bars_ratio, data["overhead_pct"]):
        if pd.isna(overhead):
            continue
        sign = "+" if overhead >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{sign}{overhead:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )
    ax.set_title(
        "(b) Tiempo normalizado: mhp / referencia",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Tiempo mhp / referencia", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Abstracción de Costo Cero: mhp vs Serial/Std (N=1)",
        fontsize=14,
        fontweight="bold",
    )
    out_path = out_dir / "fig4_zerocost_comparison.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_efficiency(summary: pd.DataFrame, out_dir: Path) -> None:
    """Eficiencia paralela E_p = S_p / N (figura complementaria)."""
    if summary.empty:
        return

    dr = summary[
        summary["benchmark"].str.contains("_DR", regex=False)
        & (summary["ranks"] > 1)
    ]
    if dr.empty:
        print("  AVISO: sin datos para el heatmap de eficiencia")
        return

    matrix = dr.pivot(
        index="benchmark", columns="ranks", values="efficiency"
    ).sort_index()
    labels = [_pretty_label(b) for b in matrix.index]
    ranks = list(matrix.columns)

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.45 + 1)))
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )

    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"N={n}" for n in ranks])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        "Eficiencia Paralela $E_p = S_p / N$\n(1.0 = perfecto)", fontsize=13
    )

    plt.colorbar(image, ax=ax, fraction=0.03, pad=0.04)

    values = matrix.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            text_color = "black" if 0.25 < value < 0.75 else "white"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    out_path = out_dir / "fig5_parallel_efficiency.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directorio con los CSV (default: results/)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directorio para PNG y resúmenes (default: results/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  paper_plots.py — Distributed Ranges")
    print(f"  Leyendo CSVs desde: {results_dir.resolve()}")
    print(f"  Guardando salidas en: {out_dir.resolve()}")
    print("=" * 60)

    data = load_scaling_data(results_dir)
    if not data:
        print("\nERROR: no se encontraron CSV scaling_n*.csv")
        print("       Ejecuta primero: ./run_experiments.sh")
        sys.exit(1)

    print(f"\n  Datos cargados para N = {sorted(data.keys())}")
    print("\n▶ Calculando métricas...")

    summary = build_scaling_summary(data)
    if summary.empty:
        print("  AVISO: no hay N=1; se omite speedup/eficiencia", file=sys.stderr)
    else:
        summary_path = out_dir / "summary_strong_scaling.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  ✓ {summary_path}")

    bandwidth = build_bandwidth_summary(data)
    if not bandwidth.empty:
        bandwidth_path = out_dir / "summary_bandwidth.csv"
        bandwidth.to_csv(bandwidth_path, index=False)
        print(f"  ✓ {bandwidth_path}")

    zerocost = pd.DataFrame()
    zerocost_path = results_dir / "zerocost_comparison.csv"
    if zerocost_path.exists():
        try:
            zerocost_df = load_benchmark_file(zerocost_path)
            zerocost = build_zerocost_summary(zerocost_df)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  AVISO: no se pudo procesar {zerocost_path}: {exc}",
                file=sys.stderr,
            )
    else:
        print(f"  AVISO: {zerocost_path} no encontrado", file=sys.stderr)

    if not zerocost.empty:
        zerocost_summary_path = out_dir / "summary_zerocost.csv"
        zerocost.to_csv(zerocost_summary_path, index=False)
        print(f"  ✓ {zerocost_summary_path}")

    print("\n▶ Generando figuras...")
    plot_paper_panels(summary, bandwidth, out_dir)
    plot_gemv(summary, out_dir)
    plot_zerocost(zerocost, out_dir)
    plot_efficiency(summary, out_dir)

    print("\n" + "=" * 60)
    print("  ✓ Proceso completado. Archivos generados:")
    generated = [
        out_dir / "fig2_paper_panels.png",
        out_dir / "fig3_gemv_time_speedup.png",
        out_dir / "fig4_zerocost_comparison.png",
        out_dir / "fig5_parallel_efficiency.png",
        out_dir / "summary_strong_scaling.csv",
        out_dir / "summary_bandwidth.csv",
        out_dir / "summary_zerocost.csv",
    ]
    for path in generated:
        if not path.exists():
            continue
        size_kb = path.stat().st_size // 1024
        print(f"    {path.name}  ({size_kb} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
