from __future__ import annotations

"""
Day5 (Visualization-first, using PROCESSED data):
- Input: data/processed/processed_v1_run_abs_features.csv
- Detection: Robust-Z ONLY for Vib_rms (sample-level)
- Visualization focus:
  1) day5_vib_robust_score_panel.png     (2-panel: Vib context + |z| score)
  2) day5_multisensor_overview.png       (multi-sensor overview + anomaly timing)
  3) figures/Day5/events/event_XX_*.png  (per-event multi-sensor context around peak)

Fixes in this version:
- Use score timeline (z_abs 1s max) as the ANCHOR timeline.
  -> Avoids sec-offset mismatch caused by inner joins / missing Temp/Press.
- Join other sensors with LEFT join onto the score timeline.
- Apply MIN_EVENT_SPACING_SEC when selecting event plots (no more clustered events).

Notes:
- No running/idle split (as requested).
- VIB_Z_K = 7.0 (as requested).
- CSV output is optional (default OFF).

Run:
  python -m src.day5_anomaly_detection
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
DATA_PATH = Path("data/processed/processed_v1_run_abs_features.csv")

OUT_TAB = Path("tables/Day5")
OUT_FIG = Path("figures/Day5")
OUT_EVT = OUT_FIG / "events"

OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_EVT.mkdir(parents=True, exist_ok=True)


# =========================
# Settings
# =========================
# --- Detection threshold
VIB_Z_K = 7.0

# --- Resample for visualization
PLOT_RESAMPLE_SEC = 1

# --- Event logic (sec-level)
EVENT_MAX_GAP_SEC = 2           # anomaly seconds within <=2s -> one event
CONTEXT_HALF_WINDOW_SEC = 8     # per-event context window: ±8s
TOP_EVENT_PLOTS = 10            # how many event plots to save
MIN_EVENT_SPACING_SEC = 60      # enforce min spacing between selected peak_secs

# --- Vib panel y-lim band (makes small changes visible)
YLIM_USE_MAD_BAND = True
YLIM_MAD_MULT = 8.0

# --- Plot styling
SCORE_LINEWIDTH = 0.6
ANOM_DOT_SIZE = 44
GRID_ALPHA = 0.2

# --- Tables (optional)
SAVE_TABLES = False


# =========================
# Helpers
# =========================
def make_ds_from_abs_ms(abs_time_ms: pd.Series) -> pd.Series:
    base = pd.Timestamp("1970-01-01")
    return base + pd.to_timedelta(abs_time_ms, unit="ms")


def robust_stats(x: pd.Series) -> tuple[float, float]:
    x = x.dropna().to_numpy()
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad


def robust_z(x: pd.Series, med: float, mad: float, eps: float = 1e-9) -> pd.Series:
    denom = mad if mad > 0 else eps
    return 0.6745 * (x - med) / (denom + eps)


def group_anomaly_events(anom_secs: np.ndarray, z_abs_by_sec: pd.Series, max_gap_sec: int) -> pd.DataFrame:
    secs = np.array(sorted(set([int(s) for s in anom_secs.tolist()])))
    if len(secs) == 0:
        return pd.DataFrame(
            columns=["event_id", "start_sec", "end_sec", "duration_sec", "peak_sec", "peak_zabs", "n_points"]
        )

    buckets: list[list[int]] = []
    bucket = [int(secs[0])]
    prev = int(secs[0])
    for s in secs[1:]:
        s = int(s)
        if s - prev <= max_gap_sec:
            bucket.append(s)
        else:
            buckets.append(bucket)
            bucket = [s]
        prev = s
    buckets.append(bucket)

    rows = []
    for i, b in enumerate(buckets, start=1):
        b_arr = np.array(b, dtype=int)
        zvals = z_abs_by_sec.reindex(b_arr).to_numpy()
        if np.all(np.isnan(zvals)):
            peak_sec = int(b_arr[0])
            peak_z = np.nan
        else:
            peak_idx = int(np.nanargmax(zvals))
            peak_sec = int(b_arr[peak_idx])
            peak_z = float(zvals[peak_idx])

        rows.append(
            {
                "event_id": i,
                "start_sec": int(b_arr.min()),
                "end_sec": int(b_arr.max()),
                "duration_sec": int(b_arr.max() - b_arr.min() + 1),
                "peak_sec": peak_sec,
                "peak_zabs": peak_z,
                "n_points": int(len(b_arr)),
            }
        )

    return pd.DataFrame(rows).sort_values("peak_zabs", ascending=False).reset_index(drop=True)


def pick_spaced_top_events(events: pd.DataFrame, top_n: int, min_spacing_sec: int) -> pd.DataFrame:
    """
    Pick events by peak_zabs descending, but enforce a minimum spacing between selected peak_sec.
    """
    if events is None or len(events) == 0:
        return events

    picked = []
    for row in events.sort_values("peak_zabs", ascending=False).itertuples(index=False):
        peak = int(row.peak_sec)
        if all(abs(peak - int(p.peak_sec)) >= min_spacing_sec for p in picked):
            picked.append(row)
        if len(picked) >= top_n:
            break

    if not picked:
        return events.head(0).copy()

    return pd.DataFrame([r._asdict() for r in picked]).reset_index(drop=True)


def _t_from_ds(ds: pd.Series) -> np.ndarray:
    return (ds - ds.iloc[0]).dt.total_seconds().to_numpy()


# =========================
# Plotters
# =========================
def plot_two_panel_signal_and_score(
    df_raw: pd.DataFrame,
    z_abs_raw: pd.Series,
    thr_k: float,
    med: float,
    mad: float,
    outpath: Path,
    title: str,
):
    """
    - signal: 1s mean (context only)  [NO dots here]
    - score : 1s max + threshold + anomaly dots [dots ONLY here]
    """
    df_raw = df_raw.sort_values("ds").copy()
    df_raw["z_abs"] = z_abs_raw.to_numpy()

    df_rs = (
        df_raw.set_index("ds")[["Vib_rms", "z_abs"]]
        .resample(f"{PLOT_RESAMPLE_SEC}s")
        .agg({"Vib_rms": "mean", "z_abs": "max"})
        .reset_index()
    )
    df_rs["is_anom_sec"] = df_rs["z_abs"] >= thr_k
    t_sec = _t_from_ds(df_rs["ds"])

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    # (1) Vib context
    ax1.plot(t_sec, df_rs["Vib_rms"].to_numpy(), linewidth=1.0, label=f"Vib_rms ({PLOT_RESAMPLE_SEC}s mean)")
    if YLIM_USE_MAD_BAND and np.isfinite(med) and np.isfinite(mad) and mad > 0:
        ax1.set_ylim(med - YLIM_MAD_MULT * mad, med + YLIM_MAD_MULT * mad)
    ax1.set_ylabel("Vib_rms")
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc="upper right")
    ax1.text(
        0.01,
        0.95,
        f"k={thr_k} | raw anomalies={int((z_abs_raw >= thr_k).sum())} | seconds flagged={int(df_rs['is_anom_sec'].sum())}",
        transform=ax1.transAxes,
        va="top",
    )

    # (2) |z|
    ax2.plot(t_sec, df_rs["z_abs"].to_numpy(), linewidth=SCORE_LINEWIDTH, label=f"|robust z| ({PLOT_RESAMPLE_SEC}s max)")
    ax2.axhline(thr_k, linestyle="--", linewidth=1.0, label=f"threshold k={thr_k}")

    if df_rs["is_anom_sec"].any():
        m = df_rs["is_anom_sec"].to_numpy()
        ax2.scatter(
            t_sec[m],
            df_rs.loc[m, "z_abs"].to_numpy(),
            s=ANOM_DOT_SIZE,
            alpha=0.95,
            marker="o",
            color="tab:red",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label="anomaly sec",
        )

    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("|z|")
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(loc="upper right")

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_multisensor_overview(df_sec: pd.DataFrame, thr_k: float, outpath: Path, title: str):
    """
    4-panel overview:
      RPM (1s mean)
      Torque (1s mean)
      Vib_rms (1s mean)
      |z| (1s max) + threshold + anomaly dots
    """
    t_sec = df_sec["sec"].to_numpy()
    is_anom = df_sec["is_anom_sec"].to_numpy()

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(t_sec, df_sec["FB_Rpm"].to_numpy(), linewidth=1.0, label="FB_Rpm (1s mean)")
    ax1.set_ylabel("RPM")
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc="upper right")

    ax2.plot(t_sec, df_sec["FB_Torque"].to_numpy(), linewidth=1.0, label="FB_Torque (1s mean)")
    ax2.set_ylabel("Torque")
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(loc="upper right")

    ax3.plot(t_sec, df_sec["Vib_rms"].to_numpy(), linewidth=1.0, label="Vib_rms (1s mean)")
    ax3.set_ylabel("Vib_rms")
    ax3.grid(True, alpha=GRID_ALPHA)
    ax3.legend(loc="upper right")

    ax4.plot(t_sec, df_sec["z_abs"].to_numpy(), linewidth=SCORE_LINEWIDTH, label="|robust z| (1s max)")
    ax4.axhline(thr_k, linestyle="--", linewidth=1.0, label=f"threshold k={thr_k}")
    if is_anom.any():
        ax4.scatter(
            t_sec[is_anom],
            df_sec.loc[is_anom, "z_abs"].to_numpy(),
            s=ANOM_DOT_SIZE,
            alpha=0.95,
            marker="o",
            color="tab:red",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label="anomaly sec",
        )
    ax4.set_xlabel("t (s)")
    ax4.set_ylabel("|z|")
    ax4.grid(True, alpha=GRID_ALPHA)
    ax4.legend(loc="upper right")

    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_event_context(df_sec: pd.DataFrame, thr_k: float, peak_sec: int, outpath: Path, title: str, half_window_sec: int):
    """
    Per-event context plot (±window around peak_sec).
    Panels:
      Vib_rms
      FB_Rpm
      FB_Torque
      (Temp_mean if exists)
      (Press_mean if exists)
      |z| + threshold + anomaly dots
    """
    s0, s1 = peak_sec - half_window_sec, peak_sec + half_window_sec
    w = df_sec[(df_sec["sec"] >= s0) & (df_sec["sec"] <= s1)].copy()
    if len(w) == 0:
        return

    panels = ["Vib_rms", "FB_Rpm", "FB_Torque"]
    if "Temp_mean" in w.columns:
        panels.append("Temp_mean")
    if "Press_mean" in w.columns:
        panels.append("Press_mean")
    panels.append("z_abs")

    n = len(panels)
    fig_h = 2.0 * n + 1.5
    fig = plt.figure(figsize=(14, fig_h))

    t = w["sec"].to_numpy()
    is_anom = w["is_anom_sec"].to_numpy()

    axes = []
    for i, col in enumerate(panels, start=1):
        ax = fig.add_subplot(n, 1, i, sharex=axes[0] if axes else None)
        axes.append(ax)

        if col != "z_abs":
            ax.plot(t, w[col].to_numpy(), linewidth=1.0, label=f"{col} (1s mean)")
            ax.set_ylabel(col)
        else:
            ax.plot(t, w["z_abs"].to_numpy(), linewidth=SCORE_LINEWIDTH, label="|robust z| (1s max)")
            ax.axhline(thr_k, linestyle="--", linewidth=1.0, label=f"threshold k={thr_k}")
            if is_anom.any():
                ax.scatter(
                    t[is_anom],
                    w.loc[is_anom, "z_abs"].to_numpy(),
                    s=ANOM_DOT_SIZE,
                    alpha=0.95,
                    marker="o",
                    color="tab:red",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=6,
                    label="anomaly sec",
                )
            ax.set_ylabel("|z|")

        ax.axvline(peak_sec, linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("t (s) [sec from start]")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    required = {"abs_time_ms", "run_id", "FB_Rpm", "FB_Torque", "Vib_rms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in processed CSV: {missing}")

    df["ds"] = make_ds_from_abs_ms(df["abs_time_ms"])

    # robust stats on ALL samples (no running/idle split)
    med, mad = robust_stats(df["Vib_rms"])
    z_abs = robust_z(df["Vib_rms"], med, mad).abs()

    # ---- build 1s tables (ANCHOR: score timeline)
    sec_mean_cols = ["Vib_rms", "FB_Rpm", "FB_Torque"]
    if "Temp_mean" in df.columns:
        sec_mean_cols.append("Temp_mean")
    if "Press_mean" in df.columns:
        sec_mean_cols.append("Press_mean")

    # (1) score timeline is the anchor (prevents sec mismatch)
    df_score = (
        df.set_index("ds")
        .assign(z_abs=z_abs.to_numpy())
        .resample(f"{PLOT_RESAMPLE_SEC}s")
        .agg({"z_abs": "max"})
    )

    # (2) sensor means
    df_sec_mean = df.set_index("ds")[sec_mean_cols].resample(f"{PLOT_RESAMPLE_SEC}s").mean()

    # (3) anchor + LEFT join
    df_sec = df_score.join(df_sec_mean, how="left").reset_index()
    df_sec = df_sec.sort_values("ds").reset_index(drop=True)

    df_sec["sec"] = (df_sec["ds"] - df_sec["ds"].iloc[0]).dt.total_seconds().astype(int)
    df_sec["is_anom_sec"] = df_sec["z_abs"] >= VIB_Z_K

    # ---- plots (always)
    plot_two_panel_signal_and_score(
        df_raw=df[["ds", "Vib_rms"]].copy(),
        z_abs_raw=z_abs,
        thr_k=VIB_Z_K,
        med=med,
        mad=mad,
        outpath=OUT_FIG / "day5_vib_robust_score_panel.png",
        title=f"Day5 Vib_rms Robust-Z (Signal + |z| Score)  k={VIB_Z_K}",
    )

    plot_multisensor_overview(
        df_sec=df_sec,
        thr_k=VIB_Z_K,
        outpath=OUT_FIG / "day5_multisensor_overview.png",
        title=f"Day5 Multi-sensor Overview (RPM/Torque/Vib + |z|)  k={VIB_Z_K}",
    )

    # ---- events + per-event context plots
    anom_secs = df_sec.loc[df_sec["is_anom_sec"], "sec"].to_numpy()
    z_abs_by_sec = df_sec.set_index("sec")["z_abs"]
    events = group_anomaly_events(anom_secs, z_abs_by_sec, max_gap_sec=EVENT_MAX_GAP_SEC)

    if len(events) > 0:
        picked_df = pick_spaced_top_events(events, top_n=TOP_EVENT_PLOTS, min_spacing_sec=MIN_EVENT_SPACING_SEC)

        # small debug prints (helps confirm spacing is applied)
        print("[EVENTS] top peaks (first 20):")
        print(events[["peak_sec", "peak_zabs", "start_sec", "end_sec", "n_points"]].head(20).to_string(index=False))
        print("[EVENTS] picked peaks:")
        print(picked_df[["peak_sec", "peak_zabs", "start_sec", "end_sec", "n_points"]].to_string(index=False))

        for rank, row in enumerate(picked_df.itertuples(index=False), start=1):
            peak_sec = int(row.peak_sec)
            peak_z = float(row.peak_zabs) if row.peak_zabs == row.peak_zabs else np.nan
            fname = f"event_{rank:02d}_sec{peak_sec:04d}.png"
            plot_event_context(
                df_sec=df_sec,
                thr_k=VIB_Z_K,
                peak_sec=peak_sec,
                outpath=OUT_EVT / fname,
                title=f"Event {rank:02d} | peak_sec={peak_sec} | peak |z|={peak_z:.2f} | window=±{CONTEXT_HALF_WINDOW_SEC}s",
                half_window_sec=CONTEXT_HALF_WINDOW_SEC,
            )

    # ---- optional tables (very minimal)
    if SAVE_TABLES:
        thr = pd.DataFrame(
            [
                {
                    "signal": "Vib_rms",
                    "note": "processed: Vib_rms already computed",
                    "median": med,
                    "mad": mad,
                    "k": VIB_Z_K,
                    "raw_anomalies": int((z_abs >= VIB_Z_K).sum()),
                    "anomaly_seconds": int(df_sec["is_anom_sec"].sum()),
                    "event_count": int(len(events)),
                    "picked_events": int(0 if len(events) == 0 else len(pick_spaced_top_events(events, TOP_EVENT_PLOTS, MIN_EVENT_SPACING_SEC))),
                    "min_event_spacing_sec": MIN_EVENT_SPACING_SEC,
                }
            ]
        )
        thr.to_csv(OUT_TAB / "day5_thresholds_robust.csv", index=False)
        events.to_csv(OUT_TAB / "day5_anomaly_events.csv", index=False)

    print("[DONE] Day5 figures:")
    print(f"  - {OUT_FIG / 'day5_vib_robust_score_panel.png'}")
    print(f"  - {OUT_FIG / 'day5_multisensor_overview.png'}")
    print(f"  - {OUT_EVT} (event plots: picked {TOP_EVENT_PLOTS} with spacing {MIN_EVENT_SPACING_SEC}s)")
    print(f"  anomaly seconds: {int(df_sec['is_anom_sec'].sum())} | events: {len(events)} | k={VIB_Z_K}")


if __name__ == "__main__":
    main()
