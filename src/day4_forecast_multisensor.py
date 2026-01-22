from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# ========= Paths =========
IN_PATH = Path("data/processed/processed_v1_run_abs_features.csv")
FIG_DIR = Path("figures/Day4")
TAB_DIR = Path("tables/Day4")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ========= Settings =========
TARGET = "FB_Torque"
REGRESSORS = ["FB_Rpm", "Press_mean"]

RESAMPLE_SEC = 1
TRAIN_RUN_RATIO = 0.9
RUNNING_RATIO_THRESHOLD = 0.5

# Eval regime filter (레짐 mismatch 제거용)
EVAL_FILTER_ENABLED = True
EVAL_RPM_MIN = 0.0
EVAL_TORQUE_MAX = -50.0

# Standardize regressors (train 통계로 z-score) + clip
STANDARDIZE_REGRESSORS = True
Z_CLIP = 5.0

# Prophet 기본: 시즌성 OFF (Day3/Day4 동일 컨셉)
PROPHET_KW = dict(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)


# ========= Utils =========
def make_ds_from_abs_ms(abs_time_ms: pd.Series) -> pd.Series:
    base = pd.Timestamp("1970-01-01")
    return base + pd.to_timedelta(abs_time_ms, unit="ms")


def resample_seconds(df: pd.DataFrame, sec: int) -> pd.DataFrame:
    if sec <= 0:
        return df.sort_values("ds").copy()
    return (
        df.sort_values("ds")
          .set_index("ds")
          .resample(f"{sec}s")
          .mean(numeric_only=True)
          .reset_index()
    )


def split_by_run_id(df: pd.DataFrame, train_ratio: float):
    runs = np.sort(df["run_id"].unique())
    n = len(runs)
    cut = int(np.floor(train_ratio * n))
    cut = max(1, min(cut, n - 1))
    train_runs = set(runs[:cut])
    test_runs = set(runs[cut:])
    return df[df["run_id"].isin(train_runs)].copy(), df[df["run_id"].isin(test_runs)].copy()


def filter_running_runs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    run_ratio = (
        df.assign(is_running=(df["FB_Rpm"] > 0).astype(int))
          .groupby("run_id")["is_running"]
          .mean()
    )
    keep = run_ratio[run_ratio >= threshold].index.to_numpy()
    print(f"[INFO] running_ratio >= {threshold:.2f} -> keep_runs={len(keep)}/{len(run_ratio)}")
    return df[df["run_id"].isin(keep)].copy()


def apply_eval_filter(test_df: pd.DataFrame) -> pd.DataFrame:
    if not EVAL_FILTER_ENABLED:
        return test_df
    return test_df[(test_df["FB_Rpm"] > EVAL_RPM_MIN) & (test_df[TARGET] < EVAL_TORQUE_MAX)].copy()


def standardize_by_train(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str], z_clip: float):
    mu = train_df[cols].mean(numeric_only=True)
    sd = train_df[cols].std(numeric_only=True).replace(0, 1.0)

    tr = train_df.copy()
    te = test_df.copy()

    tr[cols] = (tr[cols] - mu) / sd
    te[cols] = (te[cols] - mu) / sd

    if z_clip and z_clip > 0:
        tr[cols] = tr[cols].clip(-z_clip, z_clip)
        te[cols] = te[cols].clip(-z_clip, z_clip)

    return tr, te, mu, sd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def plot_compare_with_residuals(
    ds: pd.Series,
    y: np.ndarray,
    yhat_uni: np.ndarray,
    yhat_multi: np.ndarray,
    outpath: Path,
    title: str,
):
    """
    상단: actual vs uni vs multi
    하단: |actual - uni| vs |actual - multi| (잔차 그래프)
    -> multi 개선이 더 잘 보이게 함
    """
    t_sec = (ds - ds.iloc[0]).dt.total_seconds().to_numpy()

    err_uni = np.abs(y - yhat_uni)
    err_multi = np.abs(y - yhat_multi)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_sec, y, label="actual")
    ax1.plot(t_sec, yhat_uni, label="uni (Prophet)")
    ax1.plot(t_sec, yhat_multi, label="multi (Prophet + regressors)")
    ax1.set_ylabel(TARGET)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper center", ncol=3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t_sec, err_uni, label="|actual-uni|")
    ax2.plot(t_sec, err_multi, label="|actual-multi|")

    # multi가 더 좋은 구간을 음영으로 표시(선택 기능)
    better = err_multi < err_uni
    ax2.fill_between(
        t_sec,
        0,
        np.maximum(err_uni, err_multi),
        where=better,
        alpha=0.15,
        label="multi better",
    )

    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("abs error")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right")

    # 제목 잘림 방지
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ========= Main =========
def main():
    df = pd.read_csv(IN_PATH)

    # 필수 컬럼 체크
    required = {"run_id", "abs_time_ms", "FB_Rpm", TARGET, *REGRESSORS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ds 생성
    df["ds"] = make_ds_from_abs_ms(df["abs_time_ms"])
    df = df.sort_values(["run_id", "ds"]).copy()

    # run 레벨 필터(가동 run만)
    df = filter_running_runs(df, RUNNING_RATIO_THRESHOLD)

    # run 기준 split
    train_raw, test_raw = split_by_run_id(df, TRAIN_RUN_RATIO)
    print(f"[STAT] running_ratio (RPM>0): TRAIN={(train_raw['FB_Rpm'] > 0).mean():.3f}, TEST={(test_raw['FB_Rpm'] > 0).mean():.3f}")

    # 리샘플
    train_df = resample_seconds(train_raw, RESAMPLE_SEC)
    test_df = resample_seconds(test_raw, RESAMPLE_SEC)

    # eval 레짐 필터(공정 비교용)
    test_df = apply_eval_filter(test_df)
    if len(test_df) < 50:
        raise ValueError("[ERR] Too few test points after eval filter. Relax torque threshold or disable filter.")

    # 표준화 + 클리핑 (회귀변수만)
    if STANDARDIZE_REGRESSORS:
        train_df, test_df, mu, sd = standardize_by_train(train_df, test_df, REGRESSORS, Z_CLIP)
        pd.DataFrame({"regressor": REGRESSORS, "mean": mu.values, "std": sd.values}).to_csv(
            TAB_DIR / "day4_regressor_standardization.csv", index=False
        )

    # ✅ 평가 데이터는 target+regressors를 한 번에 dropna 해서 uni/multi 동일 구간 비교
    eval_cols = ["ds", TARGET] + REGRESSORS
    eval_df = test_df[eval_cols].dropna().copy()
    if len(eval_df) < 30:
        raise ValueError("[ERR] Too few eval points after dropna. Check missing values in regressors.")

    # ---- Uni ----
    tr_uni = train_df[["ds", TARGET]].dropna().rename(columns={TARGET: "y"}).copy()
    te_uni = eval_df[["ds"]].copy()

    m_uni = Prophet(**PROPHET_KW)
    m_uni.fit(tr_uni)
    fc_uni = m_uni.predict(te_uni)
    yhat_uni = fc_uni["yhat"].to_numpy()

    # ---- Multi (fixed combo) ----
    tr_multi = train_df[["ds", TARGET] + REGRESSORS].dropna().rename(columns={TARGET: "y"}).copy()
    te_multi = eval_df[["ds"] + REGRESSORS].copy()

    m_multi = Prophet(**PROPHET_KW)
    for r in REGRESSORS:
        m_multi.add_regressor(r)
    m_multi.fit(tr_multi)
    fc_multi = m_multi.predict(te_multi)
    yhat_multi = fc_multi["yhat"].to_numpy()

    # ---- Metrics ----
    ytrue = eval_df[TARGET].to_numpy()

    mae_uni = mae(ytrue, yhat_uni)
    mae_multi = mae(ytrue, yhat_multi)

    metrics_df = pd.DataFrame([
        {
            "model": "uni",
            "target": TARGET,
            "resample_sec": RESAMPLE_SEC,
            "eval_filter": bool(EVAL_FILTER_ENABLED),
            "eval_rpm_min": EVAL_RPM_MIN if EVAL_FILTER_ENABLED else "",
            "eval_torque_max": EVAL_TORQUE_MAX if EVAL_FILTER_ENABLED else "",
            "test_n": int(len(ytrue)),
            "mae": mae_uni,
            "rmse": rmse(ytrue, yhat_uni),
            "mape(%)": mape(ytrue, yhat_uni),
            "regressors": "",
        },
        {
            "model": "multi_fixed",
            "target": TARGET,
            "resample_sec": RESAMPLE_SEC,
            "eval_filter": bool(EVAL_FILTER_ENABLED),
            "eval_rpm_min": EVAL_RPM_MIN if EVAL_FILTER_ENABLED else "",
            "eval_torque_max": EVAL_TORQUE_MAX if EVAL_FILTER_ENABLED else "",
            "test_n": int(len(ytrue)),
            "mae": mae_multi,
            "rmse": rmse(ytrue, yhat_multi),
            "mape(%)": mape(ytrue, yhat_multi),
            "regressors": ",".join(REGRESSORS),
        },
    ])
    metrics_df.to_csv(TAB_DIR / "day4_metrics.csv", index=False)

    # ---- Title ----
    title = f"{TARGET} | MAE uni={mae_uni:.4f} vs multi={mae_multi:.4f}"
    if EVAL_FILTER_ENABLED:
        title += f" | eval: RPM>{EVAL_RPM_MIN}, Torque<{EVAL_TORQUE_MAX}"
    title += f" | regs={','.join(REGRESSORS)}"

    # ---- Plot (잔차 포함) ----
    plot_compare_with_residuals(
        ds=eval_df["ds"],
        y=ytrue,
        yhat_uni=yhat_uni,
        yhat_multi=yhat_multi,
        outpath=FIG_DIR / f"forecast_{TARGET}_compare.png",
        title=title,
    )

    print("[OK] Saved:")
    print(f"- {TAB_DIR / 'day4_metrics.csv'}")
    print(f"- {FIG_DIR / f'forecast_{TARGET}_compare.png'}")
    if STANDARDIZE_REGRESSORS:
        print(f"- {TAB_DIR / 'day4_regressor_standardization.csv'}")

    print("\n[RESULT]\n", metrics_df)


if __name__ == "__main__":
    main()
