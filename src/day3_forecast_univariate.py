from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

IN_PATH = Path("data/processed/processed_v1_run_abs_features.csv")
FIG_DIR = Path("figures/Day3")
TAB_DIR = Path("tables/Day3")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ===== 설정 =====
RESAMPLE_SEC = 1                      # 1초 평균(Prophet 부담/노이즈 감소)
TARGETS = ["FB_Torque", "Temp_mean"]  # 예측할 센서 1~2개
TRAIN_RUN_RATIO = 0.9                 # Split_Rule: 앞 90% run train / 뒤 10% run test

# 가동 run 선별 기준(run 내 RPM>0 비율)
RUNNING_RATIO_THRESHOLD = 0.5
# ================


def make_ds_from_abs_ms(abs_time_ms: pd.Series) -> pd.Series:
    base = pd.Timestamp("1970-01-01")
    return base + pd.to_timedelta(abs_time_ms, unit="ms")


def resample_seconds(df: pd.DataFrame, sec: int) -> pd.DataFrame:
    """ds 기준으로 sec초 평균 리샘플."""
    if sec <= 0:
        return df.copy()
    df = df.sort_values("ds")
    out = (
        df.set_index("ds")
          .resample(f"{sec}s")
          .mean(numeric_only=True)
          .reset_index()
    )
    return out


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def split_by_run_id(df: pd.DataFrame, train_ratio: float):
    """Split_Rule: run_id 오름차순 기준 마지막 10% run을 test로 hold-out."""
    runs = np.sort(df["run_id"].unique())
    n = len(runs)
    cut = int(np.floor(train_ratio * n))
    cut = max(1, min(cut, n - 1))  # 최소 1개 train / 1개 test 보장

    train_runs = set(runs[:cut])
    test_runs = set(runs[cut:])

    train_df = df[df["run_id"].isin(train_runs)].copy()
    test_df = df[df["run_id"].isin(test_runs)].copy()

    return train_df, test_df, runs[:cut], runs[cut:]


def quick_stats(df: pd.DataFrame, col: str, name: str):
    s = df[col].dropna()
    if len(s) == 0:
        print(f"[STAT] {name} {col}: empty")
        return
    print(
        f"[STAT] {name} {col}: "
        f"n={len(s)}, min={s.min():.3f}, p25={s.quantile(0.25):.3f}, "
        f"median={s.median():.3f}, p75={s.quantile(0.75):.3f}, max={s.max():.3f}, mean={s.mean():.3f}"
    )


def filter_running_runs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    run별로 running_ratio = mean(FB_Rpm > 0)를 계산하여
    running_ratio >= threshold 인 run만 남긴다.
    """
    if "FB_Rpm" not in df.columns:
        raise ValueError("[ERR] FB_Rpm column not found. Cannot compute running_ratio.")

    run_ratio = (
        df.assign(is_running=(df["FB_Rpm"] > 0).astype(int))
          .groupby("run_id")["is_running"]
          .mean()
    )
    keep_runs = run_ratio[run_ratio >= threshold].index.to_numpy()
    print(f"[INFO] running_ratio threshold={threshold:.2f} -> keep_runs={len(keep_runs)} / total_runs={run_ratio.shape[0]}")
    return df[df["run_id"].isin(keep_runs)].copy()


def plot_forecast_relative_time(te: pd.DataFrame, ytrue: np.ndarray, yhat: np.ndarray,
                                target: str, score: float, outpath: Path):
    """
    x축을 ds가 아닌 '상대시간 t(s)'로 그려서 라벨 겹침을 방지.
    """
    # te["ds"] 기준 상대초 계산
    t_sec = (te["ds"] - te["ds"].iloc[0]).dt.total_seconds().to_numpy()

    plt.figure()
    plt.plot(t_sec, ytrue, label="actual")
    plt.plot(t_sec, yhat, label="forecast")
    plt.title(f"{target} forecast (Prophet) | resample={RESAMPLE_SEC}s | MAE={score:.4f}")
    plt.xlabel("t (s)")  # 상대시간(초)
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def run_one_target(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    tr = train_df[["ds", target]].dropna().rename(columns={target: "y"}).copy()
    te = test_df[["ds", target]].dropna().rename(columns={target: "y"}).copy()

    if len(tr) < 50 or len(te) < 20:
        print(f"[SKIP] Not enough data for {target} (train={len(tr)}, test={len(te)})")
        return None

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(tr)

    fc = m.predict(te[["ds"]])
    yhat = fc["yhat"].to_numpy()
    ytrue = te["y"].to_numpy()
    score = mae(ytrue, yhat)

    # x축 겹침 해결: 상대시간으로 플롯
    plot_forecast_relative_time(
        te=te,
        ytrue=ytrue,
        yhat=yhat,
        target=target,
        score=score,
        outpath=FIG_DIR / f"forecast_{target}.png",
    )

    return {
        "target": target,
        "resample_sec": RESAMPLE_SEC,
        "running_ratio_threshold": RUNNING_RATIO_THRESHOLD,
        "train_n": len(tr),
        "test_n": len(te),
        "mae": score,
    }


def main():
    df = pd.read_csv(IN_PATH)

    # 필수 컬럼 체크
    need_cols = {"run_id", "abs_time_ms", "FB_Rpm"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns: {need_cols - set(df.columns)}")

    # ds 만들기
    df["ds"] = make_ds_from_abs_ms(df["abs_time_ms"])
    df = df.sort_values(["run_id", "ds"])

    # 가동 run만 선별 (레짐 mismatch 방지)
    df = filter_running_runs(df, RUNNING_RATIO_THRESHOLD)

    # 선별 후 run 수가 너무 적으면 중단
    n_runs = df["run_id"].nunique()
    if n_runs < 5:
        raise ValueError(f"[ERR] Too few runs after filtering (n_runs={n_runs}). Lower threshold or skip filtering.")

    # Split_Rule: run 기준 split
    train_raw, test_raw, train_runs, test_runs = split_by_run_id(df, TRAIN_RUN_RATIO)
    print(f"[INFO] Train runs: {train_runs[0]} ~ {train_runs[-1]} (n={len(train_runs)})")
    print(f"[INFO] Test  runs: {test_runs[0]} ~ {test_runs[-1]} (n={len(test_runs)})")

    # 분포/레짐 확인 로그
    for col in ["FB_Torque", "FB_Rpm", "Temp_mean", "Press_mean"]:
        if col in df.columns:
            quick_stats(train_raw, col, "TRAIN_RAW")
            quick_stats(test_raw,  col, "TEST_RAW")

    tr_run_ratio = (train_raw["FB_Rpm"] > 0).mean()
    te_run_ratio = (test_raw["FB_Rpm"] > 0).mean()
    print(f"[STAT] running_ratio (RPM>0): TRAIN={tr_run_ratio:.3f}, TEST={te_run_ratio:.3f}")

    # 리샘플은 train/test 각각 따로
    train_df = resample_seconds(train_raw, RESAMPLE_SEC)
    test_df = resample_seconds(test_raw, RESAMPLE_SEC)

    results = []
    for t in TARGETS:
        if t not in df.columns:
            print(f"[SKIP] {t} not found in columns.")
            continue
        r = run_one_target(train_df, test_df, t)
        if r is not None:
            results.append(r)

    if results:
        out = pd.DataFrame(results).sort_values("mae")
        out.to_csv(TAB_DIR / "day3_mae.csv", index=False)
        print("[OK] Saved:")
        print(f"- {TAB_DIR / 'day3_mae.csv'}")
        print(f"- forecasts in {FIG_DIR}")
        print(out)
    else:
        print("[WARN] No results saved (check targets/columns).")


if __name__ == "__main__":
    main()
