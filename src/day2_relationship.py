from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = Path("data/processed/processed_v1_run_abs_features.csv")
FIG_DIR = Path("figures/Day2")
TAB_DIR = Path("tables/Day2")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# 너무 촘촘하면 산점도가 뭉개져서, 1초 단위로 다운샘플(평균) 권장
RESAMPLE_SEC = 1  # 1 또는 5 추천. 원하면 0으로 두고 리샘플 생략 가능

TARGET = "FB_Torque"
CANDIDATES = ["FB_Rpm", "Temp_mean", "Press_mean", "Vib_rms"]


def pick_vib_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Vib_rms가 없으면, Day1과 동일하게 대표 Vib 컬럼을 찾아 Vib_rms로 사용."""
    if "Vib_rms" in df.columns:
        return df
    vib_candidates = [c for c in df.columns if "Vib" in c and c not in ["Vib_X", "Vib_Y", "Vib_Z"]]
    if vib_candidates:
        df = df.copy()
        df["Vib_rms"] = df[vib_candidates[0]]
    return df


def resample_seconds(df: pd.DataFrame, sec: int) -> pd.DataFrame:
    """abs_time_ms를 datetime 인덱스로 만들어 초 단위 평균 리샘플."""
    if sec <= 0:
        return df.copy()

    base = pd.Timestamp("1970-01-01")
    ds = base + pd.to_timedelta(df["abs_time_ms"], unit="ms")

    out = df.copy()
    out["ds"] = ds
    out = out.sort_values("ds")
    out = (
        out.set_index("ds")
           .resample(f"{sec}s")
           .mean(numeric_only=True)
           .reset_index()
    )
    return out


def save_scatter(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str):
    plt.figure()
    plt.scatter(df[xcol], df[ycol], s=8, alpha=0.3)
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    df = pd.read_csv(IN_PATH)
    df = pick_vib_rep(df)

    # 필요한 컬럼만 선별(있는 것만)
    cols = [TARGET] + [c for c in CANDIDATES if c in df.columns]
    missing_target = TARGET not in df.columns
    if missing_target:
        raise ValueError(f"[ERR] TARGET column not found: {TARGET}")

    df2 = resample_seconds(df, RESAMPLE_SEC)

    # 사용할 컬럼만 남기고 결측 제거
    used_cols = [c for c in cols if c in df2.columns]
    df_used = df2[used_cols + (["FB_Rpm"] if "FB_Rpm" in df2.columns and "FB_Rpm" not in used_cols else [])].copy()
    df_used = df_used.dropna(subset=used_cols)

    # 1) 산점도: FB_Torque vs 각 센서
    for c in used_cols:
        if c == TARGET:
            continue
        save_scatter(
            df_used,
            xcol=c,
            ycol=TARGET,
            out_path=FIG_DIR / f"scatter_{TARGET}_vs_{c}.png",
            title=f"{TARGET} vs {c} (resample={RESAMPLE_SEC}s)" if RESAMPLE_SEC > 0 else f"{TARGET} vs {c}"
        )

    # 2) 상관표(전체)
    corr_all = df_used[used_cols].corr(method="pearson")
    corr_all.to_csv(TAB_DIR / "corr_all.csv", index=True)

    # 3) 상관표(가동구간: RPM>0)
    if "FB_Rpm" in df_used.columns:
        df_run = df_used[df_used["FB_Rpm"] > 0].copy()
        if len(df_run) > 10:
            corr_run = df_run[used_cols].corr(method="pearson")
            corr_run.to_csv(TAB_DIR / "corr_running.csv", index=True)
        else:
            print("[WARN] Not enough running samples (FB_Rpm > 0) for corr_running.csv")

    print("[OK] Day2 outputs saved:")
    print(f"- figures: {FIG_DIR}")
    print(f"- tables : {TAB_DIR}")


if __name__ == "__main__":
    main()
