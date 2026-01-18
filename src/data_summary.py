# 컬럼 목록/결측치/기본 통계/run별 길이/time 이상/극단값을 tables/Day1/에 요약 파일로 뽑아줌.
from __future__ import annotations

from pathlib import Path
import pandas as pd


KEY_COLS = [
    "Time", "run_id",
    "FB_Rpm", "FB_Torque",
    "Temp_mean", "Press_mean", "Vib_rms",
]

def main():
    processed_path = Path("data/processed/processed_v1_run_abs_features.csv")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"processed 파일이 없습니다: {processed_path}\n"
            f"먼저 `python -m src.data_make` 실행해서 생성하세요."
        )

    df = pd.read_csv(processed_path)
    out_dir = Path("tables/Day1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 컬럼 목록
    columns_txt = out_dir / "columns.txt"
    columns_txt.write_text("\n".join(df.columns), encoding="utf-8")

    # 2) 결측치 요약
    na = df.isna().sum().sort_values(ascending=False)
    na_df = na.reset_index()
    na_df.columns = ["column", "na_count"]
    na_df.to_csv(out_dir / "missing_summary.csv", index=False)

    # 3) 기본 통계(키 컬럼)
    exist_key_cols = [c for c in KEY_COLS if c in df.columns]
    desc = df[exist_key_cols].describe(include="all").transpose()
    desc.to_csv(out_dir / "key_stats.csv")

    # 4) run별 길이/시간 범위
    run_summary = (
        df.groupby("run_id")
          .agg(
              n_samples=("Time", "size"),
              time_min=("Time", "min"),
              time_max=("Time", "max"),
              time_unique=("Time", "nunique"),
          )
          .reset_index()
    )
    run_summary.to_csv(out_dir / "run_summary.csv", index=False)

    # 5) Time 이상 체크(각 run 내에서 증가해야 정상)
    # - time_unique가 n_samples보다 작으면 중복 존재 가능
    # - time_max가 기대값(≈4950)에서 크게 벗어나면 run 품질 확인 필요
    time_anomaly = run_summary[
        (run_summary["time_unique"] != run_summary["n_samples"]) |
        (run_summary["time_max"] < 4900) |
        (run_summary["time_max"] > 5100)
    ]
    time_anomaly.to_csv(out_dir / "time_anomaly_runs.csv", index=False)

    # 6) 극단값(상위/하위) 몇 개 뽑기
    extreme_rows = []
    for col in ["FB_Torque", "FB_Rpm", "Temp_mean", "Press_mean", "Vib_rms"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        top = df.loc[s.nlargest(5).index, ["run_id", "Time", col]]
        bot = df.loc[s.nsmallest(5).index, ["run_id", "Time", col]]
        top["type"] = "top"
        bot["type"] = "bottom"
        top["column"] = col
        bot["column"] = col
        extreme_rows.append(top)
        extreme_rows.append(bot)

    if extreme_rows:
        extreme_df = pd.concat(extreme_rows, ignore_index=True)
        extreme_df.to_csv(out_dir / "extremes_top_bottom.csv", index=False)

    print("[OK] Day1 summary saved to:", out_dir)
    print("- columns.txt")
    print("- missing_summary.csv")
    print("- key_stats.csv")
    print("- run_summary.csv")
    print("- time_anomaly_runs.csv")
    print("- extremes_top_bottom.csv")


if __name__ == "__main__":
    main()
