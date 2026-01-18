# 전처리 파이프라인. raw CSV → 컬럼 정리(공백 제거) → run_id/abs_time_ms 생성 → (스케일 가정) → 대표 센서(Temp_mean, Press_mean, Vib_rms) 추가 → processed 저장.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .features import add_representative_sensors


@dataclass
class DataConfig:
    """
    Day0(준비단계)에서 확정해두는 데이터 처리 규칙 모음.
    Day1부터는 이 설정을 그대로 써서 일관되게 전처리한다.
    """
    # 원본에서 불필요 컬럼(예: Unnamed: 42) 제거 여부
    drop_unnamed_cols: bool = True

    # run 길이 가정 (Time: 0~4950ms => 약 5000ms)
    run_period_ms: int = 5000

    # Time 단위(가정): ms
    time_unit: str = "ms"

    # 스케일 가정(값/10): 필요하면 True로 두고, Day1에서 실제 확인 후 수정
    scale_div10: bool = True

    # 스케일 적용 대상 컬럼 prefix (Temp/Press만)
    scale_prefixes: tuple[str, ...] = ("Temp", "Press")

    # 스케일(/10) 적용할 컬럼을 명시적으로 지정 (Torque만)
    scale_columns_div10: tuple[str, ...] = ("FB_Torque",)


def load_raw_csv(csv_path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """
    원본 CSV 로딩 + 기본 정리(unnamed 컬럼 제거 등).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if cfg.drop_unnamed_cols:
        # "Unnamed: ..." 형태의 컬럼 제거
        unnamed = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

    return df


def add_run_id(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """
    Time이 감소하는 지점(예: 4950 -> 0)을 run 경계로 보고 run_id를 생성한다.
    """
    df = df.copy()

    # Time 차분이 음수면(감소) 새로운 run 시작
    time_diff = df[time_col].diff()
    run_boundary = (time_diff < 0).fillna(False)

    df["run_id"] = run_boundary.cumsum().astype(int)
    return df


def add_abs_time(df: pd.DataFrame, cfg: DataConfig, time_col: str = "Time") -> pd.DataFrame:
    """
    Prophet 입력을 위한 연속 시간축(abs_time_ms) 생성.
    abs_time_ms = run_id * run_period_ms + Time
    """
    df = df.copy()
    if "run_id" not in df.columns:
        raise ValueError("run_id 컬럼이 없습니다. add_run_id()를 먼저 호출하세요.")

    df["abs_time_ms"] = df["run_id"] * cfg.run_period_ms + df[time_col]
    return df


def apply_scale_assumption(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    값/10 스케일 가정 적용(선택).
    - Temp*, Press* 컬럼은 /10
    - FB_Torque는 컬럼명을 명시해서 /10
    - FB_Rpm은 Day1 오전에 결정할 때까지 변환 보류
    """
    if not cfg.scale_div10:
        return df

    df = df.copy()

    # 1) Temp/Press prefix 적용
    for c in df.columns:
        if c == "Time":
            continue
        if c.startswith(cfg.scale_prefixes):
            df[c] = pd.to_numeric(df[c], errors="coerce") / 10.0

    # 2) 명시 컬럼 적용 (Torque만)
    for c in cfg.scale_columns_div10:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 10.0

    return df



def make_processed_dataset(
    csv_path: str | Path,
    cfg: DataConfig = DataConfig(),
    time_col: str = "Time",
    sort_within_run: bool = True,
) -> pd.DataFrame:
    """
    Day1에서 사용할 '처리된 데이터'를 생성하는 메인 함수.
    흐름:
    1) raw 로딩
    2) run_id 생성
    3) abs_time 생성
    4) 스케일 가정 적용(/10)
    5) 대표 센서(feature) 생성(Temp_mean, Press_mean, Vib_rms)
    """
    df = load_raw_csv(csv_path, cfg)
    df = add_run_id(df, time_col=time_col)

    if sort_within_run:
        # run_id, Time 기준 정렬(그래프/모델링 시 안정적)
        df = df.sort_values(["run_id", time_col], kind="mergesort").reset_index(drop=True)

    df = add_abs_time(df, cfg, time_col=time_col)
    df = apply_scale_assumption(df, cfg)

    # 대표 센서 추가
    df = add_representative_sensors(df)

    return df


def save_processed_dataset(df: pd.DataFrame, out_path: str | Path) -> None:
    """
    processed CSV 저장.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # 예시 실행(경로는 Day1에 본인 환경에 맞게 수정)
    # python -m src.data_make
    cfg = DataConfig()
    input_csv = Path("data/raw/Basic_Model_20251015_15.csv")
    output_csv = Path("data/processed/processed_v1_run_abs_features.csv")

    df = make_processed_dataset(input_csv, cfg=cfg)
    save_processed_dataset(df, output_csv)

    print(f"[OK] saved: {output_csv} (rows={len(df)}, cols={len(df.columns)})")
