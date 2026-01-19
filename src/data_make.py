# raw CSV → 컬럼 정리(공백 제거/unnamed 제거) → run_id/abs_time_ms 생성
# → (스케일 가정) → 대표 센서(Temp_mean, Press_mean, Vib_*) 추가 → processed 저장.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Iterable

import pandas as pd

from .features import add_representative_sensors


@dataclass(frozen=True)
class DataConfig:
    """
    데이터 처리 규칙 모음.
    """
    # 기본
    drop_unnamed_cols: bool = True
    run_period_ms: int = 5000  # Time: 0~4950ms ≈ 5000ms
    time_col: str = "Time"

    # 스케일 가정 적용 여부
    enable_scale: bool = True

    # prefix 기반 스케일: Temp**, Press** 등
    # 예) {"Temp": 10.0, "Press": 10.0} -> 해당 prefix 컬럼들은 값/10
    scale_prefix_divisor: Dict[str, float] = field(default_factory=lambda: {"Temp": 10.0, "Press": 10.0})

    # 컬럼 단위 스케일: FB_Torque 등
    # 예) {"FB_Torque": 10.0} -> 값/10
    scale_column_divisor: Dict[str, float] = field(default_factory=lambda: {"FB_Torque": 10.0})

    # - False: 변환 보류(그대로 사용)
    # - True : 아래 scale_column_divisor에 "FB_Rpm": 10.0 같은 걸 넣어 적용
    apply_rpm_scale: bool = False

    # RPM 스케일을 True로 둘 경우 사용할 divisor (기본 10)
    rpm_divisor: float = 10.0

    def resolved_scale_column_divisor(self) -> Dict[str, float]:
        """
        apply_rpm_scale 설정을 반영해 최종 column 스케일 맵을 생성한다.
        """
        out = dict(self.scale_column_divisor)
        if self.apply_rpm_scale:
            out["FB_Rpm"] = float(self.rpm_divisor)
        return out


def load_raw_csv(csv_path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """
    원본 CSV 로딩 + 기본 정리(컬럼 공백 제거, unnamed 제거 등).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if cfg.drop_unnamed_cols:
        unnamed = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

    return df


def add_run_id(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Time이 감소하는 지점(예: 4950 -> 0)을 run 경계로 보고 run_id를 생성한다.
    """
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' 컬럼이 없습니다. (컬럼 목록: {list(df.columns)})")

    out = df.copy()
    time_diff = out[time_col].diff()
    run_boundary = (time_diff < 0).fillna(False)
    out["run_id"] = run_boundary.cumsum().astype(int)
    return out


def add_abs_time(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Prophet 입력을 위한 연속 시간축(abs_time_ms) 생성.
    abs_time_ms = run_id * run_period_ms + Time
    """
    if "run_id" not in df.columns:
        raise ValueError("run_id 컬럼이 없습니다. add_run_id()를 먼저 호출하세요.")
    if cfg.time_col not in df.columns:
        raise ValueError(f"'{cfg.time_col}' 컬럼이 없습니다.")

    out = df.copy()
    out["abs_time_ms"] = out["run_id"] * cfg.run_period_ms + out[cfg.time_col]
    return out


def _select_columns_by_prefix(columns: Iterable[str], prefix: str) -> list[str]:
    return [c for c in columns if c.startswith(prefix)]


def apply_scale_assumption(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    스케일 가정 적용(선택).

    - prefix 기반: Temp*, Press* 등 /divisor
    - column 기반: FB_Torque 등 /divisor
    - FB_Rpm은 cfg.apply_rpm_scale로 적용 여부를 결정 (기본: 보류)
    """
    if not cfg.enable_scale:
        return df

    out = df.copy()

    # 1) prefix 기반 스케일
    for prefix, divisor in cfg.scale_prefix_divisor.items():
        cols = _select_columns_by_prefix(out.columns, prefix)
        if not cols:
            continue
        # 안전하게 숫자 변환 후 나누기
        out[cols] = out[cols].apply(pd.to_numeric, errors="coerce") / float(divisor)

    # 2) column 기반 스케일 (+ RPM 옵션 반영)
    col_div = cfg.resolved_scale_column_divisor()
    for col, divisor in col_div.items():
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce") / float(divisor)

    return out


def make_processed_dataset(
    csv_path: str | Path,
    cfg: DataConfig = DataConfig(),
    sort_within_run: bool = True,
) -> pd.DataFrame:
    """
    Day1에서 사용할 '처리된 데이터'를 생성하는 메인 함수.

    흐름:
    1) raw 로딩
    2) run_id 생성
    3) (선택) run_id, Time 정렬
    4) abs_time_ms 생성
    5) 스케일 가정 적용
    6) 대표 센서(feature) 생성
    """
    df = load_raw_csv(csv_path, cfg)

    df = add_run_id(df, time_col=cfg.time_col)

    if sort_within_run:
        df = df.sort_values(["run_id", cfg.time_col], kind="mergesort").reset_index(drop=True)

    df = add_abs_time(df, cfg)
    df = apply_scale_assumption(df, cfg)

    # 대표 센서 추가(Temp_mean, Press_mean, Vib_* 등)
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
    # 예시 실행
    # python -m src.data_make
    cfg = DataConfig(
        enable_scale=True,
        apply_rpm_scale=False,   # Day1에서 확정 전까지는 False 권장
    )

    input_csv = Path("data/raw/Basic_Model_20251015_15.csv")
    output_csv = Path("data/processed/processed_v1_run_abs_features.csv")

    df = make_processed_dataset(input_csv, cfg=cfg)
    save_processed_dataset(df, output_csv)

    print(f"[OK] saved: {output_csv} (rows={len(df)}, cols={len(df.columns)})")
