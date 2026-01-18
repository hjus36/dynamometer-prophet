# 특징(파생변수) 생성 모음. Temp_mean, Press_mean, Vib_rms 같은 “대표 센서” 계산을 담당.
import numpy as np
import pandas as pd

TEMP_COLS = [f"Temp{str(i).zfill(2)}" for i in range(20)]     # Temp00~Temp19
PRESS_COLS = [f"Press{str(i).zfill(2)}" for i in range(16)]   # Press00~Press15
VIB_COLS = ["Vib_X", "Vib_Y", "Vib_Z"]

def add_representative_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    대표 센서 컬럼을 생성해서 df에 추가한다.
    - Temp_mean: Temp00~Temp19 평균
    - Press_mean: Press00~Press15 평균
    - Vib_rms: sqrt(Vib_X^2 + Vib_Y^2 + Vib_Z^2)
    """
    df = df.copy()

    # Temp / Press 평균
    df["Temp_mean"] = df[TEMP_COLS].mean(axis=1)
    df["Press_mean"] = df[PRESS_COLS].mean(axis=1)

    # Vib RMS
    df["Vib_rms"] = np.sqrt((df[VIB_COLS] ** 2).sum(axis=1))

    return df
