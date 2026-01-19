from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = Path("data/processed/processed_v1_run_abs_features.csv")
OUT_DIR = Path("figures/Day1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_UNIT = "s"  # <- "min"으로 바꾸면 분 단위
TIME_DIV = 1000.0 if TIME_UNIT == "s" else 60000.0

df = pd.read_csv(IN_PATH)

# 파생: 토크 절대값(부하 크기 관점)
df["Torque_abs"] = df["FB_Torque"].abs()

# 대표 진동 컬럼 자동 선택
# - Vib_X/Y/Z는 raw 축이라 제외
# - 그 외 "Vib" 포함 컬럼(대표값: Vib_rms 또는 Vib_mag 등) 중 첫 번째 사용
vib_candidates = [c for c in df.columns if "Vib" in c and c not in ["Vib_X", "Vib_Y", "Vib_Z"]]
vib_col = vib_candidates[0] if vib_candidates else None


def save_lineplot(x, y, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(f"abs_time ({TIME_UNIT})")  # 단위 반영
    plt.ylabel(y.name)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=160)
    plt.close()


# x축: abs_time_ms -> abs_time
x = df["abs_time_ms"] / TIME_DIV
tlabel = f"abs_time ({TIME_UNIT})"

# 파일명에서 A_ 제거: abs_*.png 형태로 저장
save_lineplot(x, df["FB_Rpm"], f"FB_Rpm over {tlabel}", "abs_FB_Rpm.png")
save_lineplot(x, df["FB_Torque"], f"FB_Torque over {tlabel}", "abs_FB_Torque.png")
save_lineplot(x, df["Torque_abs"], f"Torque_abs over {tlabel}", "abs_Torque_abs.png")

if "Press_mean" in df.columns:
    save_lineplot(x, df["Press_mean"], f"Press_mean over {tlabel}", "abs_Press_mean.png")

if "Temp_mean" in df.columns:
    save_lineplot(x, df["Temp_mean"], f"Temp_mean over {tlabel}", "abs_Temp_mean.png")

if vib_col:
    save_lineplot(x, df[vib_col], f"{vib_col} over {tlabel}", "abs_Vib_rep.png")
else:
    print("[WARN] representative vibration column not found (no Vib_* besides Vib_X/Y/Z).")

print(f"[OK] Saved figures to: {OUT_DIR} (x-axis unit: {TIME_UNIT})")
