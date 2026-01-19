from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = Path("data/processed/processed_v1_run_abs_features.csv")
OUT_DIR = Path("figures/Day1")  # ✅ reports/figures/... 대신 figures/... 로 저장
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    plt.xlabel("abs_time_ms")
    plt.ylabel(y.name)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=160)
    plt.close()

x = df["abs_time_ms"]

save_lineplot(x, df["FB_Rpm"], "FB_Rpm over abs_time_ms", "A_abs_FB_Rpm.png")
save_lineplot(x, df["FB_Torque"], "FB_Torque over abs_time_ms", "A_abs_FB_Torque.png")
save_lineplot(x, df["Torque_abs"], "Torque_abs over abs_time_ms", "A_abs_Torque_abs.png")

if "Press_mean" in df.columns:
    save_lineplot(x, df["Press_mean"], "Press_mean over abs_time_ms", "A_abs_Press_mean.png")

if "Temp_mean" in df.columns:
    save_lineplot(x, df["Temp_mean"], "Temp_mean over abs_time_ms", "A_abs_Temp_mean.png")

if vib_col:
    save_lineplot(x, df[vib_col], f"{vib_col} over abs_time_ms", "A_abs_Vib_rep.png")
else:
    print("[WARN] representative vibration column not found (no Vib_* besides Vib_X/Y/Z).")

print(f"[OK] Saved A-set figures to: {OUT_DIR}")
