# 다이나모미터 상태 모니터링 & 이상 예측 (Prophet)

CSV 데이터를 정리하고( run 분리/시간축 정리 ),  
센서 관계 분석 → Prophet 예측 → 이상치(Anomaly) 탐지까지 진행합니다.

가정: FB_Torque 음수는 제동/부하(흡수) 토크일 가능성이 높다고 보고 분석합니다
(부호 정의는 장비 설정에 따라 달라질 수 있음)

---

## 폴더 구조
- data/raw: 원본 CSV(수정 X, GitHub에는 미포함)
- data/interim: 중간 정리 데이터(GitHub에는 미포함)
- data/processed: 정리본(로컬 생성, GitHub 미포함)
- notebooks: Day1~Day5 분석 기록
- src: 재사용 코드(정리/특징/시각화/모델/평가)
- figures: Day별 그래프 저장
- tables: 표 저장(상관/지표 등, CSV는 대부분 미포함 / Day0 데이터 사전만 포함)
- reports: 보고서(draft/final)

---

## Data
- 원본 CSV는 용량/관리 이유로 GitHub에 포함하지 않음.
- 로컬에 `data/raw/Basic_Model_20251015_15.csv`로 배치 후 실행.
- 전처리 실행: `src/data_make.py` → `data/processed/processed_v1_run_abs_features.csv` 생성

---

## 진행 요약
- Day0 준비(구조/정의서/전처리)
- Day1 정리+기초그래프
- Day2 관계(산점도/상관)
- Day3 단변량 예측+MAE
- Day4 다변량 예측 비교
- Day5 이상치+최종 보고서