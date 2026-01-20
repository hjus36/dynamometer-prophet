# 다이나모미터 상태 모니터링 & 이상 예측 (Prophet)

CSV 데이터를 정리하고( run 분리/시간축 정리 ),  
센서 관계 분석 → Prophet 예측 → 이상치(Anomaly) 탐지까지 진행합니다.

---

## 데이터 정의
(※ 상세 표는 `tables/Day0/Day0_data_dictionary.csv` 참고)

| 구분 | 컬럼(그룹) | 신호 설명(추정) | 단위(가정) | 스케일/변환(가정) | 메모 |
|---|---|---|---|---|---|
| 시간 | `Time` | 각 run 내 상대 시간 | ms | 그대로 사용 | run마다 0부터 반복 |
| 토크 | `FB_Torque` | 피드백 토크(부하/제동 방향 가능) | N·m | 값 ÷ 10 | 음수는 흡수(제동) 토크 가정 |
| RPM | `FB_Rpm` | 분당 회전수 | rpm | 그대로 사용 | 저속 구간 중심 |
| 온도 | `Temp00~Temp19` | 온도 센서 20채널 | °C | 값 ÷ 10 | 대표값 생성 (평균) |
| 압력 | `Press00~Press15` | 압력 센서 16채널 | kPa(추정) | 값 ÷ 10 | 대표값 생성 (평균) |
| 진동 | `Vib_X,Y,Z` | 3축 진동(가속도) 출력 | (상대값) | 환산 보류 | RMS로 변화량/크기 지표 사용 |

---

## 폴더 구조
- data/raw: 원본 CSV(수정 X, GitHub에는 미포함)
- data/interim: 중간 정리 데이터(GitHub에는 미포함)
- data/processed: 정리본(로컬 생성, GitHub 미포함)
- notebooks: Day1~Day5 분석 기록
- src: 파이썬 코드(정리/특징/시각화/모델/평가)
- figures: Day별 그래프 저장
- tables: 분석 결과 표 저장(CSV)
- reports: 보고서(draft/final)

---

## 소스코드(src) 구성
- `src/data_make.py`: 원본 CSV 로딩/정리 → run_id 생성 → abs_time_ms 생성 → 스케일 가정 적용 → 대표 센서(feature) 추가 → processed CSV 저장
- `src/data_summary.py`: 전처리된 데이터의 기본 요약(컬럼/결측치/기초 통계/run 요약 등)을 계산해 `tables/Day1`에 저장
- `src/day1_figures_abs_time.py`: 전체 abs_time_ms 기준으로 주요 센서(RPM/토크/압력/온도/진동 등) 시간그래프를 생성해 `figures/Day1`에 저장
- `src/features.py`: Temp/Press 다채널 및 진동축(Vib)에서 대표값(평균/크기 지표 등)과 파생변수를 생성하는 함수 모음
- `src/day2_relationship.py`: 센서 간 관계 분석 수행 → 토크 기준으로 RPM/진동/온도/압력 산점도 4개를 `figures/Day2`에 저장하고, 전체 구간 상관표(corr_all.csv)와 가동 구간(RPM>0) 상관표(corr_running.csv)를 `tables/Day2`에 저장
---

## Data
- 원본 CSV는 용량/관리 이유로 GitHub에 포함하지 않음.
- 로컬에 `data/raw/Basic_Model_20251015_15.csv`로 배치 후 실행.
- 전처리 실행: `python -m src.data_make`
  → `data/processed/processed_v1_run_abs_features.csv` 생성

---

## 진행 요약
- Day0 준비(구조/정의서/전처리)
- Day1 정리+기초그래프
- Day2 관계(산점도/상관)
- Day3 단변량 예측+MAE
- Day4 다변량 예측 비교
- Day5 이상치+최종 보고서
