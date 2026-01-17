# Train/Test Split Rule (run 기준)

## 목적
시계열/반복 실험(run) 구조에서 데이터 누수(leakage)를 방지하기 위해
행(row) 단위 랜덤 분할은 사용하지 않는다.

## 기본 규칙
- 분할 단위: `run_id`
- 정렬 기준: `run_id` 오름차순 (과거 → 미래)
- 테스트셋 구성: 가장 마지막 구간의 run들을 hold-out

## 권장 설정(기본)
- Train: 앞 90% run
- Test: 뒤 10% run

예)
- 전체 run이 N개라면,
  - train_run_end = floor(0.9 * N) - 1
  - Test runs = train_run_end+1 ~ N-1

## 참고
- 데이터가 적거나 안정적인 평가가 필요하면:
  - Test를 “마지막 K runs”로 고정(K=50/100 등)할 수 있다.
- 모든 평가/검증은 run 기준으로만 수행한다.
