# Time Axis Plan (abs_time & Prophet ds)

## 문제
원본 데이터의 `Time`은 각 run마다 0부터 반복되는 상대시간이므로,
Prophet 입력으로 그대로 사용하면 시간축이 단조 증가하지 않아 모델 입력에 부적절하다.

## 해결 전략
1) `run_id` 생성
- `Time`이 감소하는 지점(예: 4950 → 0)을 run 경계로 판단
- 규칙: `Time.diff() < 0`이면 새로운 run 시작 → run_id 증가

2) 연속 시간축 생성
- 가정: 1 run은 약 5초 구간
- `abs_time_ms = run_id * 5000 + Time`
  - (Time 단위: ms, 0~4950ms, step=50ms)

3) Prophet 입력 생성(ds, y)
- Prophet 형식: `ds`(datetime), `y`(target)
- `ds`는 `abs_time_ms`를 기준으로 임의의 시작시각(start_datetime)에서 누적 시간을 더해 생성한다.
  - 예: start_datetime = 2026-01-01 00:00:00 (임의)
  - ds = start_datetime + abs_time_ms(milliseconds)

## 비고
- 절대 시간의 실제 의미가 중요한 분석이 아니므로, start_datetime은 임의값이어도 무방하다.
- 핵심은 `ds`가 전체 데이터에서 단조 증가하도록 만드는 것이다.
