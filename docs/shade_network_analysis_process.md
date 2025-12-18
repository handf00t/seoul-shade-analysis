# 가로수 그늘 연결성 분석 프로세스

## 개요

서울시 가로수 데이터를 기반으로 보행자 관점의 그늘 연결성을 분석하는 파이프라인입니다.
SCI(Shade Connectivity Index) 방식을 사용하여 실제 보행 경험을 반영한 그늘 품질을 평가합니다.

## 분석 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 데이터 수집                                                  │
│     seoul_trees_unified.py                                      │
│     └─→ trees_unified.geojson (서울시 전체 나무)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. 데이터 전처리                                                │
│     seoul_trees_data_cleaning.py                                │
│     └─→ 이상치 제거, 좌표 검증                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. 구별 추출                                                    │
│     extract_district_trees.py                                   │
│     └─→ {구이름}_roadside_trees.csv                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 도로명 추출                                                  │
│     extract_road_names.py                                       │
│     └─→ {구이름}_roadside_trees_with_roads.csv                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. SCI 분석 ⭐                                                  │
│     segment_sci_analyzer.py                                     │
│     ├─→ 세그먼트 분리 (gap/lateral 기준)                        │
│     ├─→ 인접 세그먼트 병합 (min_length 200m)                    │
│     └─→ {district}_sci_segments.csv                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. 시각화                                                       │
│     shade_sci_map_visualizer.py                                 │
│     ├─→ 필터링 (min_trees=3, min_length=30m)                    │
│     └─→ {district}_sci_map.html (인터랙티브 지도)               │
└─────────────────────────────────────────────────────────────────┘
```

## 핵심 분석: SCI (Shade Connectivity Index)

### 개념

SCI는 보행자가 도로를 걸을 때 **그늘로 덮인 거리의 비율**을 측정합니다.

```
나무1 ────●────  gap  ────●──── 나무2
     [그늘반경]  [노출]  [그늘반경]

SCI = Coverage Ratio = 그늘 거리 / 전체 거리 (0~1, 높을수록 좋음)
```

### 그늘 연결 판정

```
두 나무 사이 거리: d
나무1 그늘 반경: r1 = (수관폭1 / 2) × shade_factor
나무2 그늘 반경: r2 = (수관폭2 / 2) × shade_factor

실제 gap = d - r1 - r2

if gap ≤ max_gap (8m):
    → 연결됨 (Covered)
else:
    → 끊어짐 (Exposed)
```

### 분석 단계

```
1. 도로별 그룹화
   └─→ road_name 기준

2. 도로 양측 분리 (PCA)
   └─→ 주 방향의 수직 성분으로 A/B측 분리

3. 도로 방향 정렬 (PCA main direction)
   └─→ 나무들을 도로 방향으로 정렬 후 분석

4. 각 측면 내 세그먼트 분리
   └─→ gap > 25m (교차로) 또는 측면거리 > 6m (다른 열)

5. 인접 세그먼트 병합 ⭐ (v2 추가)
   └─→ 200m 미만 세그먼트를 인접 세그먼트와 병합
   └─→ 병합 조건: lateral gap ≤ 6m, longitudinal gap ≤ 50m

6. 세그먼트별 SCI 계산
   └─→ Coverage Ratio, 노출 거리, gap 수 등

7. 등급 부여
   └─→ Excellent ~ Very Poor
```

## 파라미터 설정

### 세그먼트 분리 파라미터 (`SegmentSCIAnalyzer`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `shade_factor` | 0.65 | 그늘 반경 = 수관폭/2 × 0.65 (태양 각도 고려) |
| `max_gap` | 8.0m | 연결로 인정하는 최대 gap (약 10걸음) |
| `segment_split_gap` | 25.0m | 세그먼트 분리 gap (교차로/횡단보도) |
| `max_lateral_distance` | 6.0m | 측면 거리 임계값 (보도 폭) ⭐ 변경됨 |
| `lat_to_m` | 111000 | 위도 → 미터 변환 계수 |
| `lon_to_m` | 88740 | 경도 → 미터 변환 (서울 위도 기준) |

### 세그먼트 병합 파라미터 (`_merge_adjacent_segments`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `min_length` | 200m | 최소 세그먼트 길이 (미만 시 병합 시도) |
| `max_lateral_gap` | 6m | 병합 허용 횡단 방향 거리 ⭐ 변경됨 |
| `max_longitudinal_gap` | 50m | 병합 허용 도로 방향 거리 |

### 시각화 파라미터 (`create_sci_map`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `min_trees` | 3 | 표시할 최소 나무 수 |
| `min_length` | 30m | 표시할 최소 세그먼트 길이 ⭐ 추가됨 |

## 등급 기준

| 등급 | Coverage Ratio | 의미 |
|------|---------------|------|
| Excellent | ≥ 95% | 거의 완벽한 그늘 연결 |
| Good | 85% ~ 95% | 양호한 그늘 연결 |
| Fair | 70% ~ 85% | 보통 수준 |
| Poor | 50% ~ 70% | 개선 필요 |
| Very Poor | < 50% | 심각한 그늘 부족 |

## 실행 방법

### 새로운 구 분석

```python
import pandas as pd
import sys
sys.path.insert(0, 'src/4_shade_analysis')
sys.path.insert(0, 'src/5_visualization')

from segment_sci_analyzer import SegmentSCIAnalyzer
from shade_sci_map_visualizer import create_sci_map

# 1. 데이터 로드
trees_df = pd.read_csv('seoul_trees_output/{구이름}_roadside_trees_with_roads.csv')

# 2. SCI 분석
analyzer = SegmentSCIAnalyzer()
segments_df = analyzer.analyze(trees_df)

# 3. 결과 저장
segments_df.to_csv('shade_network_output/{구이름}_sci_segments.csv', index=False)

# 4. 지도 생성 (min_length=30m 적용)
create_sci_map(segments_df, trees_df, 'shade_network_output/{구이름}_sci_map.html',
               min_trees=3, min_length=30)
```

### CLI 실행

```bash
# SCI 분석
python src/4_shade_analysis/segment_sci_analyzer.py

# 지도 생성
python src/5_visualization/shade_sci_map_visualizer.py
```

## 출력 파일

### SCI 세그먼트 CSV (`{district}_sci_segments.csv`)

| 컬럼 | 설명 |
|------|------|
| `segment_name` | 세그먼트 ID (도로명_측면_번호) |
| `original_road` | 원래 도로명 |
| `side` | 도로 측면 (A/B) |
| `tree_count` | 포함된 나무 수 |
| `total_length_m` | 세그먼트 총 길이 (m) |
| `covered_length_m` | 그늘 거리 (m) |
| `exposed_length_m` | 노출 거리 (m) |
| `coverage_ratio` | Coverage 비율 (0~1) |
| `sci` | Shade Connectivity Index (= coverage_ratio) |
| `max_gap_m` | 최대 gap 길이 |
| `avg_gap_m` | 평균 gap 길이 |
| `n_gaps` | gap 개수 |
| `avg_canopy` | 평균 수관폭 |
| `avg_spacing` | 평균 나무 간격 |
| `center_lon/lat` | 세그먼트 중심 좌표 |
| `min/max_lon/lat` | 세그먼트 경계 좌표 |
| `polyline` | 시각화용 경로 (JSON) |
| `rating` | 등급 (Excellent~Very Poor) |
| `rank` | Coverage 순위 |

### 인터랙티브 지도 (`{district}_sci_map.html`)

- Coverage Ratio에 따른 색상 표시 (초록~빨강)
- 세그먼트 클릭시 상세 정보 팝업
- 등급별 레이어 필터링
- 개별 나무 표시 (옵션)
- **30m 미만 세그먼트는 시각화에서 제외** ⭐

## 분석 결과 예시

### 강남구 (2024-12-17) ⭐ 업데이트

| 지표 | 값 |
|------|-----|
| 총 나무 | 21,097그루 |
| 분석 포함 나무 | 12,803그루 (60.7%) |
| 총 세그먼트 | 1,770개 |
| 시각화 세그먼트 | 658개 (30m 이상) |
| 평균 Coverage Ratio | 85.7% |
| 총 그늘 거리 | 72.8km |
| 총 노출 거리 | 21.4km |

#### 등급별 분포
- Excellent (≥95%): 1,088개 (61.5%)
- Good (85-95%): 113개 (6.4%)
- Fair (70-85%): 173개 (9.8%)
- Poor (50-70%): 213개 (12.0%)
- Very Poor (<50%): 183개 (10.3%)

#### Best 도로
1. 일원동길: 98.3%
2. 압구정로: 95.7%
3. 도곡중앙길: 94.7%

#### 개선 필요 도로
1. 탄천길: 35.3%
2. 선릉아래길: 58.1%
3. 가로수길: 67.9%

### 마포구
- 총 세그먼트: 1,516개
- 포함 나무: 7,963그루 (67.2%)
- 평균 Coverage Ratio: 76.8%

### 서대문구
- 총 세그먼트: 970개
- 포함 나무: 5,523그루 (76.5%)
- 평균 Coverage Ratio: 76.6%
- 총 그늘 거리: 26.1km
- 총 노출 거리: 12.5km

## 주요 파일 위치

```
Archive/
├── src/
│   ├── 4_shade_analysis/
│   │   └── segment_sci_analyzer.py    # SCI 분석기 ⭐
│   └── 5_visualization/
│       └── shade_sci_map_visualizer.py # 지도 시각화 ⭐
│
├── seoul_trees_output/                 # 나무 데이터
│   ├── 강남구_roadside_trees_with_roads.csv
│   ├── 마포구_roadside_trees_with_roads.csv
│   └── 서대문구_roadside_trees_with_roads.csv
│
└── shade_network_output/               # 분석 결과
    ├── gangnam_sci_segments.csv
    ├── gangnam_sci_map.html
    ├── mapo_sci_segments.csv
    ├── mapo_sci_map.html
    ├── seodaemun_sci_segments.csv
    └── seodaemun_sci_map.html
```

## 알고리즘 상세

### 1. 도로 양측 분리 (`_split_by_road_side`)

```python
# PCA로 도로 주 방향 계산
pca = PCA(n_components=2)
pca.fit(coords)  # coords: 나무 좌표 (미터)

main_direction = pca.components_[0]    # 도로 방향
perpendicular = pca.components_[1]     # 수직 방향

# 각 나무의 측면 거리 계산
lateral_distances = np.dot(coords - center, perpendicular)

# 중앙값 기준 분리
median = np.median(lateral_distances)
side_a = lateral_distances <= median
side_b = lateral_distances > median
```

### 2. 세그먼트 분리 (`_split_at_large_gaps`) ⭐ 업데이트

```python
# PCA로 도로 방향 계산
pca = PCA(n_components=2)
pca.fit(coords)
main_direction = pca.components_[0]   # 도로 방향
perpendicular = pca.components_[1]    # 수직 방향

# 도로 방향으로 정렬 (x좌표 정렬 대신) ⭐ 추가
projections = np.dot(coords - center, main_direction)
sort_idx = np.argsort(projections)
coords = coords[sort_idx]  # 정렬된 좌표로 분석

# 분리 조건 체크
for i in range(n - 1):
    gap = distance[i] - shade_radius[i] - shade_radius[i+1]
    lateral = perpendicular_distance(tree[i], tree[i+1])

    if gap > segment_split_gap:       # 25m 초과 → 분리
        split_points.append(i + 1)
    elif lateral > max_lateral_distance:  # 6m 초과 → 분리 ⭐ 변경
        split_points.append(i + 1)
```

### 3. 인접 세그먼트 병합 (`_merge_adjacent_segments`) ⭐ 추가

```python
# 같은 도로, 같은 측면의 세그먼트를 도로 방향 순으로 정렬
for (road, side), group in segments_df.groupby(['original_road', 'side']):
    # PCA로 도로 방향 계산
    pca = PCA(n_components=2)
    main_direction = pca.components_[0]
    perpendicular = pca.components_[1]

    # 도로 방향으로 정렬
    projections = np.dot(coords, main_direction)
    group = group.iloc[np.argsort(projections)]

    # 인접 세그먼트 병합
    for seg in group:
        # 이전 세그먼트와의 거리 체크
        lateral_dist = abs(np.dot(diff, perpendicular))
        longitudinal_dist = abs(np.dot(diff, main_direction))

        if lateral_dist > 6 or longitudinal_dist > 50:
            # 거리가 멀면 병합 차단, 새 세그먼트 시작
            flush_buffer()
        else:
            # 버퍼에 추가
            buffer.append(seg)

        # 200m 이상이면 flush
        if buffer_length >= 200:
            flush_buffer()
```

### 4. SCI 계산 (`_calculate_sci`)

```python
for i in range(n - 1):
    dist = distance(tree[i], tree[i+1])
    coverage = shade_radius[i] + shade_radius[i+1]
    gap = max(0, dist - coverage)

    if gap <= max_gap:  # 8m 이내
        covered_length += dist
    else:
        covered_length += coverage
        exposed_length += gap

coverage_ratio = covered_length / total_length  # SCI 값
```

## 변경 이력

### v2.1 (2024-12-17) ⭐ 현재 버전

**파라미터 변경**
- `max_lateral_distance`: 12m → 6m (보도 폭 기준으로 조정)
- `max_lateral_gap` (병합): 15m → 6m (지그재그 방지)

**알고리즘 개선**
- `_split_at_large_gaps`: PCA main direction으로 정렬 후 lateral 체크
  - 기존: x좌표 정렬 → 도로 방향과 불일치 가능
  - 변경: 도로 방향(PCA)으로 정렬 → 정확한 lateral 계산

**시각화 개선**
- `min_length=30m` 필터 추가 (짧은 세그먼트 제외)
- 시각화 세그먼트 수: 1,770개 → 658개 (30m 이상만)

### v2.0 (2024-12-16)
- 인접 세그먼트 병합 기능 추가 (`_merge_adjacent_segments`)
- 200m 최소 길이 기준 도입
- polyline 기반 실제 거리 계산

### v1.0 (2024-12-15)
- SCI 분석 기본 구현
- 도로 양측 분리 (PCA)
- Coverage Ratio 기반 등급 시스템

## 한계 및 개선점

1. **태양 각도 미반영**: 현재 고정 shade_factor(0.65) 사용, 시간대별 그늘 변화 미반영
2. **건물 그늘 미포함**: 가로수 그늘만 분석, 건물 음영 미고려
3. **지형 미반영**: 경사면에서의 그늘 변화 미고려
4. **계절 미반영**: 낙엽수의 계절별 그늘 변화 미고려
5. **나무 누락**: 짧은 세그먼트(<200m) 병합 불가 시 분석에서 제외 (약 39%)

## 참고

- 좌표 변환: 서울 위도(37.5°)에서 위도 1° ≈ 111km, 경도 1° ≈ 88.74km
- shade_factor 0.65: 정오 기준 태양 고도각에서의 그늘 투사 비율 추정
- max_lateral_distance 6m: 일반적인 보도 폭(3-4m) + 여유 고려
