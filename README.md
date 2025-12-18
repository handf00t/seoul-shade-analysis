# 서울시 가로수 그늘 연결성 분석 프로젝트

## 폴더 구조

```
Archive/
├── src/                          # 소스 코드
│   ├── 1_data_collection/        # 데이터 수집
│   │   ├── seoul_trees_unified.py    # 서울시 나무 통합 수집기 (보호수/가로수/공원수목)
│   │   ├── park_trees_collector.py   # 공원수목 전용 수집기
│   │   └── ecobenefits.py            # 생태적 편익 계산기
│   │
│   ├── 2_preprocessing/          # 데이터 전처리
│   │   ├── seoul_trees_data_quality.py   # 데이터 품질 분석
│   │   ├── seoul_trees_data_cleaning.py  # 데이터 정제
│   │   ├── extract_district_trees.py     # 구별 나무 추출
│   │   ├── extract_road_names.py         # 도로명 추출
│   │   └── extract_species*.py           # 수종 추출
│   │
│   ├── 3_district/               # 행정구역 처리
│   │   ├── seoul_districts.py        # 서울 행정구역 GeoJSON 생성
│   │   └── seoul_districts_group.py  # 구별 나무 수 집계
│   │
│   ├── 4_shade_analysis/         # 그늘 연결성 분석 (현재 사용)
│   │   ├── segment_sci_analyzer.py   # SCI 기반 세그먼트 분석 ⭐
│   │   └── shade_network_builder.py  # 그늘 네트워크 구축
│   │
│   ├── 5_visualization/          # 시각화
│   │   ├── shade_sci_map_visualizer.py  # SCI 지도 시각화 ⭐
│   │   └── shade_map_visualizer.py      # λ₂ 지도 시각화
│   │
│   └── _deprecated/              # 구버전 (참고용)
│       ├── grid_shade_analyzer.py       # 그리드 분석 (deprecated)
│       ├── road_shade_analyzer.py       # 도로 분석 (deprecated)
│       ├── segment_shade_analyzer.py    # λ₂ 분석 (SCI로 대체)
│       └── ...
│
├── data/                         # 원본 데이터
│   ├── shp/                      # Shapefile (행정구역)
│   ├── geojson/                  # GeoJSON 파일
│   └── mbtiles/                  # 타일 데이터
│
├── seoul_trees_output/           # 나무 데이터 출력
│   ├── trees_unified.geojson         # 통합 나무 데이터
│   ├── trees_with_benefits*.geojson  # 편익 계산된 데이터
│   ├── 강남구_roadside_trees*.csv    # 강남구 가로수
│   └── 마포구_roadside_trees*.csv    # 마포구 가로수
│
├── shade_network_output/         # 그늘 분석 출력
│   ├── gangnam_sci_segments.csv      # 강남구 SCI 분석 결과
│   ├── gangnam_sci_map.html          # 강남구 인터랙티브 지도 ⭐
│   └── mapo/                         # 마포구 분석 결과
│
├── data_quality_reports/         # 데이터 품질 보고서
│
├── config/                       # 설정 파일
│
└── docs/                         # 문서
    ├── work_history_20251216.md      # 작업 히스토리
    └── species_list.txt              # 수종 목록
```

## 주요 파일

### 현재 사용 중인 분석 파이프라인

1. **데이터 수집**: `src/1_data_collection/seoul_trees_unified.py`
2. **데이터 정제**: `src/2_preprocessing/seoul_trees_data_cleaning.py`
3. **구별 추출**: `src/2_preprocessing/extract_district_trees.py`
4. **도로명 추출**: `src/2_preprocessing/extract_road_names.py`
5. **SCI 분석**: `src/4_shade_analysis/segment_sci_analyzer.py` ⭐
6. **지도 시각화**: `src/5_visualization/shade_sci_map_visualizer.py` ⭐

### 분석 결과 확인

```bash
# 강남구 SCI 지도 열기
open shade_network_output/gangnam_sci_map.html
```

## 파라미터 설정 (segment_sci_analyzer.py)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| shade_factor | 0.65 | 그늘 반경 = 수관폭/2 × 0.65 |
| max_gap | 8.0m | 연결 허용 gap (10걸음) |
| segment_split_gap | 25.0m | 구간 분리 gap (교차로) |
| max_lateral_distance | 6.0m | 측면 거리 (보도 폭) |

## 최종 결과 (강남구, 2024-12-16)

- 총 세그먼트: 2,724개
- 평균 Coverage Ratio: 89.5%
- 총 그늘 거리: 75.9km
- 총 노출 거리: 20.6km
