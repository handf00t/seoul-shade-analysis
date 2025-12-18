# 서울시 가로수 그늘 연결성 분석

서울시 열린데이터 API를 활용하여 가로수 그늘 연결성(Shade Connectivity Index)을 분석하는 프로젝트입니다.

## 설치

```bash
pip install pandas geopandas folium requests
```

## API 키 설정

서울 열린데이터 광장에서 API 키를 발급받으세요: https://data.seoul.go.kr

```bash
export SEOUL_API_KEY='your_api_key'
```

## 폴더 구조

```
src/
├── 1_data_collection/     # 데이터 수집
├── 2_preprocessing/       # 전처리
├── 3_district/            # 행정구역 처리
├── 4_shade_analysis/      # 그늘 분석 ⭐
└── 5_visualization/       # 시각화 ⭐
```

## 사용법

```bash
# 1. 데이터 수집
python src/1_data_collection/seoul_trees_unified.py

# 2. 구별 가로수 추출
python src/2_preprocessing/extract_district_trees.py

# 3. 그늘 연결성 분석
python src/4_shade_analysis/segment_sci_analyzer.py

# 4. 지도 시각화
python src/5_visualization/shade_sci_map_visualizer.py
```

## 분석 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| shade_factor | 0.65 | 그늘 반경 = 수관폭/2 × 0.65 |
| max_gap | 8.0m | 연결 허용 gap |
| segment_split_gap | 25.0m | 구간 분리 gap |

## 관련 프로젝트

- [서울트리맵](https://github.com/handf00t/seoul-tree-map) - 서울시 나무 인터랙티브 지도
