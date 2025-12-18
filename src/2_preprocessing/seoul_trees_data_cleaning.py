#!/usr/bin/env python3
"""
서울시 가로수 데이터 클리닝 스크립트
Seoul Street Trees Data Cleaning Script

Cleaning Actions:
1. Canopy width: 제외 (< 1m), 캡핑 (> 20m)
2. DBH: 제외 (< 5cm), 캡핑 (> 300cm)
3. Species: 빈 값 → "미상"으로 대체
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = Path("seoul_trees_output/trees_unified.geojson")
OUTPUT_PATH = Path("seoul_trees_output/trees_cleaned.geojson")
LOG_PATH = Path("data_quality_reports/cleaning_log.txt")

# Thresholds
CANOPY_WIDTH_MIN = 1.0    # 제외
CANOPY_WIDTH_MAX = 20.0   # 캡핑
DBH_MIN = 5.0             # 제외
DBH_MAX = 300.0           # 캡핑


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_geojson(filepath: Path) -> tuple:
    """GeoJSON 로드 → DataFrame 변환"""
    print(f"데이터 로드: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    records = []
    for i, feature in enumerate(geojson['features']):
        coords = feature['geometry']['coordinates']
        record = {
            'feature_idx': i,
            'longitude': coords[0],
            'latitude': coords[1],
            **feature['properties']
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"총 {len(df):,}건 로드 완료")
    return geojson, df


def clean_data(df: pd.DataFrame) -> tuple:
    """데이터 클리닝 수행"""
    log = {'original_count': len(df)}

    print("\n" + "=" * 50)
    print("데이터 클리닝 시작")
    print("=" * 50)

    # ----- 1. Canopy Width 클리닝 -----
    print("\n[1] Canopy Width 클리닝")

    # 제외: < 1m
    excluded_small_canopy = len(df[df['canopy_width_m'] < CANOPY_WIDTH_MIN])
    df = df[df['canopy_width_m'] >= CANOPY_WIDTH_MIN].copy()
    print(f"    제외 (< {CANOPY_WIDTH_MIN}m): {excluded_small_canopy:,}건")
    log['canopy_excluded'] = excluded_small_canopy

    # 캡핑: > 20m → 20m
    capped_large_canopy = (df['canopy_width_m'] > CANOPY_WIDTH_MAX).sum()
    df.loc[df['canopy_width_m'] > CANOPY_WIDTH_MAX, 'canopy_width_m'] = CANOPY_WIDTH_MAX
    print(f"    캡핑 (> {CANOPY_WIDTH_MAX}m → {CANOPY_WIDTH_MAX}m): {capped_large_canopy:,}건")
    log['canopy_capped'] = capped_large_canopy

    # ----- 2. DBH 클리닝 -----
    print("\n[2] DBH 클리닝")

    # 제외: < 5cm
    excluded_small_dbh = len(df[df['dbh_cm'] < DBH_MIN])
    df = df[df['dbh_cm'] >= DBH_MIN].copy()
    print(f"    제외 (< {DBH_MIN}cm): {excluded_small_dbh:,}건")
    log['dbh_excluded'] = excluded_small_dbh

    # 캡핑: > 300cm → 300cm
    capped_large_dbh = (df['dbh_cm'] > DBH_MAX).sum()
    df.loc[df['dbh_cm'] > DBH_MAX, 'dbh_cm'] = DBH_MAX
    print(f"    캡핑 (> {DBH_MAX}cm → {DBH_MAX}cm): {capped_large_dbh:,}건")
    log['dbh_capped'] = capped_large_dbh

    # ----- 3. Species 클리닝 -----
    print("\n[3] Species 클리닝")

    empty_species = (df['species_kr'] == '') | (df['species_kr'].isna())
    empty_count = empty_species.sum()
    df.loc[empty_species, 'species_kr'] = "미상"
    print(f"    빈 값 → '미상' 대체: {empty_count:,}건")
    log['species_imputed'] = empty_count

    # ----- 4. Size Class 재계산 -----
    print("\n[4] Size Class 재계산")

    def get_size_class(dbh):
        if pd.isna(dbh) or dbh == 0:
            return "unknown"
        elif dbh < 20:
            return "small"
        elif dbh < 50:
            return "medium"
        else:
            return "large"

    df['size_class'] = df['dbh_cm'].apply(get_size_class)

    size_dist = df['size_class'].value_counts()
    for size, count in size_dist.items():
        print(f"    {size}: {count:,}건")

    log['final_count'] = len(df)

    return df, log


def save_cleaned_geojson(df: pd.DataFrame, output_path: Path):
    """클리닝된 데이터를 GeoJSON으로 저장"""
    print("\n" + "=" * 50)
    print("GeoJSON 저장")
    print("=" * 50)

    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            },
            "properties": {
                "species_kr": row['species_kr'],
                "dbh_cm": row['dbh_cm'],
                "height_m": row['height_m'],
                "canopy_width_m": row['canopy_width_m'],
                "borough": row['borough'],
                "district": row['district'],
                "address": row['address'],
                "tree_type": row['tree_type'],
                "size_class": row['size_class'],
                "source_id": row['source_id']
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"저장 완료: {output_path} ({file_size_mb:.1f} MB)")


def save_cleaning_log(df: pd.DataFrame, log: dict, log_path: Path):
    """클리닝 로그 저장"""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("서울시 가로수 데이터 클리닝 로그\n")
        f.write("=" * 60 + "\n")
        f.write(f"작업 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 60 + "\n")
        f.write("클리닝 요약\n")
        f.write("-" * 60 + "\n")
        f.write(f"원본 데이터: {log['original_count']:,}건\n")
        f.write(f"최종 데이터: {log['final_count']:,}건\n")
        removed = log['original_count'] - log['final_count']
        f.write(f"제거된 데이터: {removed:,}건 ({removed/log['original_count']*100:.2f}%)\n\n")

        f.write("-" * 60 + "\n")
        f.write("클리닝 상세\n")
        f.write("-" * 60 + "\n")
        f.write(f"1. Canopy Width:\n")
        f.write(f"   - 제외 (< 1m): {log['canopy_excluded']:,}건\n")
        f.write(f"   - 캡핑 (> 20m): {log['canopy_capped']:,}건\n")
        f.write(f"2. DBH:\n")
        f.write(f"   - 제외 (< 5cm): {log['dbh_excluded']:,}건\n")
        f.write(f"   - 캡핑 (> 300cm): {log['dbh_capped']:,}건\n")
        f.write(f"3. Species:\n")
        f.write(f"   - '미상' 대체: {log['species_imputed']:,}건\n\n")

        f.write("-" * 60 + "\n")
        f.write("최종 데이터 통계\n")
        f.write("-" * 60 + "\n")
        f.write(f"Canopy Width: {df['canopy_width_m'].min():.1f}m ~ {df['canopy_width_m'].max():.1f}m (평균: {df['canopy_width_m'].mean():.2f}m)\n")
        f.write(f"DBH: {df['dbh_cm'].min():.1f}cm ~ {df['dbh_cm'].max():.1f}cm (평균: {df['dbh_cm'].mean():.2f}cm)\n")
        f.write(f"\n모든 데이터가 그늘 연결성 분석에 사용 가능합니다.\n")

    print(f"로그 저장: {log_path}")


def main():
    print("\n" + "*" * 50)
    print("  서울시 가로수 데이터 클리닝")
    print("*" * 50)

    # 1. 데이터 로드
    _, df = load_geojson(INPUT_PATH)

    # 2. 클리닝 수행
    df_cleaned, log = clean_data(df)

    # 3. 결과 저장
    save_cleaned_geojson(df_cleaned, OUTPUT_PATH)
    save_cleaning_log(df_cleaned, log, LOG_PATH)

    # 4. 최종 요약
    print("\n" + "=" * 50)
    print("클리닝 완료")
    print("=" * 50)
    removed = log['original_count'] - log['final_count']
    print(f"원본: {log['original_count']:,}건")
    print(f"최종: {log['final_count']:,}건")
    print(f"제거: {removed:,}건 ({removed/log['original_count']*100:.2f}%)")
    print(f"\n출력 파일: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
