# seoul_districts_fixed.py
import os
os.environ['SHAPE_RESTORE_SHX'] = 'YES'
import geopandas as gpd

SRC = "/Users/sonchaerin/seoultree/emd.shp"
gdf = gpd.read_file(SRC, encoding="cp949")

print("원본 컬럼들:", gdf.columns.tolist())
print("샘플 EMD_CD:", gdf['EMD_CD'].head())

# 좌표계 변환
gdf = gdf.set_crs(epsg=5179, allow_override=True) if gdf.crs is None else gdf
gdf = gdf.to_crs(4326)

# 서울만 필터링 (EMD_CD가 11로 시작)
seoul = gdf[gdf['EMD_CD'].astype(str).str.startswith("11")].copy()

# EMD_CD에서 구 코드 추출 (앞 5자리)
seoul['SIG_CD'] = seoul['EMD_CD'].astype(str).str[:5]

# 구별 매핑 테이블
district_mapping = {
    "11110": "종로구", "11140": "중구", "11170": "용산구",
    "11200": "성동구", "11215": "광진구", "11230": "동대문구",
    "11260": "중랑구", "11290": "성북구", "11305": "강북구",
    "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구",
    "11500": "강서구", "11530": "구로구", "11545": "금천구",
    "11560": "영등포구", "11590": "동작구", "11620": "관악구",
    "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구"
}

# 구 이름 매핑
seoul['borough'] = seoul['SIG_CD'].map(district_mapping)

print("매핑 전 고유 SIG_CD:", seoul['SIG_CD'].nunique())
print("매핑 후 고유 구:", seoul['borough'].nunique())
print("구 목록:", seoul['borough'].unique())

# 구별로 병합 (dissolve)
sgg = seoul.dissolve(by='borough', as_index=False)
sgg = sgg[['borough', 'geometry']]

# 기하학적 오류 정리
sgg['geometry'] = sgg.buffer(0)

# 저장
OUT = "/Users/sonchaerin/seoultree/seoul_districts_final.geojson"
sgg.to_file(OUT, driver="GeoJSON")

print(f"저장 완료: {OUT}")
print(f"구 개수: {len(sgg)}")
print("구 목록:", sgg['borough'].tolist())