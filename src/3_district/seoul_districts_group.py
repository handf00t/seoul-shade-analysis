import geopandas as gpd
import pandas as pd

def count_trees_by_district(districts_file, trees_file, output_file):
    """
    각 행정구역에 속한 나무의 개수를 계산하여 새로운 GeoJSON 파일로 저장합니다.
    """
    try:
        # 1. 두 GeoJSON 파일 불러오기
        districts_gdf = gpd.read_file('seoul_districts.geojson')
        trees_gdf = gpd.read_file('seoul_trees_output/trees_with_benefits.geojson')

        # ✅ 2. 'borough' 열을 기준으로 구역 폴리곤을 하나로 병합(Dissolve)
        # 이 과정에서 'borough' 열을 인덱스로 사용하게 됩니다.
        districts_gdf = districts_gdf.dissolve(by='borough', aggfunc='first')
        
        # 병합 후 인덱스를 다시 열로 만들어 줍니다.
        districts_gdf = districts_gdf.reset_index()

        print(f"'{districts_file}' 파일에서 {len(districts_gdf)}개의 구역을 불러왔습니다.")
        print(f"'{trees_file}' 파일에서 {len(trees_gdf)}개의 나무 데이터를 불러왔습니다.")

        # 3. CRS(좌표계) 통일
        if districts_gdf.crs != trees_gdf.crs:
            print("좌표계가 일치하지 않아, 나무 데이터를 구역 데이터의 좌표계로 변환합니다.")
            trees_gdf = trees_gdf.to_crs(districts_gdf.crs)

        # 4. 공간 결합(Spatial Join)을 사용하여 각 나무가 어느 구역에 속하는지 파악
        trees_in_districts = gpd.sjoin(trees_gdf, districts_gdf, how="inner", predicate='within')
        
        # 5. 각 구역별 나무 개수 세기
        # 'groupby' 결과로 'index_right'와 개수(size) 열이 포함된 데이터프레임 생성
        tree_counts = trees_in_districts.groupby('index_right').size().reset_index(name='tree_count')
        
        # 6. 나무 개수 데이터를 기존 구역 GeoDataFrame에 병합(merge)
        districts_with_counts = districts_gdf.reset_index().merge(
            tree_counts, 
            left_on='index', 
            right_on='index_right', 
            how='left'
        )
        
        # 구역별로 나무가 없는 경우를 위해 'tree_count' 값이 없는 행은 0으로 채웁니다.
        districts_with_counts['tree_count'] = districts_with_counts['tree_count'].fillna(0).astype(int)

        # 7. 결과 GeoDataFrame을 새로운 GeoJSON 파일로 저장
        # to_file 함수를 호출하기 전에 'index' 열을 제거합니다.
        districts_with_counts = districts_with_counts.drop(columns=['index_right', 'index'])
        districts_with_counts.to_file(output_file, driver='GeoJSON', encoding='utf-8')

        print(f"\n'{output_file}' 파일이 성공적으로 생성되었습니다!")
        print(f"각 구역에 'tree_count' 속성이 추가되었습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# 아래 경로를 사용자 파일 경로에 맞게 수정하세요.
districts_file_path = 'seoul_districts.geojson'
trees_file_path = 'seoul_trees_output/trees_with_benefits.geojson'
output_file_path = 'seoul_districts_with_tree_counts.geojson'

# 함수 실행
count_trees_by_district(districts_file_path, trees_file_path, output_file_path)