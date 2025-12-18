#!/usr/bin/env python3
"""
Shade Network Map Visualizer
세그먼트별 그늘 연결성을 지도 위에 시각화
"""

import pandas as pd
import folium
from folium import plugins
import os

# 등급별 색상
RATING_COLORS = {
    'Excellent': '#2ecc71',  # 초록
    'Good': '#27ae60',       # 진한 초록
    'Fair': '#f1c40f',       # 노랑
    'Poor': '#e67e22',       # 주황
    'Disconnected': '#e74c3c'  # 빨강
}

def load_segment_data(segment_csv_path):
    """세그먼트 데이터 로드"""
    df = pd.read_csv(segment_csv_path)
    print(f"세그먼트 데이터 로드: {len(df)}개")
    return df

def load_tree_data(tree_csv_path):
    """나무 데이터 로드"""
    df = pd.read_csv(tree_csv_path)
    print(f"나무 데이터 로드: {len(df)}개")
    return df

def create_shade_map(segments_df, trees_df=None, output_path='shade_map.html'):
    """
    인터랙티브 지도 생성

    Args:
        segments_df: 세그먼트 DataFrame
        trees_df: 나무 DataFrame (선택사항)
        output_path: 출력 HTML 파일 경로
    """
    # 지도 중심점 계산
    center_lat = segments_df['center_lat'].mean()
    center_lon = segments_df['center_lon'].mean()

    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='cartodbpositron'
    )

    # 등급별 FeatureGroup 생성
    rating_groups = {}
    for rating in RATING_COLORS.keys():
        rating_groups[rating] = folium.FeatureGroup(name=f'{rating} ({len(segments_df[segments_df["rating"]==rating])}개)')

    # 세그먼트를 선으로 표시
    for _, row in segments_df.iterrows():
        rating = row['rating']
        color = RATING_COLORS.get(rating, '#999999')

        # 세그먼트 선 (min_lon/lat에서 max_lon/lat까지)
        coordinates = [
            [row['min_lat'], row['min_lon']],
            [row['max_lat'], row['max_lon']]
        ]

        # 팝업 내용
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: {color};">{row['segment_name']}</h4>
            <table style="font-size: 12px; width: 100%;">
                <tr><td><b>원래 도로:</b></td><td>{row['original_road']}</td></tr>
                <tr><td><b>등급:</b></td><td style="color: {color};"><b>{rating}</b></td></tr>
                <tr><td><b>λ₂:</b></td><td>{row['lambda2']:.4f}</td></tr>
                <tr><td><b>나무 수:</b></td><td>{row['tree_count']}그루</td></tr>
                <tr><td><b>평균 수관폭:</b></td><td>{row['avg_canopy']:.1f}m</td></tr>
                <tr><td><b>평균 간격:</b></td><td>{row['avg_spacing']:.1f}m</td></tr>
                <tr><td><b>구간 길이:</b></td><td>{row['segment_length_m']:.0f}m</td></tr>
            </table>
        </div>
        """

        # 선 두께: 나무 수에 비례 (최소 3, 최대 10)
        weight = min(max(row['tree_count'] // 5 + 3, 3), 10)

        # PolyLine 추가
        line = folium.PolyLine(
            coordinates,
            color=color,
            weight=weight,
            opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250)
        )
        line.add_to(rating_groups[rating])

    # FeatureGroup을 지도에 추가
    for rating, group in rating_groups.items():
        group.add_to(m)

    # 나무 데이터가 있으면 마커 클러스터로 추가
    if trees_df is not None and len(trees_df) > 0:
        tree_group = folium.FeatureGroup(name=f'개별 나무 ({len(trees_df)}그루)', show=False)

        # 마커 클러스터 사용 (많은 포인트 처리용)
        marker_cluster = plugins.MarkerCluster()

        for _, tree in trees_df.iterrows():
            popup_text = f"""
            <b>종:</b> {tree.get('species_kr', 'N/A')}<br>
            <b>수관폭:</b> {tree.get('canopy_width_m', 'N/A')}m<br>
            <b>도로:</b> {tree.get('road_name', 'N/A')}
            """
            folium.CircleMarker(
                location=[tree['latitude'], tree['longitude']],
                radius=3,
                color='#27ae60',
                fill=True,
                fillOpacity=0.7,
                popup=popup_text
            ).add_to(marker_cluster)

        marker_cluster.add_to(tree_group)
        tree_group.add_to(m)

    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)

    # 범례 추가
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
        <h4 style="margin: 0 0 10px 0;">그늘 연결성 등급</h4>
        <div style="display: flex; flex-direction: column; gap: 5px;">
            <div><span style="background: #2ecc71; width: 20px; height: 10px; display: inline-block; margin-right: 8px;"></span>Excellent (λ₂ ≥ 0.5)</div>
            <div><span style="background: #27ae60; width: 20px; height: 10px; display: inline-block; margin-right: 8px;"></span>Good (0.2 ≤ λ₂ < 0.5)</div>
            <div><span style="background: #f1c40f; width: 20px; height: 10px; display: inline-block; margin-right: 8px;"></span>Fair (0.05 ≤ λ₂ < 0.2)</div>
            <div><span style="background: #e67e22; width: 20px; height: 10px; display: inline-block; margin-right: 8px;"></span>Poor (λ₂ < 0.05)</div>
            <div><span style="background: #e74c3c; width: 20px; height: 10px; display: inline-block; margin-right: 8px;"></span>Disconnected (λ₂ = 0)</div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # 타이틀 추가
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000;
                background-color: white; padding: 10px 20px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
        <h3 style="margin: 0;">🌳 가로수 그늘 연결성 지도</h3>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # 저장
    m.save(output_path)
    print(f"\n지도 저장 완료: {output_path}")
    print(f"브라우저에서 열어보세요!")

    return m

def create_road_summary_map(segments_df, output_path='road_summary_map.html'):
    """
    도로별 요약 지도 (세그먼트를 도로별로 그룹화)
    """
    # 도로별 통계 계산
    road_stats = segments_df.groupby('original_road').agg({
        'tree_count': 'sum',
        'lambda2': 'mean',
        'center_lat': 'mean',
        'center_lon': 'mean',
        'segment_length_m': 'sum',
        'segment_name': 'count'  # 세그먼트 수
    }).rename(columns={'segment_name': 'n_segments'})

    # 평균 lambda2로 등급 부여
    def get_rating(l2):
        if l2 >= 0.5: return 'Excellent'
        elif l2 >= 0.2: return 'Good'
        elif l2 >= 0.05: return 'Fair'
        elif l2 > 0: return 'Poor'
        else: return 'Disconnected'

    road_stats['rating'] = road_stats['lambda2'].apply(get_rating)
    road_stats = road_stats.reset_index()

    # 지도 중심점 계산
    center_lat = road_stats['center_lat'].mean()
    center_lon = road_stats['center_lon'].mean()

    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='cartodbpositron'
    )

    # 도로별 마커 추가
    for _, road in road_stats.iterrows():
        rating = road['rating']
        color = RATING_COLORS.get(rating, '#999999')

        # 마커 크기: 나무 수에 비례
        radius = min(max(road['tree_count'] // 20 + 5, 5), 25)

        popup_html = f"""
        <div style="font-family: Arial; width: 180px;">
            <h4 style="margin: 0 0 8px 0;">{road['original_road']}</h4>
            <table style="font-size: 12px;">
                <tr><td><b>등급:</b></td><td style="color: {color};"><b>{rating}</b></td></tr>
                <tr><td><b>평균 λ₂:</b></td><td>{road['lambda2']:.4f}</td></tr>
                <tr><td><b>총 나무:</b></td><td>{road['tree_count']}그루</td></tr>
                <tr><td><b>세그먼트:</b></td><td>{road['n_segments']}개</td></tr>
                <tr><td><b>총 길이:</b></td><td>{road['segment_length_m']:.0f}m</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[road['center_lat'], road['center_lon']],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)

    # 범례 추가
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
        <h4 style="margin: 0 0 10px 0;">도로별 평균 등급</h4>
        <div style="display: flex; flex-direction: column; gap: 5px;">
            <div><span style="background: #2ecc71; width: 15px; height: 15px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>Excellent</div>
            <div><span style="background: #27ae60; width: 15px; height: 15px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>Good</div>
            <div><span style="background: #f1c40f; width: 15px; height: 15px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>Fair</div>
            <div><span style="background: #e67e22; width: 15px; height: 15px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>Poor</div>
            <div><span style="background: #e74c3c; width: 15px; height: 15px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>Disconnected</div>
        </div>
        <p style="font-size: 10px; color: #666; margin-top: 8px;">원 크기 = 나무 수</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"\n도로 요약 지도 저장: {output_path}")

    return m

def main():
    """메인 실행"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'shade_network_output')

    # 강남구 세그먼트 데이터
    segment_path = os.path.join(output_dir, 'gangnam_segments.csv')
    tree_path = os.path.join(base_dir, 'seoul_trees_output', '강남구_roadside_trees_with_roads.csv')

    if not os.path.exists(segment_path):
        print(f"세그먼트 파일을 찾을 수 없습니다: {segment_path}")
        return

    # 데이터 로드
    segments_df = load_segment_data(segment_path)

    trees_df = None
    if os.path.exists(tree_path):
        trees_df = load_tree_data(tree_path)

    # 등급별 통계 출력
    print("\n=== 등급별 세그먼트 분포 ===")
    rating_counts = segments_df['rating'].value_counts()
    for rating in ['Excellent', 'Good', 'Fair', 'Poor', 'Disconnected']:
        count = rating_counts.get(rating, 0)
        pct = count / len(segments_df) * 100
        print(f"  {rating}: {count}개 ({pct:.1f}%)")

    # 세그먼트 상세 지도
    segment_map_path = os.path.join(output_dir, 'gangnam_shade_map.html')
    create_shade_map(segments_df, trees_df, segment_map_path)

    # 도로 요약 지도
    road_map_path = os.path.join(output_dir, 'gangnam_road_summary_map.html')
    create_road_summary_map(segments_df, road_map_path)

    print("\n완료!")

if __name__ == '__main__':
    main()
