#!/usr/bin/env python3
"""
SCI 기반 그늘 연결성 지도 시각화
Coverage Ratio로 색상 표시
"""

import pandas as pd
import folium
from folium import plugins
import os
import json

# Coverage Ratio 기반 색상
def get_color(coverage_ratio):
    """Coverage Ratio에 따른 그라데이션 색상"""
    if coverage_ratio >= 0.95:
        return '#1a9850'  # 진한 초록 (Excellent)
    elif coverage_ratio >= 0.85:
        return '#66bd63'  # 초록 (Good)
    elif coverage_ratio >= 0.70:
        return '#a6d96a'  # 연두 (Fair)
    elif coverage_ratio >= 0.50:
        return '#fee08b'  # 노랑 (Poor)
    elif coverage_ratio >= 0.30:
        return '#fdae61'  # 주황
    else:
        return '#d73027'  # 빨강 (Very Poor)

RATING_COLORS = {
    'Excellent': '#1a9850',
    'Good': '#66bd63',
    'Fair': '#a6d96a',
    'Poor': '#fee08b',
    'Very Poor': '#d73027'
}

def create_sci_map(segments_df, trees_df=None, output_path='sci_shade_map.html', min_trees=5, min_length=30):
    """
    SCI 기반 인터랙티브 지도 생성

    Args:
        segments_df: SCI 분석 결과 DataFrame
        trees_df: 나무 DataFrame (선택사항)
        output_path: 출력 HTML 파일 경로
        min_trees: 표시할 최소 나무 수 (작은 구간 필터링)
        min_length: 표시할 최소 길이 (m)
    """
    # 최소 나무 수 및 최소 길이로 필터링
    filtered_df = segments_df[
        (segments_df['tree_count'] >= min_trees) &
        (segments_df['total_length_m'] >= min_length)
    ].copy()
    print(f"필터링: {len(segments_df)} → {len(filtered_df)}개 세그먼트 (최소 {min_trees}그루, {min_length}m)")

    # 측면별 분포 출력
    if 'side' in filtered_df.columns:
        for side in filtered_df['side'].unique():
            count = len(filtered_df[filtered_df['side'] == side])
            print(f"  - {side}측: {count}개")

    if len(filtered_df) == 0:
        print("표시할 세그먼트가 없습니다.")
        return None

    # 지도 중심점 계산
    center_lat = filtered_df['center_lat'].mean()
    center_lon = filtered_df['center_lon'].mean()

    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='cartodbpositron'
    )

    # 등급별 FeatureGroup 생성
    rating_groups = {}
    for rating in RATING_COLORS.keys():
        count = len(filtered_df[filtered_df['rating'] == rating])
        rating_groups[rating] = folium.FeatureGroup(name=f'{rating} ({count}개)')

    # 세그먼트를 선으로 표시
    for _, row in filtered_df.iterrows():
        rating = row['rating']
        color = get_color(row['coverage_ratio'])

        # 세그먼트 선 - polyline 좌표 사용 (있으면)
        if 'polyline' in row and pd.notna(row['polyline']):
            try:
                coordinates = json.loads(row['polyline'])
            except:
                coordinates = [
                    [row['min_lat'], row['min_lon']],
                    [row['max_lat'], row['max_lon']]
                ]
        else:
            coordinates = [
                [row['min_lat'], row['min_lon']],
                [row['max_lat'], row['max_lon']]
            ]

        # 측면 정보
        side_info = row.get('side', '-')

        # 팝업 내용
        popup_html = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="margin: 0 0 8px 0; color: {color};">{row['segment_name']}</h4>
            <table style="font-size: 12px; width: 100%;">
                <tr><td><b>원래 도로:</b></td><td>{row['original_road']} ({side_info}측)</td></tr>
                <tr><td><b>등급:</b></td><td style="color: {color};"><b>{rating}</b></td></tr>
                <tr style="background: #f0f0f0;"><td colspan="2"><b>📊 Coverage 분석</b></td></tr>
                <tr><td><b>Coverage Ratio:</b></td><td><b>{row['coverage_ratio']:.1%}</b></td></tr>
                <tr><td><b>총 거리:</b></td><td>{row['total_length_m']:.0f}m</td></tr>
                <tr><td><b>그늘 거리:</b></td><td style="color: green;">{row['covered_length_m']:.0f}m</td></tr>
                <tr><td><b>노출 거리:</b></td><td style="color: red;">{row['exposed_length_m']:.0f}m</td></tr>
                <tr style="background: #f0f0f0;"><td colspan="2"><b>🌳 나무 정보</b></td></tr>
                <tr><td><b>나무 수:</b></td><td>{row['tree_count']}그루</td></tr>
                <tr><td><b>평균 수관폭:</b></td><td>{row['avg_canopy']:.1f}m</td></tr>
                <tr><td><b>평균 간격:</b></td><td>{row['avg_spacing']:.1f}m</td></tr>
                <tr><td><b>Gap 수:</b></td><td>{row['n_gaps']}개</td></tr>
                <tr><td><b>최대 gap:</b></td><td>{row['max_gap_m']:.1f}m</td></tr>
            </table>
        </div>
        """

        # 선 두께: 나무 수에 비례
        weight = min(max(row['tree_count'] // 10 + 3, 3), 12)

        line = folium.PolyLine(
            coordinates,
            color=color,
            weight=weight,
            opacity=0.8,
            popup=folium.Popup(popup_html, max_width=280)
        )
        line.add_to(rating_groups[rating])

    # FeatureGroup을 지도에 추가
    for rating, group in rating_groups.items():
        group.add_to(m)

    # 나무 데이터 추가 (선택)
    if trees_df is not None and len(trees_df) > 0:
        tree_group = folium.FeatureGroup(name=f'개별 나무 ({len(trees_df)}그루)', show=False)
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

    # 레이어 컨트롤
    folium.LayerControl().add_to(m)

    # 범례
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
        <h4 style="margin: 0 0 10px 0;">Coverage Ratio (그늘 비율)</h4>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
            <div><span style="background: #1a9850; width: 30px; height: 10px; display: inline-block; margin-right: 8px;"></span>Excellent (≥95%)</div>
            <div><span style="background: #66bd63; width: 30px; height: 10px; display: inline-block; margin-right: 8px;"></span>Good (85-95%)</div>
            <div><span style="background: #a6d96a; width: 30px; height: 10px; display: inline-block; margin-right: 8px;"></span>Fair (70-85%)</div>
            <div><span style="background: #fee08b; width: 30px; height: 10px; display: inline-block; margin-right: 8px;"></span>Poor (50-70%)</div>
            <div><span style="background: #d73027; width: 30px; height: 10px; display: inline-block; margin-right: 8px;"></span>Very Poor (<50%)</div>
        </div>
        <p style="font-size: 10px; color: #666; margin-top: 8px;">
            SCI = Shade Connectivity Index<br>
            선 두께 = 나무 수
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # 타이틀
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000;
                background-color: white; padding: 10px 20px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
        <h3 style="margin: 0;">🌳 가로수 그늘 연결성 지도 (SCI 분석)</h3>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # 저장
    m.save(output_path)
    print(f"\n지도 저장: {output_path}")

    return m


def main():
    """메인 실행"""
    # 프로젝트 루트 디렉토리 (src/5_visualization/ → Archive/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, 'shade_network_output')

    segment_path = os.path.join(output_dir, 'gangnam_sci_segments.csv')
    tree_path = os.path.join(base_dir, 'seoul_trees_output', '강남구_roadside_trees_with_roads.csv')

    if not os.path.exists(segment_path):
        print(f"세그먼트 파일을 찾을 수 없습니다: {segment_path}")
        return

    segments_df = pd.read_csv(segment_path)
    print(f"세그먼트 로드: {len(segments_df)}개")

    trees_df = None
    if os.path.exists(tree_path):
        trees_df = pd.read_csv(tree_path)
        print(f"나무 로드: {len(trees_df)}그루")

    # 통계
    print("\n=== 등급별 분포 ===")
    for rating in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
        count = len(segments_df[segments_df['rating'] == rating])
        pct = count / len(segments_df) * 100
        print(f"  {rating}: {count}개 ({pct:.1f}%)")

    # 지도 생성 (최소 5그루 이상 구간만)
    map_path = os.path.join(output_dir, 'gangnam_sci_map.html')
    create_sci_map(segments_df, trees_df, map_path, min_trees=5)

    print("\n완료!")


if __name__ == '__main__':
    main()
