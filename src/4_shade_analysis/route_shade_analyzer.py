#!/usr/bin/env python3
"""
경로 기반 그늘 분석기
- 출발지 → 도착지 경로의 그늘 비율 계산
- 실제 보행 경로를 따라 그늘 Coverage 측정
"""

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import folium
import os

class RouteShadeAnalyzer:
    """경로 기반 그늘 분석"""

    # 주요 지하철역 좌표 (강남구)
    STATIONS = {
        '강남역': (127.027621, 37.497942),
        '논현역': (127.021202, 37.510580),
        '신논현역': (127.024952, 37.504489),
        '역삼역': (127.036456, 37.500622),
        '선릉역': (127.048960, 37.504503),
        '삼성역': (127.063028, 37.508844),
        '봉은사역': (127.066219, 37.514246),
        '종합운동장역': (127.073620, 37.510997),
        '교대역': (127.014362, 37.493415),
        '서초역': (127.007702, 37.491897),
        '방배역': (126.997596, 37.481426),
        '압구정역': (127.028462, 37.527067),
        '압구정로데오역': (127.039963, 37.527456),
        '학동역': (127.031782, 37.514699),
        '언주역': (127.034380, 37.507416),
        '선정릉역': (127.044590, 37.510290),
        '대치역': (127.063720, 37.494755),
        '학여울역': (127.077940, 37.496666),
        '대청역': (127.086930, 37.492455),
        '일원역': (127.086790, 37.483380),
        '수서역': (127.101880, 37.487630),
        '양재역': (127.043270, 37.484100),
        '양재시민의숲역': (127.038870, 37.469790),
        '청담역': (127.053715, 37.519169),
        '뱅뱅사거리역': (127.047268, 37.481980),
        '도곡역': (127.046560, 37.490904),
        '매봉역': (127.034650, 37.486070),
        '개포동역': (127.074850, 37.493460),
        '구룡역': (127.077740, 37.485120),
    }

    def __init__(self,
                 shade_factor=0.65,
                 max_gap=8.0,
                 buffer_distance=15.0,  # 경로에서 나무 탐색 거리 (m)
                 lat_to_m=111000,
                 lon_to_m=88740):

        self.shade_factor = shade_factor
        self.max_gap = max_gap
        self.buffer_distance = buffer_distance
        self.lat_to_m = lat_to_m
        self.lon_to_m = lon_to_m
        self.trees_df = None
        self.tree_coords = None
        self.tree_kdtree = None

    def load_trees(self, trees_csv_path):
        """나무 데이터 로드"""
        self.trees_df = pd.read_csv(trees_csv_path)
        print(f"나무 데이터 로드: {len(self.trees_df)}그루")

        # KDTree 구축 (미터 단위)
        coords = np.column_stack([
            self.trees_df['longitude'].values * self.lon_to_m,
            self.trees_df['latitude'].values * self.lat_to_m
        ])
        self.tree_coords = coords
        self.tree_kdtree = KDTree(coords)

    def analyze_route(self, start, end, route_name=None):
        """
        경로 분석

        Args:
            start: 출발지 (역 이름 또는 (lon, lat) 튜플)
            end: 도착지 (역 이름 또는 (lon, lat) 튜플)
            route_name: 경로 이름 (선택)

        Returns:
            dict: 분석 결과
        """
        # 좌표 가져오기
        start_coord = self._get_coordinate(start)
        end_coord = self._get_coordinate(end)

        if start_coord is None or end_coord is None:
            print("좌표를 찾을 수 없습니다.")
            return None

        start_name = start if isinstance(start, str) else f"({start[0]:.4f}, {start[1]:.4f})"
        end_name = end if isinstance(end, str) else f"({end[0]:.4f}, {end[1]:.4f})"

        if route_name is None:
            route_name = f"{start_name} → {end_name}"

        print(f"\n{'='*60}")
        print(f"경로 분석: {route_name}")
        print(f"{'='*60}")

        # 경로 생성 (직선 경로를 1m 간격 점으로 분할)
        route_points = self._create_route_points(start_coord, end_coord)
        route_length = len(route_points) - 1  # 미터

        print(f"출발: {start_name} ({start_coord[0]:.6f}, {start_coord[1]:.6f})")
        print(f"도착: {end_name} ({end_coord[0]:.6f}, {end_coord[1]:.6f})")
        print(f"경로 길이: {route_length}m")

        # 경로 주변 나무 찾기
        route_trees = self._find_trees_along_route(route_points)
        print(f"경로 주변 나무: {len(route_trees)}그루 (버퍼 {self.buffer_distance}m)")

        if len(route_trees) < 2:
            print("경로 주변에 나무가 부족합니다.")
            return {
                'route_name': route_name,
                'start': start_name,
                'end': end_name,
                'route_length_m': route_length,
                'tree_count': len(route_trees),
                'coverage_ratio': 0,
                'covered_length_m': 0,
                'exposed_length_m': route_length,
                'message': '나무 부족'
            }

        # 그늘 Coverage 계산
        result = self._calculate_route_coverage(route_points, route_trees)

        result['route_name'] = route_name
        result['start'] = start_name
        result['end'] = end_name
        result['route_length_m'] = route_length

        self._print_result(result)

        return result

    def _get_coordinate(self, location):
        """위치를 좌표로 변환"""
        if isinstance(location, str):
            # 역 이름으로 좌표 찾기
            if location in self.STATIONS:
                return self.STATIONS[location]
            else:
                print(f"알 수 없는 역: {location}")
                print(f"사용 가능한 역: {', '.join(self.STATIONS.keys())}")
                return None
        elif isinstance(location, (tuple, list)) and len(location) == 2:
            return tuple(location)
        else:
            return None

    def _create_route_points(self, start_coord, end_coord, interval=1.0):
        """경로를 일정 간격의 점으로 분할 (미터 단위)"""
        start_m = np.array([start_coord[0] * self.lon_to_m, start_coord[1] * self.lat_to_m])
        end_m = np.array([end_coord[0] * self.lon_to_m, end_coord[1] * self.lat_to_m])

        distance = np.linalg.norm(end_m - start_m)
        n_points = int(distance / interval) + 1

        points = []
        for i in range(n_points):
            t = i / max(n_points - 1, 1)
            point = start_m + t * (end_m - start_m)
            points.append(point)

        return np.array(points)

    def _find_trees_along_route(self, route_points):
        """경로 주변 나무 찾기"""
        # 모든 경로 점에서 buffer_distance 내의 나무 인덱스 수집
        all_tree_indices = set()

        for point in route_points[::10]:  # 10m 간격으로 샘플링 (속도 향상)
            indices = self.tree_kdtree.query_ball_point(point, self.buffer_distance)
            all_tree_indices.update(indices)

        if not all_tree_indices:
            return pd.DataFrame()

        return self.trees_df.iloc[list(all_tree_indices)].copy()

    def _calculate_route_coverage(self, route_points, route_trees):
        """경로의 그늘 Coverage 계산 - 구간별 분리"""
        # 나무 좌표와 그늘 영향 범위
        tree_coords = np.column_stack([
            route_trees['longitude'].values * self.lon_to_m,
            route_trees['latitude'].values * self.lat_to_m
        ])
        # 보행자 기준: 나무 중심에서 (수관폭/2 + 3m) 이내면 그늘 효과
        effective_shade_radii = (route_trees['canopy_width_m'].values / 2) + 3.0

        # 각 경로 점의 그늘 여부 계산
        shade_status = []  # True=그늘, False=노출
        for point in route_points:
            distances = np.linalg.norm(tree_coords - point, axis=1)
            in_shade = np.any(distances <= effective_shade_radii)
            shade_status.append(in_shade)

        # 구간 분리 (그늘 있는 구간 vs 없는 구간)
        segments = []
        current_type = shade_status[0]
        current_start = 0

        for i, status in enumerate(shade_status):
            if status != current_type:
                # 구간 종료
                segments.append({
                    'type': '그늘' if current_type else '노출',
                    'start_m': current_start,
                    'end_m': i,
                    'length_m': i - current_start
                })
                current_type = status
                current_start = i

        # 마지막 구간
        segments.append({
            'type': '그늘' if current_type else '노출',
            'start_m': current_start,
            'end_m': len(shade_status),
            'length_m': len(shade_status) - current_start
        })

        # 짧은 구간 병합 (10m 미만)
        merged_segments = self._merge_short_segments(segments, min_length=10)

        # 통계 계산
        covered_length = sum(s['length_m'] for s in merged_segments if s['type'] == '그늘')
        exposed_length = sum(s['length_m'] for s in merged_segments if s['type'] == '노출')
        route_length = len(route_points)

        return {
            'tree_count': len(route_trees),
            'coverage_ratio': round(covered_length / route_length, 4) if route_length > 0 else 0,
            'covered_length_m': covered_length,
            'exposed_length_m': exposed_length,
            'segments': merged_segments,
            'n_shade_segments': sum(1 for s in merged_segments if s['type'] == '그늘'),
            'n_exposed_segments': sum(1 for s in merged_segments if s['type'] == '노출'),
            'avg_canopy': round(route_trees['canopy_width_m'].mean(), 1),
        }

    def _merge_short_segments(self, segments, min_length=10):
        """짧은 구간을 인접 구간에 병합"""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            if seg['length_m'] < min_length:
                # 이전 구간에 병합
                merged[-1]['end_m'] = seg['end_m']
                merged[-1]['length_m'] = merged[-1]['end_m'] - merged[-1]['start_m']
            elif merged[-1]['length_m'] < min_length:
                # 이전 구간이 짧으면 현재 구간 타입으로 변경
                merged[-1]['type'] = seg['type']
                merged[-1]['end_m'] = seg['end_m']
                merged[-1]['length_m'] = merged[-1]['end_m'] - merged[-1]['start_m']
            elif merged[-1]['type'] == seg['type']:
                # 같은 타입이면 병합
                merged[-1]['end_m'] = seg['end_m']
                merged[-1]['length_m'] = merged[-1]['end_m'] - merged[-1]['start_m']
            else:
                merged.append(seg)

        return merged

    def _print_result(self, result):
        """결과 출력"""
        print(f"\n=== 분석 결과 ===")
        print(f"경로 길이: {result['route_length_m']}m")
        print(f"주변 나무: {result['tree_count']}그루")
        print(f"")
        print(f"📊 Coverage Ratio: {result['coverage_ratio']:.1%}")
        print(f"   - 그늘 거리: {result['covered_length_m']}m")
        print(f"   - 노출 거리: {result['exposed_length_m']}m")

        # 구간별 상세
        if 'segments' in result and result['segments']:
            print(f"\n=== 구간별 상세 ===")
            for i, seg in enumerate(result['segments']):
                icon = "🌳" if seg['type'] == '그늘' else "☀️"
                print(f"  {i+1}. {icon} {seg['type']}: {seg['start_m']}m ~ {seg['end_m']}m ({seg['length_m']}m)")

            print(f"\n  그늘 구간: {result['n_shade_segments']}개")
            print(f"  노출 구간: {result['n_exposed_segments']}개")

        # 등급 판정
        cr = result['coverage_ratio']
        if cr >= 0.8:
            grade = "🌳 우수 (Excellent)"
        elif cr >= 0.6:
            grade = "🌲 양호 (Good)"
        elif cr >= 0.4:
            grade = "🌿 보통 (Fair)"
        else:
            grade = "☀️ 미흡 (Poor)"

        print(f"\n종합 등급: {grade}")

    def analyze_route_with_waypoints(self, waypoints, route_name=None):
        """
        경유지를 포함한 경로 분석

        Args:
            waypoints: 경유지 리스트 [출발, 경유1, 경유2, ..., 도착]
            route_name: 경로 이름
        """
        if len(waypoints) < 2:
            print("최소 2개 지점이 필요합니다.")
            return None

        if route_name is None:
            route_name = ' → '.join(str(w) for w in waypoints)

        print(f"\n{'='*60}")
        print(f"경로 분석: {route_name}")
        print(f"{'='*60}")

        # 각 구간별 경로 점 생성
        all_route_points = []
        total_length = 0

        for i in range(len(waypoints) - 1):
            start_coord = self._get_coordinate(waypoints[i])
            end_coord = self._get_coordinate(waypoints[i + 1])

            if start_coord is None or end_coord is None:
                return None

            segment_points = self._create_route_points(start_coord, end_coord)
            if i > 0:
                segment_points = segment_points[1:]  # 중복 제거

            all_route_points.extend(segment_points)

        all_route_points = np.array(all_route_points)
        route_length = len(all_route_points) - 1

        print(f"총 경유지: {len(waypoints)}개")
        print(f"경로 길이: {route_length}m")

        # 경로 주변 나무 찾기
        route_trees = self._find_trees_along_route(all_route_points)
        print(f"경로 주변 나무: {len(route_trees)}그루 (버퍼 {self.buffer_distance}m)")

        if len(route_trees) < 2:
            print("경로 주변에 나무가 부족합니다.")
            return None

        # 그늘 Coverage 계산
        result = self._calculate_route_coverage(all_route_points, route_trees)
        result['route_name'] = route_name
        result['route_length_m'] = route_length
        result['waypoints'] = waypoints

        self._print_result(result)

        return result

    def visualize_route(self, result, route_points, output_path='route_shade_map.html'):
        """
        경로 분석 결과를 지도에 시각화

        Args:
            result: analyze_route 결과
            route_points: 경로 점들 (미터 단위)
            output_path: 출력 파일 경로
        """
        if result is None or 'segments' not in result:
            print("시각화할 결과가 없습니다.")
            return

        # 경로 점을 위경도로 변환
        route_latlons = [
            (p[1] / self.lat_to_m, p[0] / self.lon_to_m)
            for p in route_points
        ]

        # 지도 중심
        center_lat = np.mean([p[0] for p in route_latlons])
        center_lon = np.mean([p[1] for p in route_latlons])

        # 지도 생성
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='cartodbpositron'
        )

        # 구간별로 다른 색상으로 표시
        for seg in result['segments']:
            start_idx = seg['start_m']
            end_idx = min(seg['end_m'], len(route_latlons) - 1)

            segment_coords = route_latlons[start_idx:end_idx + 1]

            if len(segment_coords) < 2:
                continue

            color = '#2ecc71' if seg['type'] == '그늘' else '#e74c3c'
            icon = '🌳' if seg['type'] == '그늘' else '☀️'

            # 구간 선 그리기
            folium.PolyLine(
                locations=segment_coords,
                color=color,
                weight=8,
                opacity=0.8,
                popup=f"{icon} {seg['type']}: {seg['length_m']}m"
            ).add_to(m)

        # 출발점 마커
        folium.Marker(
            location=route_latlons[0],
            popup=f"출발: {result.get('start', result.get('waypoints', [''])[0])}",
            icon=folium.Icon(color='blue', icon='play')
        ).add_to(m)

        # 도착점 마커
        folium.Marker(
            location=route_latlons[-1],
            popup=f"도착: {result.get('end', result.get('waypoints', [''])[-1])}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

        # 범례
        legend_html = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 15px; border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
            <h4 style="margin: 0 0 10px 0;">{result['route_name']}</h4>
            <div style="margin-bottom: 8px;">
                <span style="background: #2ecc71; width: 30px; height: 8px; display: inline-block; margin-right: 8px;"></span>
                그늘 ({result['covered_length_m']}m, {result['coverage_ratio']:.1%})
            </div>
            <div style="margin-bottom: 8px;">
                <span style="background: #e74c3c; width: 30px; height: 8px; display: inline-block; margin-right: 8px;"></span>
                노출 ({result['exposed_length_m']}m)
            </div>
            <hr style="margin: 8px 0;">
            <div style="font-size: 12px;">
                총 거리: {result['route_length_m']}m<br>
                나무: {result['tree_count']}그루
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # 저장
        m.save(output_path)
        print(f"\n지도 저장: {output_path}")

        return m

    def analyze_and_visualize(self, start, end, output_path=None):
        """분석 후 시각화까지 한 번에"""
        start_coord = self._get_coordinate(start)
        end_coord = self._get_coordinate(end)

        if start_coord is None or end_coord is None:
            return None

        route_points = self._create_route_points(start_coord, end_coord)
        result = self.analyze_route(start, end)

        if result and output_path is None:
            start_name = start if isinstance(start, str) else 'start'
            end_name = end if isinstance(end, str) else 'end'
            output_path = f"route_{start_name}_{end_name}.html"

        if result:
            self.visualize_route(result, route_points, output_path)

        return result

    def analyze_and_visualize_waypoints(self, waypoints, route_name=None, output_path=None):
        """경유지 포함 분석 후 시각화"""
        # 경로 점 생성
        all_route_points = []
        for i in range(len(waypoints) - 1):
            start_coord = self._get_coordinate(waypoints[i])
            end_coord = self._get_coordinate(waypoints[i + 1])
            if start_coord is None or end_coord is None:
                return None
            segment_points = self._create_route_points(start_coord, end_coord)
            if i > 0:
                segment_points = segment_points[1:]
            all_route_points.extend(segment_points)

        all_route_points = np.array(all_route_points)

        result = self.analyze_route_with_waypoints(waypoints, route_name)

        if result and output_path is None:
            output_path = f"route_{'_'.join(str(w) for w in waypoints)}.html"

        if result:
            self.visualize_route(result, all_route_points, output_path)

        return result

    def list_stations(self):
        """사용 가능한 역 목록 출력"""
        print("\n=== 사용 가능한 지하철역 ===")
        for name, coord in sorted(self.STATIONS.items()):
            print(f"  {name}: ({coord[0]:.6f}, {coord[1]:.6f})")


def main():
    """메인 실행"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(base_dir))
    output_dir = os.path.join(project_dir, 'shade_network_output')

    # 강남구 나무 데이터
    trees_path = os.path.join(project_dir, 'seoul_trees_output', '강남구_roadside_trees_with_roads.csv')

    if not os.path.exists(trees_path):
        print(f"파일을 찾을 수 없습니다: {trees_path}")
        return

    # 분석기 초기화
    analyzer = RouteShadeAnalyzer(
        shade_factor=0.65,
        max_gap=8.0,
        buffer_distance=20.0  # 경로에서 20m 내 나무 탐색
    )

    analyzer.load_trees(trees_path)

    # 강남역 → 논현역 (신논현역 경유 - 강남대로 따라가기) + 시각화
    output_path = os.path.join(output_dir, 'route_gangnam_nonhyeon.html')
    analyzer.analyze_and_visualize_waypoints(
        ['강남역', '신논현역', '논현역'],
        '강남역 → 논현역 (강남대로)',
        output_path
    )

    print("\n" + "="*60)

    # 추가 경로 분석
    analyzer.analyze_route('강남역', '역삼역')
    analyzer.analyze_route('선릉역', '삼성역')


if __name__ == '__main__':
    main()
