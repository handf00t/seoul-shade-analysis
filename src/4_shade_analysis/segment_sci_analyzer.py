#!/usr/bin/env python3
"""
Segment SCI Analyzer v2
- 도로 양쪽 분리 (측면 거리 기반)
- 큰 gap (교차로 수준)에서만 구간 분리
- SCI (Shade Connectivity Index) 방식으로 연결성 평가
"""

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from collections import defaultdict
import os

class SegmentSCIAnalyzer:
    """
    SCI (Shade Connectivity Index) 기반 세그먼트 분석기

    v2: 도로 양쪽 분리 기능 추가
    - 도로 방향 계산 (PCA)
    - 측면 거리가 max_lateral_distance 초과하면 반대편으로 분리
    """

    def __init__(self,
                 shade_factor=0.65,           # 그늘 반경 계수
                 max_gap=8.0,                 # 그늘 연결 최대 허용 gap (8m = 10걸음)
                 segment_split_gap=25.0,      # 구간 분리 gap (25m = 교차로/횡단보도)
                 max_lateral_distance=6.0,    # 최대 측면 거리 (보도 폭, 6m)
                 lat_to_m=111000,             # 위도 → 미터 변환
                 lon_to_m=88740):             # 경도 → 미터 변환 (서울 위도 기준)

        self.shade_factor = shade_factor
        self.max_gap = max_gap
        self.segment_split_gap = segment_split_gap
        self.max_lateral_distance = max_lateral_distance
        self.lat_to_m = lat_to_m
        self.lon_to_m = lon_to_m

    def analyze(self, trees_df):
        """
        전체 분석 실행

        Args:
            trees_df: DataFrame with columns [longitude, latitude, canopy_width_m, road_name]

        Returns:
            segments_df: 세그먼트별 분석 결과
        """
        print("=" * 60)
        print("Shade Connectivity Index (SCI) 세그먼트 분석 v2")
        print("=" * 60)
        print(f"설정:")
        print(f"  - 그늘 계수: {self.shade_factor}")
        print(f"  - 연결 허용 gap: {self.max_gap}m")
        print(f"  - 구간 분리 gap: {self.segment_split_gap}m")
        print(f"  - 최대 측면 거리: {self.max_lateral_distance}m (도로 폭)")
        print()

        # 도로별로 그룹화
        road_groups = trees_df.groupby('road_name')
        print(f"총 {len(road_groups)}개 도로 분석 시작...")

        all_segments = []

        for road_name, road_trees in road_groups:
            if len(road_trees) < 2:
                continue

            segments = self._analyze_road(road_name, road_trees)
            all_segments.extend(segments)

        # DataFrame 생성
        segments_df = pd.DataFrame(all_segments)

        if len(segments_df) == 0:
            print("분석된 세그먼트가 없습니다.")
            return segments_df

        # 인접 세그먼트 병합 (200m 이상 되도록)
        before_count = len(segments_df)
        before_trees = segments_df['tree_count'].sum()
        segments_df = self._merge_adjacent_segments(segments_df, min_length=200)
        print(f"인접 세그먼트 병합: {before_count}개 → {len(segments_df)}개")
        print(f"나무 포함: {segments_df['tree_count'].sum()}그루 (병합 전 {before_trees}그루)")

        # 등급 부여 (Coverage Ratio 기반)
        segments_df['rating'] = segments_df['coverage_ratio'].apply(self._get_rating)

        # 순위 매기기 (Coverage Ratio 높은 순)
        segments_df = segments_df.sort_values('coverage_ratio', ascending=False)
        segments_df['rank'] = range(1, len(segments_df) + 1)

        self._print_summary(segments_df)

        return segments_df

    def _analyze_road(self, road_name, road_trees):
        """도로 분석 - 양쪽 분리 후, 큰 gap에서 구간 분리, SCI 계산"""
        # 좌표를 미터로 변환
        coords = np.column_stack([
            road_trees['longitude'].values * self.lon_to_m,
            road_trees['latitude'].values * self.lat_to_m
        ])

        canopies = road_trees['canopy_width_m'].values
        lons = road_trees['longitude'].values
        lats = road_trees['latitude'].values

        # 도로 양쪽으로 분리
        sides = self._split_by_road_side(coords, canopies, lons, lats, road_name)

        all_segments = []

        for side_idx, side_data in enumerate(sides):
            if len(side_data['coords']) < 2:
                continue

            side_suffix = 'A' if side_idx == 0 else 'B'

            # 각 측면에서 구간 분리 및 CGP 계산
            segments = self._analyze_road_side(
                road_name, side_suffix, side_data
            )
            all_segments.extend(segments)

        return all_segments

    def _split_by_road_side(self, coords, canopies, lons, lats, road_name):
        """
        도로 양쪽으로 나무 분리

        1. PCA로 도로 주 방향 계산
        2. 주 방향에 수직인 축으로 투영
        3. 측면 거리 기준으로 클러스터링
        """
        n = len(coords)

        if n < 3:
            # 나무가 적으면 분리하지 않음
            return [{
                'coords': coords,
                'canopies': canopies,
                'lons': lons,
                'lats': lats
            }]

        # PCA로 도로 주 방향 계산
        pca = PCA(n_components=2)
        pca.fit(coords)

        # 주 방향 (첫 번째 주성분)
        main_direction = pca.components_[0]
        # 수직 방향 (두 번째 주성분)
        perpendicular = pca.components_[1]

        # 중심점
        center = pca.mean_

        # 각 나무의 수직 방향 거리 계산 (측면 거리)
        lateral_distances = np.dot(coords - center, perpendicular)

        # 측면 거리로 양쪽 분리
        # 중앙값 기준으로 분리 (양쪽에 나무가 있는 경우)
        median_lateral = np.median(lateral_distances)

        # 측면 거리의 범위 확인
        lateral_range = np.max(lateral_distances) - np.min(lateral_distances)

        if lateral_range < self.max_lateral_distance:
            # 도로 폭보다 좁으면 분리하지 않음 (한쪽에만 나무)
            return [{
                'coords': coords,
                'canopies': canopies,
                'lons': lons,
                'lats': lats
            }]

        # 양쪽으로 분리
        side_a_mask = lateral_distances <= median_lateral
        side_b_mask = lateral_distances > median_lateral

        sides = []

        if np.sum(side_a_mask) >= 2:
            sides.append({
                'coords': coords[side_a_mask],
                'canopies': canopies[side_a_mask],
                'lons': lons[side_a_mask],
                'lats': lats[side_a_mask]
            })

        if np.sum(side_b_mask) >= 2:
            sides.append({
                'coords': coords[side_b_mask],
                'canopies': canopies[side_b_mask],
                'lons': lons[side_b_mask],
                'lats': lats[side_b_mask]
            })

        if len(sides) == 0:
            # 분리 실패시 원본 반환
            return [{
                'coords': coords,
                'canopies': canopies,
                'lons': lons,
                'lats': lats
            }]

        return sides

    def _analyze_road_side(self, road_name, side_suffix, side_data):
        """한쪽 측면 분석"""
        coords = side_data['coords']
        canopies = side_data['canopies']
        lons = side_data['lons']
        lats = side_data['lats']

        shade_radii = (canopies / 2) * self.shade_factor

        # x좌표로 정렬
        sort_idx = np.argsort(coords[:, 0])
        coords = coords[sort_idx]
        shade_radii = shade_radii[sort_idx]
        canopies = canopies[sort_idx]
        lons = lons[sort_idx]
        lats = lats[sort_idx]

        # 큰 gap에서 구간 분리
        segments = self._split_at_large_gaps(
            coords, shade_radii, canopies, lons, lats,
            road_name, side_suffix
        )

        return segments

    def _nearest_neighbor_chain(self, coords):
        """최근접 이웃 체이닝으로 나무들을 도로 방향으로 정렬"""
        n = len(coords)
        if n <= 2:
            return np.arange(n)

        visited = np.zeros(n, dtype=bool)
        chain = []

        # 가장 왼쪽 점에서 시작
        current = np.argmin(coords[:, 0])
        chain.append(current)
        visited[current] = True

        kdtree = KDTree(coords)

        for _ in range(n - 1):
            # 현재 점에서 가장 가까운 미방문 점 찾기
            distances, indices = kdtree.query(coords[current], k=min(20, n))

            next_idx = None
            for idx in indices:
                if not visited[idx]:
                    next_idx = idx
                    break

            if next_idx is None:
                break

            chain.append(next_idx)
            visited[next_idx] = True
            current = next_idx

        return np.array(chain)

    def _split_at_large_gaps(self, coords, shade_radii, canopies, lons, lats,
                             road_name, side_suffix):
        """segment_split_gap 이상이거나 도로를 가로지르면 구간 분리"""
        n = len(coords)
        if n < 2:
            return []

        # 도로 방향 계산 (PCA)
        from sklearn.decomposition import PCA
        if n >= 3:
            pca = PCA(n_components=2)
            pca.fit(coords)
            main_direction = pca.components_[0]   # 도로 방향
            perpendicular = pca.components_[1]    # 도로 수직 방향

            # 도로 방향으로 정렬 (x좌표 정렬 대신)
            projections = np.dot(coords - coords.mean(axis=0), main_direction)
            sort_idx = np.argsort(projections)
            coords = coords[sort_idx]
            shade_radii = shade_radii[sort_idx]
            canopies = canopies[sort_idx]
            lons = lons[sort_idx]
            lats = lats[sort_idx]
        else:
            # 나무가 적으면 두 점 사이 방향 사용
            direction = coords[1] - coords[0]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            perpendicular = np.array([-direction[1], direction[0]])

        # 연속된 나무들 사이의 거리, gap, 측면거리 계산
        gaps = []
        lateral_distances = []

        for i in range(n - 1):
            diff = coords[i+1] - coords[i]
            dist = np.linalg.norm(diff)
            effective_coverage = shade_radii[i] + shade_radii[i+1]
            gap = max(0, dist - effective_coverage)

            # 측면 거리 (도로 수직 방향 성분)
            lateral = abs(np.dot(diff, perpendicular))

            gaps.append(gap)
            lateral_distances.append(lateral)

        # 분리 지점 찾기
        # 1) gap이 segment_split_gap 초과
        # 2) 측면 거리가 max_lateral_distance 초과 (도로 횡단)
        split_points = [0]
        for i in range(len(gaps)):
            if gaps[i] > self.segment_split_gap:
                split_points.append(i + 1)
            elif lateral_distances[i] > self.max_lateral_distance:
                split_points.append(i + 1)
        split_points.append(n)

        # 중복 제거 및 정렬
        split_points = sorted(set(split_points))

        # 각 구간 분석
        segments = []
        for seg_idx in range(len(split_points) - 1):
            start = split_points[seg_idx]
            end = split_points[seg_idx + 1]

            if end - start < 2:  # 최소 2그루 필요
                continue

            seg_coords = coords[start:end]
            seg_shade_radii = shade_radii[start:end]
            seg_canopies = canopies[start:end]
            seg_lons = lons[start:end]
            seg_lats = lats[start:end]

            # SCI 계산
            sci_result = self._calculate_sci(
                seg_coords, seg_shade_radii, seg_canopies
            )

            # 세그먼트 이름: 도로명_측면_번호
            segment_name = f"{road_name}_{side_suffix}_{seg_idx + 1}"

            # 도로 방향으로 정렬된 좌표 (시각화용)
            # PCA로 주 방향 찾기
            if len(seg_coords) >= 2:
                from sklearn.decomposition import PCA
                pca_seg = PCA(n_components=1)
                pca_seg.fit(seg_coords)
                main_dir = pca_seg.components_[0]

                # 주 방향으로 투영하여 정렬
                projections = np.dot(seg_coords - seg_coords.mean(axis=0), main_dir)
                sort_by_dir = np.argsort(projections)

                # 정렬된 좌표로 polyline 생성 (JSON 형식)
                sorted_lons = seg_lons[sort_by_dir]
                sorted_lats = seg_lats[sort_by_dir]
                polyline_coords = [[float(lat), float(lon)] for lat, lon in zip(sorted_lats, sorted_lons)]
            else:
                polyline_coords = [[float(seg_lats[0]), float(seg_lons[0])]]

            import json
            segment_info = {
                'segment_name': segment_name,
                'original_road': road_name,
                'side': side_suffix,
                'tree_count': len(seg_coords),
                'total_length_m': sci_result['total_length'],
                'covered_length_m': sci_result['covered_length'],
                'exposed_length_m': sci_result['exposed_length'],
                'coverage_ratio': sci_result['coverage_ratio'],
                'sci': sci_result['sci'],  # Shade Connectivity Index
                'max_gap_m': sci_result['max_gap'],
                'avg_gap_m': sci_result['avg_gap'],
                'n_gaps': sci_result['n_gaps'],
                'avg_canopy': np.mean(seg_canopies),
                'avg_spacing': sci_result['avg_spacing'],
                'center_lon': np.mean(seg_lons),
                'center_lat': np.mean(seg_lats),
                'min_lon': np.min(seg_lons),
                'max_lon': np.max(seg_lons),
                'min_lat': np.min(seg_lats),
                'max_lat': np.max(seg_lats),
                'polyline': json.dumps(polyline_coords),  # 시각화용 경로
            }

            segments.append(segment_info)

        return segments

    def _calculate_sci(self, coords, shade_radii, canopies):
        """
        Shade Connectivity Index 계산

        그늘 연결 상태를 걸으면서 평가:
        - 두 나무의 그늘이 겹치면: 연속 그늘 (covered)
        - gap이 max_gap 이내면: 허용 (covered로 간주)
        - gap이 max_gap 초과면: 노출 구간
        """
        n = len(coords)

        total_length = 0
        covered_length = 0
        exposed_length = 0
        gaps = []
        spacings = []

        for i in range(n - 1):
            dist = np.linalg.norm(coords[i+1] - coords[i])
            spacings.append(dist)

            effective_coverage = shade_radii[i] + shade_radii[i+1]
            gap = max(0, dist - effective_coverage)

            total_length += dist

            if gap <= self.max_gap:
                # 연결됨 (gap이 허용 범위 내)
                covered_length += dist
            else:
                # 끊어짐 - 그늘 부분만 covered, 나머지는 exposed
                covered_length += effective_coverage
                exposed_length += gap
                gaps.append(gap)

        # 첫 번째와 마지막 나무의 그늘 반경도 추가
        covered_length += shade_radii[0] + shade_radii[-1]
        total_length += shade_radii[0] + shade_radii[-1]

        # Coverage Ratio (0~1)
        coverage_ratio = covered_length / total_length if total_length > 0 else 0
        coverage_ratio = min(coverage_ratio, 1.0)  # 1 초과 방지

        return {
            'total_length': round(total_length, 1),
            'covered_length': round(covered_length, 1),
            'exposed_length': round(exposed_length, 1),
            'coverage_ratio': round(coverage_ratio, 4),
            'sci': round(coverage_ratio, 4),  # SCI = Coverage Ratio (0~1)
            'max_gap': round(max(gaps) if gaps else 0, 1),
            'avg_gap': round(np.mean(gaps) if gaps else 0, 1),
            'n_gaps': len(gaps),
            'avg_spacing': round(np.mean(spacings) if spacings else 0, 1),
        }

    def _merge_adjacent_segments(self, segments_df, min_length=200, max_lateral_gap=6, max_longitudinal_gap=50):
        """
        인접 세그먼트 병합

        같은 도로, 같은 측면의 세그먼트 중 실제로 인접한 것만 병합하여
        min_length 이상이 되도록 함

        Args:
            min_length: 최소 세그먼트 길이 (m)
            max_lateral_gap: 횡단 방향 최대 허용 거리 (m) - 이 이상이면 병합 차단
            max_longitudinal_gap: 도로 방향 최대 허용 거리 (m) - 이 이상이면 병합 차단
        """
        import json
        from sklearn.decomposition import PCA

        merged_results = []

        for (road, side), group in segments_df.groupby(['original_road', 'side']):
            if len(group) < 1:
                continue

            # 도로 방향 계산 (PCA)
            coords = np.column_stack([
                group['center_lon'].values * self.lon_to_m,
                group['center_lat'].values * self.lat_to_m
            ])

            if len(coords) >= 2:
                pca = PCA(n_components=2)
                pca.fit(coords)
                main_direction = pca.components_[0]  # 도로 방향
                perpendicular = pca.components_[1]   # 횡단 방향
            else:
                main_direction = np.array([1, 0])
                perpendicular = np.array([0, 1])

            # 위치 순 정렬 (도로 방향 투영 기준)
            projections = np.dot(coords - coords.mean(axis=0), main_direction)
            sort_idx = np.argsort(projections)
            group = group.iloc[sort_idx].reset_index(drop=True)

            # 병합 버퍼
            buffer = {
                'segments': [],
                'tree_count': 0,
                'total_length': 0,
                'covered_length': 0,
                'exposed_length': 0,
                'n_gaps': 0,
                'max_gap': 0,
                'canopies': [],
                'spacings': [],
                'lons': [],
                'lats': [],
                'polylines': []
            }

            def flush_buffer(buf, road, side, idx, lat_to_m, lon_to_m):
                """버퍼의 세그먼트들을 병합하여 결과에 추가"""
                if buf['tree_count'] == 0:
                    return None

                # polyline 병합
                all_coords = []
                for pl in buf['polylines']:
                    try:
                        coords = json.loads(pl)
                        all_coords.extend(coords)
                    except:
                        pass

                # polyline을 도로 방향으로 정렬 (지그재그 제거)
                if len(all_coords) >= 2:
                    coords_m = np.array([[lon * lon_to_m, lat * lat_to_m] for lat, lon in all_coords])
                    # PCA로 주 방향 계산
                    from sklearn.decomposition import PCA
                    pca_line = PCA(n_components=1)
                    pca_line.fit(coords_m)
                    main_dir = pca_line.components_[0]
                    # 주 방향으로 투영하여 정렬
                    projections = np.dot(coords_m - coords_m.mean(axis=0), main_dir)
                    sort_idx = np.argsort(projections)
                    all_coords = [all_coords[i] for i in sort_idx]

                # polyline 기반 실제 거리 계산
                if len(all_coords) >= 2:
                    total_len = 0
                    for i in range(len(all_coords) - 1):
                        lat1, lon1 = all_coords[i]
                        lat2, lon2 = all_coords[i + 1]
                        dist = np.sqrt(((lat2 - lat1) * lat_to_m) ** 2 +
                                      ((lon2 - lon1) * lon_to_m) ** 2)
                        total_len += dist
                else:
                    total_len = buf['total_length']

                covered_len = buf['covered_length']
                exposed_len = total_len - covered_len  # 실제 거리 기반으로 재계산
                exposed_len = max(0, exposed_len)

                coverage = covered_len / total_len if total_len > 0 else 0
                coverage = min(coverage, 1.0)

                return {
                    'segment_name': f"{road}_{side}_{idx}",
                    'original_road': road,
                    'side': side,
                    'tree_count': buf['tree_count'],
                    'total_length_m': round(total_len, 1),
                    'covered_length_m': round(covered_len, 1),
                    'exposed_length_m': round(exposed_len, 1),
                    'coverage_ratio': round(coverage, 4),
                    'sci': round(coverage, 4),
                    'max_gap_m': buf['max_gap'],
                    'avg_gap_m': round(exposed_len / buf['n_gaps'], 1) if buf['n_gaps'] > 0 else 0,
                    'n_gaps': buf['n_gaps'],
                    'avg_canopy': round(np.mean(buf['canopies']), 1) if buf['canopies'] else 0,
                    'avg_spacing': round(np.mean(buf['spacings']), 1) if buf['spacings'] else 0,
                    'center_lon': np.mean(buf['lons']) if buf['lons'] else 0,
                    'center_lat': np.mean(buf['lats']) if buf['lats'] else 0,
                    'min_lon': min(buf['lons']) if buf['lons'] else 0,
                    'max_lon': max(buf['lons']) if buf['lons'] else 0,
                    'min_lat': min(buf['lats']) if buf['lats'] else 0,
                    'max_lat': max(buf['lats']) if buf['lats'] else 0,
                    'polyline': json.dumps(all_coords),
                    'merged_count': len(buf['segments'])
                }

            merged_idx = 1
            prev_seg = None

            for _, seg in group.iterrows():
                # 이전 세그먼트와의 거리 체크 (횡단 + 도로 방향)
                if prev_seg is not None and buffer['tree_count'] > 0:
                    # 세그먼트 간 벡터
                    diff = np.array([
                        (seg['center_lon'] - prev_seg['center_lon']) * self.lon_to_m,
                        (seg['center_lat'] - prev_seg['center_lat']) * self.lat_to_m
                    ])
                    # 횡단 방향 거리
                    lateral_dist = abs(np.dot(diff, perpendicular))
                    # 도로 방향 거리
                    longitudinal_dist = abs(np.dot(diff, main_direction))

                    # 횡단 방향 또는 도로 방향 거리가 초과하면 버퍼 flush
                    if lateral_dist > max_lateral_gap or longitudinal_dist > max_longitudinal_gap:
                        result = flush_buffer(buffer, road, side, merged_idx, self.lat_to_m, self.lon_to_m)
                        if result:
                            merged_results.append(result)
                            merged_idx += 1
                        buffer = {
                            'segments': [], 'tree_count': 0, 'total_length': 0,
                            'covered_length': 0, 'exposed_length': 0, 'n_gaps': 0,
                            'max_gap': 0, 'canopies': [], 'spacings': [],
                            'lons': [], 'lats': [], 'polylines': []
                        }

                # 버퍼에 세그먼트 추가
                buffer['segments'].append(seg['segment_name'])
                buffer['tree_count'] += seg['tree_count']
                buffer['total_length'] += seg['total_length_m']
                buffer['covered_length'] += seg['covered_length_m']
                buffer['exposed_length'] += seg['exposed_length_m']
                buffer['n_gaps'] += seg['n_gaps']
                buffer['max_gap'] = max(buffer['max_gap'], seg['max_gap_m'])
                buffer['canopies'].extend([seg['avg_canopy']] * seg['tree_count'])
                buffer['spacings'].append(seg['avg_spacing'])
                buffer['lons'].extend([seg['center_lon'], seg['min_lon'], seg['max_lon']])
                buffer['lats'].extend([seg['center_lat'], seg['min_lat'], seg['max_lat']])
                buffer['polylines'].append(seg['polyline'])

                prev_seg = seg  # 이전 세그먼트 업데이트

                # min_length 이상이면 flush
                if buffer['total_length'] >= min_length:
                    result = flush_buffer(buffer, road, side, merged_idx, self.lat_to_m, self.lon_to_m)
                    if result:
                        merged_results.append(result)
                        merged_idx += 1

                    # 버퍼 초기화
                    buffer = {
                        'segments': [], 'tree_count': 0, 'total_length': 0,
                        'covered_length': 0, 'exposed_length': 0, 'n_gaps': 0,
                        'max_gap': 0, 'canopies': [], 'spacings': [],
                        'lons': [], 'lats': [], 'polylines': []
                    }
                    prev_seg = None  # flush 후 prev_seg 초기화

            # 남은 버퍼 처리 (min_length 미만이어도 포함)
            if buffer['tree_count'] > 0:
                result = flush_buffer(buffer, road, side, merged_idx, self.lat_to_m, self.lon_to_m)
                if result:
                    merged_results.append(result)

        return pd.DataFrame(merged_results)

    def _get_rating(self, coverage_ratio):
        """Coverage Ratio 기반 등급"""
        if coverage_ratio >= 0.95:
            return 'Excellent'  # 95% 이상 그늘
        elif coverage_ratio >= 0.85:
            return 'Good'       # 85% 이상
        elif coverage_ratio >= 0.70:
            return 'Fair'       # 70% 이상
        elif coverage_ratio >= 0.50:
            return 'Poor'       # 50% 이상
        else:
            return 'Very Poor'  # 50% 미만

    def _print_summary(self, segments_df):
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("분석 결과 요약")
        print("=" * 60)

        print(f"\n총 세그먼트: {len(segments_df)}개")
        print(f"총 나무: {segments_df['tree_count'].sum()}그루")
        print(f"총 분석 거리: {segments_df['total_length_m'].sum()/1000:.1f}km")

        # 측면별 통계
        if 'side' in segments_df.columns:
            print("\n=== 측면별 분포 ===")
            for side in segments_df['side'].unique():
                count = len(segments_df[segments_df['side'] == side])
                print(f"  {side}측: {count}개 세그먼트")

        print("\n=== 등급별 분포 ===")
        for rating in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
            count = len(segments_df[segments_df['rating'] == rating])
            pct = count / len(segments_df) * 100
            print(f"  {rating}: {count}개 ({pct:.1f}%)")

        print("\n=== 전체 통계 ===")
        total_covered = segments_df['covered_length_m'].sum()
        total_exposed = segments_df['exposed_length_m'].sum()
        total_length = segments_df['total_length_m'].sum()
        overall_coverage = total_covered / total_length if total_length > 0 else 0

        print(f"  전체 Coverage Ratio: {overall_coverage:.1%}")
        print(f"  총 그늘 거리: {total_covered/1000:.2f}km")
        print(f"  총 노출 거리: {total_exposed/1000:.2f}km")
        print(f"  평균 Coverage Ratio: {segments_df['coverage_ratio'].mean():.1%}")
        print(f"  평균 gap 수/구간: {segments_df['n_gaps'].mean():.1f}개")

        print("\n=== 상위 5개 구간 (Best) ===")
        top5 = segments_df.head(5)
        for _, row in top5.iterrows():
            print(f"  {row['segment_name']}: Coverage {row['coverage_ratio']:.1%}, "
                  f"{row['tree_count']}그루, {row['total_length_m']:.0f}m")

        print("\n=== 하위 5개 구간 (Needs Improvement) ===")
        bottom5 = segments_df.tail(5)
        for _, row in bottom5.iterrows():
            print(f"  {row['segment_name']}: Coverage {row['coverage_ratio']:.1%}, "
                  f"노출 {row['exposed_length_m']:.0f}m, gap {row['n_gaps']}개")


def main():
    """메인 실행"""
    # 프로젝트 루트 디렉토리 (src/4_shade_analysis/ → Archive/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 강남구 데이터
    input_path = os.path.join(base_dir, 'seoul_trees_output', '강남구_roadside_trees_with_roads.csv')
    output_dir = os.path.join(base_dir, 'shade_network_output')

    if not os.path.exists(input_path):
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return

    # 데이터 로드
    print(f"데이터 로드: {input_path}")
    trees_df = pd.read_csv(input_path)
    print(f"총 {len(trees_df)}그루 로드")

    # 분석 실행
    analyzer = SegmentSCIAnalyzer(
        shade_factor=0.65,
        max_gap=8.0,              # 연결 허용 gap (10걸음)
        segment_split_gap=25.0,   # 구간 분리 gap (교차로 수준)
        max_lateral_distance=6.0   # 도로 횡단 감지 (6m 초과시 분리)
    )

    segments_df = analyzer.analyze(trees_df)

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gangnam_sci_segments.csv')
    segments_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {output_path}")


if __name__ == '__main__':
    main()
