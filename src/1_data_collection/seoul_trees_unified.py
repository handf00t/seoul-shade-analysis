#!/usr/bin/env python3
"""
서울시 나무 데이터 통합 수집기
- 보호수, 가로수, 공원수목 데이터를 한 번에 수집
- WGS84 좌표계 API 사용 (변환 불필요)
- 수관너비(canopy_width) 포함
- GeoJSON 출력
"""

import requests
import xml.etree.ElementTree as ET
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TreeData:
    """표준화된 나무 데이터 구조"""
    species_kr: str
    dbh_cm: Optional[float]
    height_m: Optional[float]
    canopy_width_m: Optional[float]
    borough: str
    district: str
    address: str
    tree_type: str  # 'protected', 'roadside', 'park'
    longitude: float
    latitude: float
    source_id: str


class SeoulTreesUnifiedCollector:
    """서울시 나무 데이터 통합 수집기"""

    def __init__(self, api_key: str, output_dir: str = "seoul_trees_output"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 서울시 25개 구
        self.districts = [
            '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구',
            '성북구', '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구',
            '양천구', '강서구', '구로구', '금천구', '영등포구', '동작구', '관악구',
            '서초구', '강남구', '송파구', '강동구'
        ]

        # API 엔드포인트 (모두 WGS84 버전 사용)
        self.api_endpoints = {
            'protected': 'GeoInfoNurseTreeOldTreeWGS',  # 보호수/노거수
            'roadside': 'GeoInfoOfRoadsideTreeW',        # 가로수
            'park': 'GeoInfoParkAndPrivateLandWGS'       # 공원/사유지수목
        }

        # 통계
        self.stats = {
            'protected': 0,
            'roadside': 0,
            'park': 0,
            'skipped': 0
        }

    def safe_get_text(self, element, tag_name: str, default: str = "") -> str:
        """XML에서 안전하게 텍스트 추출"""
        try:
            elem = element.find(tag_name)
            if elem is not None and elem.text is not None:
                return elem.text.strip()
            return default
        except Exception:
            return default

    def safe_get_float(self, element, tag_name: str) -> Optional[float]:
        """XML에서 안전하게 float 추출"""
        try:
            elem = element.find(tag_name)
            if elem is not None and elem.text is not None and elem.text.strip():
                return float(elem.text.strip())
            return None
        except (ValueError, TypeError):
            return None

    def fetch_xml(self, url: str) -> Optional[ET.Element]:
        """API에서 XML 데이터 가져오기"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # API 오류 체크
            result_code = root.find('.//RESULT/CODE')
            if result_code is not None and result_code.text != 'INFO-000':
                error_msg = root.find('.//RESULT/MESSAGE')
                error_text = error_msg.text if error_msg is not None else "Unknown"
                logger.warning(f"API 오류: {result_code.text} - {error_text}")
                return None

            return root

        except requests.RequestException as e:
            logger.error(f"API 요청 실패: {e}")
            return None
        except ET.ParseError as e:
            logger.error(f"XML 파싱 실패: {e}")
            return None

    def get_total_count(self, endpoint: str, district: str = None) -> int:
        """API의 전체 데이터 개수 조회"""
        if district:
            url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/{endpoint}/1/1/{district}"
        else:
            url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/{endpoint}/1/1"

        root = self.fetch_xml(url)
        if root is None:
            return 0

        total_elem = root.find('.//list_total_count')
        if total_elem is not None and total_elem.text:
            return int(total_elem.text)
        return 0

    def is_valid_seoul_coords(self, lon: float, lat: float) -> bool:
        """서울 범위 내 좌표인지 확인"""
        return 126.7 <= lon <= 127.3 and 37.4 <= lat <= 37.7

    def collect_protected_trees(self) -> List[TreeData]:
        """보호수/노거수 데이터 수집"""
        logger.info("=" * 60)
        logger.info("보호수/노거수 데이터 수집 시작")
        logger.info("=" * 60)

        trees = []
        endpoint = self.api_endpoints['protected']
        total_count = self.get_total_count(endpoint)

        if total_count == 0:
            logger.warning("보호수 데이터가 없습니다.")
            return trees

        logger.info(f"전체 보호수: {total_count:,}건")

        batch_size = 1000
        for start_idx in range(1, total_count + 1, batch_size):
            end_idx = min(start_idx + batch_size - 1, total_count)
            url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/{endpoint}/{start_idx}/{end_idx}"

            root = self.fetch_xml(url)
            if root is None:
                continue

            batch_count = 0
            for row in root.findall('.//row'):
                # 필드 추출 (보호수 API 필드명)
                species_kr = self.safe_get_text(row, 'TRSPC_KORN', "미상")
                dbh_cm = self.safe_get_float(row, 'BHT_GRH')  # 흉고둘레 (cm)
                height_m = self.safe_get_float(row, 'THT_HGT')
                canopy_width_m = self.safe_get_float(row, 'ASCTL_BRETH')

                borough = self.safe_get_text(row, 'GU_NM')
                district = self.safe_get_text(row, 'DONG_NM')
                address = self.safe_get_text(row, 'LCTN')

                lon = self.safe_get_float(row, 'LOT')  # LOT = 경도
                lat = self.safe_get_float(row, 'LAT')

                if lon is None or lat is None:
                    self.stats['skipped'] += 1
                    continue

                if not self.is_valid_seoul_coords(lon, lat):
                    self.stats['skipped'] += 1
                    continue

                source_id = self.safe_get_text(row, 'OBJECTID', f"protected_{len(trees)}")

                trees.append(TreeData(
                    species_kr=species_kr,
                    dbh_cm=dbh_cm,
                    height_m=height_m,
                    canopy_width_m=canopy_width_m,
                    borough=borough,
                    district=district,
                    address=address,
                    tree_type='protected',
                    longitude=lon,
                    latitude=lat,
                    source_id=source_id
                ))
                batch_count += 1

            logger.info(f"  배치 {start_idx:,}~{end_idx:,}: {batch_count}건 수집")
            time.sleep(0.3)

        self.stats['protected'] = len(trees)
        logger.info(f"보호수 수집 완료: {len(trees):,}건")
        return trees

    def collect_roadside_trees(self) -> List[TreeData]:
        """가로수 데이터 수집 (모든 구)"""
        logger.info("=" * 60)
        logger.info("가로수 데이터 수집 시작")
        logger.info("=" * 60)

        trees = []
        endpoint = self.api_endpoints['roadside']

        for district in self.districts:
            total_count = self.get_total_count(endpoint, district)

            if total_count == 0:
                logger.info(f"  {district}: 데이터 없음")
                continue

            logger.info(f"  {district}: {total_count:,}건 수집 시작")
            district_count = 0

            batch_size = 1000
            for start_idx in range(1, total_count + 1, batch_size):
                end_idx = min(start_idx + batch_size - 1, total_count)
                url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/{endpoint}/{start_idx}/{end_idx}/{district}"

                root = self.fetch_xml(url)
                if root is None:
                    continue

                for row in root.findall('.//row'):
                    # 필드 추출 (가로수 API 필드명)
                    species_kr = self.safe_get_text(row, 'TREE_NM', "미상")
                    dbh_cm = self.safe_get_float(row, 'BHT_DMM')  # 흉고직경 (mm -> cm 변환 필요시)
                    height_m = self.safe_get_float(row, 'THT_HGT')
                    canopy_width_m = self.safe_get_float(row, 'ASCTL_BRETH')

                    borough = self.safe_get_text(row, 'GU_NM') or district
                    address = self.safe_get_text(row, 'PSTN')
                    road_name = self.safe_get_text(row, 'WDTH_NM')
                    full_address = f"{address} {road_name}".strip()

                    lon = self.safe_get_float(row, 'LOT')  # LOT = 경도
                    lat = self.safe_get_float(row, 'LAT')

                    if lon is None or lat is None:
                        self.stats['skipped'] += 1
                        continue

                    if not self.is_valid_seoul_coords(lon, lat):
                        self.stats['skipped'] += 1
                        continue

                    source_id = self.safe_get_text(row, 'OBJECTID', f"roadside_{len(trees)}")

                    trees.append(TreeData(
                        species_kr=species_kr,
                        dbh_cm=dbh_cm,
                        height_m=height_m,
                        canopy_width_m=canopy_width_m,
                        borough=borough,
                        district="",
                        address=full_address,
                        tree_type='roadside',
                        longitude=lon,
                        latitude=lat,
                        source_id=source_id
                    ))
                    district_count += 1

                time.sleep(0.2)

            logger.info(f"    -> {district_count:,}건 완료")

        self.stats['roadside'] = len(trees)
        logger.info(f"가로수 수집 완료: {len(trees):,}건")
        return trees

    def collect_park_trees(self) -> List[TreeData]:
        """공원/사유지 수목 데이터 수집"""
        logger.info("=" * 60)
        logger.info("공원/사유지 수목 데이터 수집 시작")
        logger.info("=" * 60)

        trees = []
        endpoint = self.api_endpoints['park']
        total_count = self.get_total_count(endpoint)

        if total_count == 0:
            logger.warning("공원수목 데이터가 없습니다.")
            return trees

        logger.info(f"전체 공원수목: {total_count:,}건")

        batch_size = 1000
        for start_idx in range(1, total_count + 1, batch_size):
            end_idx = min(start_idx + batch_size - 1, total_count)
            url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/{endpoint}/{start_idx}/{end_idx}"

            root = self.fetch_xml(url)
            if root is None:
                logger.warning(f"  배치 {start_idx} 실패, 건너뜀")
                continue

            batch_count = 0
            for row in root.findall('.//row'):
                # 필드 추출 (공원수목 API 필드명)
                species_kr = self.safe_get_text(row, 'TREE_NM', "미상")
                dbh_cm = self.safe_get_float(row, 'BHT_DMM')  # 흉고직경
                height_m = self.safe_get_float(row, 'THT_HGT')
                canopy_width_m = self.safe_get_float(row, 'ASCTL_BRETH')

                borough = self.safe_get_text(row, 'GU_NM')
                district = self.safe_get_text(row, 'DONG')
                address = self.safe_get_text(row, 'PSTN')

                lon = self.safe_get_float(row, 'LOT')  # LOT = 경도
                lat = self.safe_get_float(row, 'LAT')

                if lon is None or lat is None:
                    self.stats['skipped'] += 1
                    continue

                if not self.is_valid_seoul_coords(lon, lat):
                    self.stats['skipped'] += 1
                    continue

                source_id = self.safe_get_text(row, 'OBJECTID', f"park_{len(trees)}")

                trees.append(TreeData(
                    species_kr=species_kr,
                    dbh_cm=dbh_cm,
                    height_m=height_m,
                    canopy_width_m=canopy_width_m,
                    borough=borough,
                    district=district,
                    address=address,
                    tree_type='park',
                    longitude=lon,
                    latitude=lat,
                    source_id=source_id
                ))
                batch_count += 1

            # 진행률 표시
            progress = min(end_idx / total_count * 100, 100)
            logger.info(f"  배치 {start_idx:,}~{end_idx:,}: {batch_count}건 ({progress:.1f}%)")
            time.sleep(0.3)

        self.stats['park'] = len(trees)
        logger.info(f"공원수목 수집 완료: {len(trees):,}건")
        return trees

    def create_geojson(self, trees: List[TreeData]) -> Dict:
        """GeoJSON 생성"""
        features = []

        for tree in trees:
            # DBH 기반 크기 분류
            if tree.dbh_cm:
                if tree.dbh_cm < 20:
                    size_class = "small"
                elif tree.dbh_cm < 50:
                    size_class = "medium"
                else:
                    size_class = "large"
            else:
                size_class = "unknown"

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [tree.longitude, tree.latitude]
                },
                "properties": {
                    "species_kr": tree.species_kr,
                    "dbh_cm": tree.dbh_cm,
                    "height_m": tree.height_m,
                    "canopy_width_m": tree.canopy_width_m,
                    "borough": tree.borough,
                    "district": tree.district,
                    "address": tree.address,
                    "tree_type": tree.tree_type,
                    "size_class": size_class,
                    "source_id": tree.source_id
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def print_statistics(self, trees: List[TreeData]):
        """수집 통계 출력"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("수집 통계")
        logger.info("=" * 60)

        total = len(trees)
        logger.info(f"총 수집: {total:,}건")
        logger.info(f"  - 보호수: {self.stats['protected']:,}건")
        logger.info(f"  - 가로수: {self.stats['roadside']:,}건")
        logger.info(f"  - 공원수목: {self.stats['park']:,}건")
        logger.info(f"  - 제외(좌표 누락/범위 외): {self.stats['skipped']:,}건")

        # 수종별 통계
        species_count = {}
        for tree in trees:
            species_count[tree.species_kr] = species_count.get(tree.species_kr, 0) + 1

        logger.info("")
        logger.info("상위 10개 수종:")
        for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {species}: {count:,}건")

        # 구별 통계
        borough_count = {}
        for tree in trees:
            if tree.borough:
                borough_count[tree.borough] = borough_count.get(tree.borough, 0) + 1

        logger.info("")
        logger.info("구별 분포 (상위 10개):")
        for borough, count in sorted(borough_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {borough}: {count:,}건")

        # 수관너비 통계
        canopy_trees = [t for t in trees if t.canopy_width_m is not None]
        if canopy_trees:
            avg_canopy = sum(t.canopy_width_m for t in canopy_trees) / len(canopy_trees)
            logger.info("")
            logger.info(f"수관너비 데이터: {len(canopy_trees):,}건 ({len(canopy_trees)/total*100:.1f}%)")
            logger.info(f"평균 수관너비: {avg_canopy:.1f}m")

    def run(self) -> bool:
        """전체 수집 프로세스 실행"""
        logger.info("")
        logger.info("*" * 60)
        logger.info("  서울시 나무 데이터 통합 수집 시작")
        logger.info("*" * 60)
        logger.info("")

        start_time = time.time()

        try:
            # 1. 모든 데이터 수집
            all_trees = []
            all_trees.extend(self.collect_protected_trees())
            all_trees.extend(self.collect_roadside_trees())
            all_trees.extend(self.collect_park_trees())

            if not all_trees:
                logger.error("수집된 데이터가 없습니다.")
                return False

            # 2. 통계 출력
            self.print_statistics(all_trees)

            # 3. GeoJSON 생성 및 저장
            geojson = self.create_geojson(all_trees)

            output_path = self.output_dir / "trees_unified.geojson"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)

            # 파일 크기 확인
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            elapsed = time.time() - start_time

            logger.info("")
            logger.info("=" * 60)
            logger.info("수집 완료!")
            logger.info("=" * 60)
            logger.info(f"출력 파일: {output_path}")
            logger.info(f"파일 크기: {file_size_mb:.1f}MB")
            logger.info(f"소요 시간: {elapsed/60:.1f}분")

            return True

        except Exception as e:
            logger.error(f"수집 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # API 키 설정 (환경변수에서 읽기)
    API_KEY = os.getenv('SEOUL_API_KEY')
    if not API_KEY:
        print("❌ 환경변수 SEOUL_API_KEY를 설정해주세요.")
        print("   export SEOUL_API_KEY='your_api_key'")
        print("   API 키는 https://data.seoul.go.kr 에서 발급받을 수 있습니다.")
        exit(1)

    # 수집기 생성 및 실행
    collector = SeoulTreesUnifiedCollector(
        api_key=API_KEY,
        output_dir="seoul_trees_output"
    )

    success = collector.run()

    if success:
        print("\n" + "=" * 60)
        print("서울시 나무 데이터 수집 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  - seoul_trees_output/trees_unified.geojson")
        print("\n다음 단계:")
        print("  1. ecobenefits.py로 생태적 편익 계산")
        print("  2. tippecanoe로 MBTiles 생성")
        print("  3. Mapbox Studio에서 스타일 설정")
    else:
        print("\n수집 실패. 로그를 확인하세요.")
