
"""
공원수목 전용 수집기 (WGS84 API)
기존 trees_clean.geojson에 공원수목 데이터 추가
"""

import requests
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
import time
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParkTreesCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def safe_get_text(self, element, tag_name: str, default_value: str = "") -> str:
        """XML 요소에서 안전하게 텍스트 추출"""
        try:
            elem = element.find(tag_name)
            if elem is not None and elem.text is not None:
                return elem.text.strip()
            return default_value
        except Exception:
            return default_value
    
    def safe_get_float(self, element, tag_name: str, default_value: Optional[float] = None) -> Optional[float]:
        """XML 요소에서 안전하게 float 값 추출"""
        try:
            elem = element.find(tag_name)
            if elem is not None and elem.text is not None and elem.text.strip():
                return float(elem.text.strip())
            return default_value
        except (ValueError, TypeError):
            return default_value
    
    def fetch_xml_data(self, url: str) -> Optional[ET.Element]:
        """API에서 XML 데이터 가져오기"""
        try:
            logger.info(f"API 요청: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # API 오류 응답 체크
            result_code = root.find('.//RESULT/CODE')
            if result_code is not None and result_code.text != 'INFO-000':
                error_msg = root.find('.//RESULT/MESSAGE')
                error_text = error_msg.text if error_msg is not None else "Unknown error"
                logger.error(f"API 오류: {result_code.text} - {error_text}")
                return None
            
            return root
            
        except requests.RequestException as e:
            logger.error(f"API 요청 실패: {url} - {e}")
            return None
        except ET.ParseError as e:
            logger.error(f"XML 파싱 실패: {url} - {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 오류: {url} - {e}")
            return None
    
    def collect_park_trees(self) -> list:
        """공원 및 사유지수목 데이터 수집 (WGS84 API)"""
        logger.info("🌳 공원 및 사유지수목 데이터 수집 시작 (WGS84 API)...")
        park_features = []
        
        # 전체 개수 먼저 확인
        count_url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/GeoInfoParkAndPrivateLandWGS/1/1"
        count_root = self.fetch_xml_data(count_url)
        
        if count_root is None:
            logger.error("데이터 개수 확인 실패")
            return park_features
        
        total_count_elem = count_root.find('.//list_total_count')
        if total_count_elem is not None:
            total_count = int(total_count_elem.text)
            logger.info(f"📊 공원수목 전체 데이터 개수: {total_count:,}건")
        else:
            total_count = 1000  # 기본값
            logger.warning("전체 개수 확인 실패, 기본값 사용")
        
        # 배치로 데이터 수집
        batch_size = 1000
        start_idx = 1
        total_collected = 0
        
        while start_idx <= total_count:
            end_idx = min(start_idx + batch_size - 1, total_count)
            url = f"http://openAPI.seoul.go.kr:8088/{self.api_key}/xml/GeoInfoParkAndPrivateLandWGS/{start_idx}/{end_idx}"
            
            logger.info(f"📥 공원수목 데이터 수집 중: {start_idx:,}~{end_idx:,} ({total_collected:,}/{total_count:,})")
            root = self.fetch_xml_data(url)
            
            if root is None:
                logger.warning(f"배치 {start_idx} 실패, 다음 배치로 진행")
                start_idx = end_idx + 1
                continue
            
            batch_count = 0
            for row in root.findall('.//row'):
                try:
                    # 기본 정보 추출
                    species_kr = self.safe_get_text(row, 'WDPT_NM', "미상")
                    dbh_cm = self.safe_get_float(row, 'BHT_DM')
                    height_m = self.safe_get_float(row, 'THT_HG')
                    
                    # 위치 정보
                    borough = self.safe_get_text(row, 'GU_NM')
                    district = self.safe_get_text(row, 'DONG_NM')
                    address = self.safe_get_text(row, 'LC')
                    
                    # 좌표 (이미 WGS84)
                    lon = self.safe_get_float(row, 'LNG')
                    lat = self.safe_get_float(row, 'LAT')
                    
                    if lon is None or lat is None:
                        logger.warning(f"좌표 누락: {species_kr}")
                        continue
                    
                    # 서울 지역 범위 체크 (대략적)
                    if not (126.7 <= lon <= 127.3 and 37.4 <= lat <= 37.7):
                        logger.warning(f"서울 범위 밖 좌표: {lon}, {lat}")
                        continue
                    
                    source_id = self.safe_get_text(row, 'OBJECTID', f"park_wgs_{total_collected}")
                    
                    # DBH 기준 크기 분류
                    if dbh_cm:
                        if dbh_cm < 20:
                            size_class = "small"
                        elif dbh_cm < 50:
                            size_class = "medium"
                        else:
                            size_class = "large"
                    else:
                        size_class = "unknown"
                    
                    # GeoJSON Feature 생성
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "species_kr": species_kr,
                            "dbh_cm": dbh_cm,
                            "height_m": height_m,
                            "age_years": None,
                            "borough": borough,
                            "district": district,
                            "address": address,
                            "tree_type": "park",
                            "size_class": size_class,
                            "source_id": source_id
                        }
                    }
                    
                    park_features.append(feature)
                    batch_count += 1
                    total_collected += 1
                    
                except Exception as e:
                    logger.warning(f"행 처리 오류: {e}")
                    continue
            
            logger.info(f"✅ 배치 완료: {batch_count:,}건 수집 (누적: {total_collected:,}건)")
            start_idx = end_idx + 1
            
            # API 부하 방지
            time.sleep(0.3)
            
            # 진행률 표시
            progress = (total_collected / total_count) * 100
            logger.info(f"📈 진행률: {progress:.1f}%")
        
        logger.info(f"🎉 공원수목 데이터 수집 완료: {total_collected:,}건")
        return park_features
    
    def add_to_existing_geojson(self, park_features: list, 
                              input_file: str = "seoul_trees_output/trees_clean.geojson",
                              output_file: str = "seoul_trees_output/trees_with_park.geojson") -> bool:
        """기존 GeoJSON에 공원수목 추가"""
        try:
            # 기존 파일 로드
            input_path = Path(input_file)
            if not input_path.exists():
                logger.error(f"기존 파일이 없습니다: {input_file}")
                return False
            
            logger.info(f"📂 기존 파일 로드: {input_file}")
            with open(input_path, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
            
            original_count = len(geojson['features'])
            logger.info(f"📊 기존 데이터: {original_count:,}건")
            
            # 새 데이터 추가
            geojson['features'].extend(park_features)
            
            new_count = len(geojson['features'])
            logger.info(f"📈 추가 후 총 데이터: {new_count:,}건 (+{len(park_features):,}건)")
            
            # 업데이트된 파일 저장
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True)
            
            logger.info(f"💾 업데이트된 파일 저장: {output_file}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)
            
            # 파일 크기 확인
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"📁 파일 크기: {file_size:.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"파일 처리 오류: {e}")
            return False
    
    def run_collection(self):
        """전체 수집 프로세스 실행"""
        logger.info("=== 공원수목 데이터 추가 시작 ===")
        
        try:
            # 1. 공원수목 데이터 수집
            park_features = self.collect_park_trees()
            
            if not park_features:
                logger.error("수집된 공원수목 데이터가 없습니다.")
                return False
            
            # 2. 기존 GeoJSON에 추가
            success = self.add_to_existing_geojson(park_features)
            
            if success:
                logger.info("🎉 공원수목 데이터 추가 완료!")
                logger.info("📋 다음 단계:")
                logger.info("  1. trees_with_park.geojson → MBTiles 생성")
                logger.info("  2. Mapbox에 업로드")
                logger.info("  3. 웹사이트에서 확인")
                return True
            else:
                logger.error("❌ 파일 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"전체 프로세스 오류: {e}")
            return False

# 사용 예시
if __name__ == "__main__":
    import os

    # API 키 설정 (환경변수에서 읽기)
    API_KEY = os.getenv('SEOUL_API_KEY')
    if not API_KEY:
        print("❌ 환경변수 SEOUL_API_KEY를 설정해주세요.")
        print("   export SEOUL_API_KEY='your_api_key'")
        exit(1)

    # 수집기 생성 및 실행
    collector = ParkTreesCollector(api_key=API_KEY)
    success = collector.run_collection()
    
    if success:
        print("\n🌳 성공! 이제 다음 명령어로 MBTiles를 생성하세요:")
        print("cd seoul_trees_output")
        print("tippecanoe -o seoul-trees-with-park.mbtiles -z 16 -Z 9 --drop-densest-as-needed trees_with_park.geojson")
    else:
        print("\n❌ 실패. 로그를 확인하세요.")