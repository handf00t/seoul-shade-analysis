#!/usr/bin/env python3
"""
서울시 나무 생태적 편익 계산기 (수정버전)
- 기존 나무 데이터를 활용한 생태적 편익 계산
- 한국 실정에 맞는 편익 계수 적용
- 수종별, 크기별 차별화된 계산
- 주요 수정사항:
  1. 탄소 편익 과대 계산 문제 해결
  2. 에너지 계산 함수 중복 코드 제거
  3. 수관면적 추정 공식 개선
  4. 종별 보정계수 이중 반영 방지
"""

import pandas as pd
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EcologicalBenefits:
    """생태적 편익 결과"""
    tree_id: str
    species_kr: str
    dbh_cm: float
    canopy_area_m2: float  # 수관면적 추가
    # 연간 편익
    stormwater_liters: float
    stormwater_value_krw: float
    energy_kwh: float
    energy_value_krw: float
    air_pollution_kg: float
    air_pollution_value_krw: float
    carbon_storage_kg: float
    carbon_storage_value_krw: float
    total_annual_value_krw: float

class SeoulTreeBenefitsCalculator:
    def __init__(self):
        # 한국형 i-Tree Eco 연구 기반 편익 계수 (CCF 환산 적용)
        self.benefits_coefficients = {
            # 단위면적당 연간 편익 (수관 1㎡당) - CCF 0.35 기준 환산
            'stormwater_interception': {
                'base_rate': 157,  # L/㎡/년 (토지기준 55L ÷ CCF0.35 = 157L 수관기준)
                'price_per_liter': 0.85,  # 원/L (상수도 요금 기준)
            },
            'energy_conservation': {
                'base_kwh_per_m2': 0.6,   # B값: 원거리 기본 에너지 원단위 (kWh/㎡/년)
                'building_buffer_multiplier': 2.5,  # G값: 건물 10m 이내 가중치
                'price_per_kwh': 175,      # 원/kWh (2025년 기준)
                
                # 2020년 서울도시생태현황도 실제 불투수면적 비율 기반 건물 근접비율
                # 계산식: kWh/m² = B*(1-p) + (B*G)*p, 여기서 B=0.6, G=2.5
                # 출처: 서울열린데이터광장 OA-22363 공식 데이터
                'district_proximity_ratio': {
                    '중구': 0.750,       # 불투수 75.0% → 최고밀도 (CBD, 명동, 시청)
                    '동대문구': 0.750,   # 불투수 73.9% → 최고밀도 (동대문시장, 청량리)
                    '양천구': 0.672,     # 불투수 67.3% → 고밀도 (목동, 신정)
                    '성동구': 0.614,     # 불투수 63.4% → 고밀도 (뚝섬, 성수)
                    '구로구': 0.574,     # 불투수 60.7% → 중고밀도 (공단지역)
                    '금천구': 0.570,     # 불투수 60.4% → 중고밀도 (가산디지털단지)
                    '동작구': 0.565,     # 불투수 60.0% → 중고밀도 (한강공원)
                    '영등포구': 0.558,   # 불투수 59.6% → 중고밀도 (여의도, 영등포역)
                    '중랑구': 0.542,     # 불투수 58.5% → 중고밀도 (중랑천변)
                    '송파구': 0.523,     # 불투수 57.2% → 중밀도 (올림픽공원, 한강)
                    '광진구': 0.523,     # 불투수 57.2% → 중밀도 (건대, 구의)
                    '강남구': 0.488,     # 불투수 54.9% → 중밀도 (공원/대로변)
                    '마포구': 0.470,     # 불투수 53.7% → 중밀도 (한강공원, 홍대)
                    '서대문구': 0.459,   # 불투수 52.9% → 중밀도 (안산, 홍대)
                    '성북구': 0.453,     # 불투수 52.5% → 중밀도 (북한산 인근)
                    '강동구': 0.374,     # 불투수 47.2% → 중밀도 (길동, 둔촌)
                    '용산구': 0.329,     # 불투수 44.2% → 저밀도 (용산공원, 한강변)
                    '종로구': 0.297,     # 불투수 42.0% → 저밀도 (북촌, 인사동)
                    '도봉구': 0.273,     # 불투수 40.4% → 저밀도 (도봉산)
                    '강서구': 0.263,     # 불투수 39.7% → 저밀도 (한강공원, 김포공항)
                    '관악구': 0.255,     # 불투수 39.2% → 저밀도 (관악산)
                    '은평구': 0.245,     # 불투수 38.5% → 저밀도 (북한산, 불광천)
                    '노원구': 0.223,     # 불투수 37.0% → 저밀도 (중랑천)
                    '서초구': 0.210,     # 불투수 36.2% → 저밀도 (양재천, 우면산)
                    '강북구': 0.210,     # 불투수 36.1% → 저밀도 (북한산, 우이천)
                    'default': 0.450     # 서울시 전체 평균
                }
            },
            'air_purification': {
                'total_removal_g_per_m2': 129,  # g/㎡/년 (토지기준 45g ÷ CCF0.35 = 129g)
                # 오염물질별 비중 (수원시 + 울산 연구 기반)
                'pm25_ratio': 0.18,       # PM2.5 비중 (건강피해 큰 항목)
                'pm10_ratio': 0.22,       # PM10 비중
                'no2_ratio': 0.35,        # NO2 비중 (가장 높음)
                'so2_ratio': 0.10,        # SO2 비중
                'o3_ratio': 0.15,         # O3 비중
                'price_per_g': 3.2,       # 원/g (PM2.5 가중치 반영하여 상향)
            },
            'carbon_storage': {
                # 수정된 탄소 편익 계수 (도시 가로수 실정 반영)
                'base_co2_kg_per_m2': 1.2,  # kg/㎡/년 (기존 3.6에서 현실적 수준으로 하향)
                'price_per_kg': 45,         # 원/kg (K-ETS 평균 가격)
                # 개선된 상대생장식 적용
                'use_allometric_equation': True,
                'annual_growth_rate': 0.02  # 연간 성장률 2%로 현실적 조정 (기존 4%→2%)
            },
            # 서울시 평균 수관피복률 (참고용)
            'ccf': 0.35  # Crown Coverage Factor (가로수: 0.3, 공원: 0.4 정도)
        }
        
        # 수종별 보정계수 (주요 서울시 가로수/보호수 기준) - 이중반영 방지를 위해 단순화
        self.species_factors = {
            # 수관 특성은 canopy diameter 추정에만 사용, 편익 계산에는 공기정화만 적용
            '은행나무': {'air_purification': 1.1, 'canopy_coeff': {'a': 2.8, 'b': 0.95}},
            '플라타너스': {'air_purification': 1.2, 'canopy_coeff': {'a': 3.2, 'b': 1.00}},
            '버즘나무': {'air_purification': 1.2, 'canopy_coeff': {'a': 3.2, 'b': 1.00}},
            '소나무': {'air_purification': 1.3, 'canopy_coeff': {'a': 2.2, 'b': 0.85}},
            '느티나무': {'air_purification': 1.0, 'canopy_coeff': {'a': 3.5, 'b': 1.05}},
            '벚나무': {'air_purification': 0.9, 'canopy_coeff': {'a': 2.0, 'b': 0.88}},
            '메타세쿼이아': {'air_purification': 1.4, 'canopy_coeff': {'a': 2.0, 'b': 0.80}},
            '단풍나무': {'air_purification': 1.0, 'canopy_coeff': {'a': 2.4, 'b': 0.90}},
            '회화나무': {'air_purification': 1.1, 'canopy_coeff': {'a': 2.6, 'b': 0.92}},
            '참나무': {'air_purification': 1.0, 'canopy_coeff': {'a': 2.8, 'b': 0.95}},
            # 미상 처리
            '미상': {'air_purification': 1.0, 'canopy_coeff': {'a': 2.5, 'b': 0.90}},
            # 기본값 (백업용)
            'default': {'air_purification': 1.0, 'canopy_coeff': {'a': 2.5, 'b': 0.90}}
        }
    
    def normalize_species_name(self, species_name: str) -> str:
        """수종명 정규화"""
        if not species_name or species_name.strip() == "":
            return '미상'
        
        # 공백 제거 및 정리
        species_name = species_name.strip()
        
        # 일반적인 수종명으로 매핑
        name_mapping = {
            '양버즘나무': '플라타너스',
            '버즘나무': '플라타너스',
            '은행': '은행나무',
            '소나무류': '소나무',
            '잣나무': '소나무',
            '느릅나무': '느티나무',
            '왕벚나무': '벚나무',
            '산벚나무': '벚나무',
            '단풍류': '단풍나무',
            '참나무류': '참나무'
        }
        
        normalized = name_mapping.get(species_name, species_name)
        return normalized if normalized in self.species_factors else '미상'
    
    def calculate_canopy_area(self, tree_data: dict) -> float:
        """개선된 수관면적 계산 (수종별 회귀식 보정)"""
        
        # 1순위: 수관너비 데이터 활용
        canopy_width_m = tree_data.get('canopy_width_m')
        if canopy_width_m and canopy_width_m > 0:
            # 실제 수관너비로 면적 계산
            canopy_area_m2 = math.pi * (canopy_width_m / 2) ** 2
            return canopy_area_m2
        
        # 2순위: 개선된 DBH 기반 추정
        dbh_cm = tree_data.get('dbh_cm', 0) or 0
        if dbh_cm > 0:
            species_kr = tree_data.get('species_kr', '미상')
            normalized_species = self.normalize_species_name(species_kr)
            
            # 수종별 개선된 수관지름 추정 계수 (실측 데이터 기반)
            species_data = self.species_factors.get(normalized_species, self.species_factors['default'])
            coeffs = species_data['canopy_coeff']
            
            # 개선된 추정식: 지름(m) = a + b × DBH(m)
            estimated_diameter_m = coeffs['a'] + coeffs['b'] * (dbh_cm / 100)
            
            # 현실적 범위 제한
            estimated_diameter_m = max(1.5, min(estimated_diameter_m, 20.0))
            
            canopy_area_m2 = math.pi * (estimated_diameter_m / 2) ** 2
            return canopy_area_m2
        
        # 3순위: 기본값 (작은 나무 가정)
        default_area = math.pi * (2.5 / 2) ** 2  # 지름 2.5m 가정 (기존 2.0m에서 상향)
        return default_area
    
    def calculate_stormwater_benefits(self, canopy_area_m2: float) -> tuple:
        """우수차집 편익 계산 (종별 보정 제거)"""
        coeff = self.benefits_coefficients['stormwater_interception']
        
        annual_liters = canopy_area_m2 * coeff['base_rate']
        annual_value = annual_liters * coeff['price_per_liter']
        
        return annual_liters, annual_value
    
    def calculate_energy_benefits(self, canopy_area_m2: float, tree_data: dict) -> tuple:
        """에너지 절약 편익 계산 (불투수면적 기반 구별 근접비율 적용, 중복 코드 제거)"""
        coeff = self.benefits_coefficients['energy_conservation']
        
        # 개별 나무의 건물 근접도 확인 (향후 확장용)
        is_near_building = tree_data.get('is_near_bld10m')
        
        if is_near_building is not None:
            # 개별 나무의 근접도 데이터가 있는 경우 (향후 구현)
            if is_near_building == 1:
                kwh_per_m2 = coeff['base_kwh_per_m2'] * coeff['building_buffer_multiplier']
            else:
                kwh_per_m2 = coeff['base_kwh_per_m2']
        else:
            # 불투수면적 기반 구별 근접비율 적용 (현재 방식)
            district = tree_data.get('borough', 'default')
            proximity_ratio = coeff['district_proximity_ratio'].get(district, 0.450)
            
            B = coeff['base_kwh_per_m2']      # 0.6 kWh/m² (원거리 기본값)
            G = coeff['building_buffer_multiplier']  # 2.5 (근접 가중치)
            
            # 불투수면적 기반 가중평균 : kWh/m² = B*(1-p) + (B*G)*p
            kwh_per_m2 = B * (1 - proximity_ratio) + (B * G) * proximity_ratio
        
        total_kwh = canopy_area_m2 * kwh_per_m2
        annual_value = total_kwh * coeff['price_per_kwh']
        
        return total_kwh, annual_value
    
    def calculate_air_purification_benefits(self, canopy_area_m2: float, species_factor: float) -> tuple:
        """대기정화 편익 계산 (한국 연구 기반 오염물질 비중 적용)"""
        coeff = self.benefits_coefficients['air_purification']
        
        # 총 오염물질 제거량 (종별 보정 적용)
        total_removal_g = canopy_area_m2 * coeff['total_removal_g_per_m2'] * species_factor
        
        # 오염물질별 세분화 (참고용)
        pm25_removal = total_removal_g * coeff['pm25_ratio']
        pm10_removal = total_removal_g * coeff['pm10_ratio'] 
        no2_removal = total_removal_g * coeff['no2_ratio']
        so2_removal = total_removal_g * coeff['so2_ratio']
        o3_removal = total_removal_g * coeff['o3_ratio']
        
        total_removal_kg = total_removal_g / 1000
        annual_value = total_removal_g * coeff['price_per_g']
        
        return total_removal_kg, annual_value
    
    def calculate_carbon_benefits_allometric(self, dbh_cm: float, species_kr: str) -> tuple:
        """수정된 상대생장식 기반 탄소흡수량 계산 (현실적 수준으로 조정)"""
        if dbh_cm <= 0:
            return 0, 0
            
        # 수정된 상대생장식 계수 (도시 가로수 특성 반영)
        # 기존 계수를 1/10 ~ 1/20 수준으로 하향 조정
        allometric_params = {
            # 활엽수 (단위: kg)
            '느티나무': {'a': 0.0034, 'b': 2.65},    # 기존 0.0678 → 0.0034
            '플라타너스': {'a': 0.0036, 'b': 2.55},  # 기존 0.0712 → 0.0036
            '은행나무': {'a': 0.0032, 'b': 2.60},    # 기존 0.0634 → 0.0032
            '벚나무': {'a': 0.0026, 'b': 2.50},      # 기존 0.0523 → 0.0026
            '단풍나무': {'a': 0.0030, 'b': 2.58},    # 기존 0.0598 → 0.0030
            '회화나무': {'a': 0.0032, 'b': 2.65},    # 기존 0.0645 → 0.0032
            '참나무': {'a': 0.0034, 'b': 2.62},      # 기존 0.0687 → 0.0034
            # 침엽수
            '소나무': {'a': 0.0023, 'b': 2.70},      # 기존 0.0456 → 0.0023
            '메타세쿼이아': {'a': 0.0026, 'b': 2.68}, # 기존 0.0523 → 0.0026
            # 기본값 (활엽수 평균)
            '미상': {'a': 0.0031, 'b': 2.60},
            'default': {'a': 0.0031, 'b': 2.60}
        }
        
        # 수종별 파라미터 선택
        normalized_species = self.normalize_species_name(species_kr)
        params = allometric_params.get(normalized_species, allometric_params['default'])
        
        # 상대생장식으로 바이오매스 계산 (kg)
        biomass_kg = params['a'] * (dbh_cm ** params['b'])
        
        # 바이오매스 → 탄소량 (46% 적용) → CO2 (×44/12)
        carbon_kg = biomass_kg * 0.46
        co2_kg_total = carbon_kg * (44/12)
        
        # 연간 흡수량 (총 저장량의 2%로 현실적 조정)
        growth_rate = self.benefits_coefficients['carbon_storage']['annual_growth_rate']
        annual_co2_kg = co2_kg_total * growth_rate
        
        # 경제적 가치
        coeff = self.benefits_coefficients['carbon_storage']
        annual_value = annual_co2_kg * coeff['price_per_kg']
        
        return annual_co2_kg, annual_value
    
    def calculate_carbon_benefits_legacy(self, canopy_area_m2: float, dbh_cm: float) -> tuple:
        """수정된 수관면적 기반 탄소저장 편익 계산"""
        coeff = self.benefits_coefficients['carbon_storage']
        
        # DBH 기반 크기 보정 (현실적 범위)
        size_factor = min(dbh_cm / 30, 2.0)  # 30cm를 기준으로 최대 2배
        
        annual_co2_storage = canopy_area_m2 * coeff['base_co2_kg_per_m2'] * size_factor
        annual_value = annual_co2_storage * coeff['price_per_kg']
        
        return annual_co2_storage, annual_value 
    
    def calculate_tree_benefits(self, tree_data: dict) -> EcologicalBenefits:
        """개별 나무의 생태적 편익 계산 (수관너비 우선 활용, 이중반영 방지)"""
        species_kr = tree_data.get('species_kr', 'default')
        dbh_cm = tree_data.get('dbh_cm', 0) or 0
        canopy_width_m = tree_data.get('canopy_width_m')
        tree_id = tree_data.get('source_id', 'unknown')
        
        # DBH 기본값 처리
        if dbh_cm <= 0:
            dbh_cm = 15  # 15cm 기본값
        
        # 수종별 보정계수 (공기정화만 적용)
        normalized_species = self.normalize_species_name(species_kr)
        species_data = self.species_factors.get(normalized_species, self.species_factors['미상'])
        air_factor = species_data['air_purification']
        
        # 수관면적 계산 (수관너비 우선, DBH 백업)
        canopy_area = self.calculate_canopy_area(tree_data)
        
        # 실제 사용된 데이터 로깅 (개발용)
        if canopy_width_m and canopy_width_m > 0:
            data_source = f"수관너비 {canopy_width_m}m"
        elif dbh_cm > 0:
            data_source = f"DBH {dbh_cm}cm 추정"
        else:
            data_source = "기본값"
        
        # 각 편익 계산 (종별 보정은 공기정화만 적용)
        stormwater_liters, stormwater_value = self.calculate_stormwater_benefits(canopy_area)
        
        energy_kwh, energy_value = self.calculate_energy_benefits(canopy_area, tree_data)
        
        air_pollution_kg, air_pollution_value = self.calculate_air_purification_benefits(
            canopy_area, air_factor
        )
        
        # 개선된 상대생장식 기반 탄소 계산
        if self.benefits_coefficients['carbon_storage']['use_allometric_equation']:
            carbon_storage_kg, carbon_storage_value = self.calculate_carbon_benefits_allometric(
                dbh_cm, species_kr
            )
        else:
            # 수정된 수관면적 기반 방식
            carbon_storage_kg, carbon_storage_value = self.calculate_carbon_benefits_legacy(
                canopy_area, dbh_cm
            )
        
        total_value = stormwater_value + energy_value + air_pollution_value + carbon_storage_value
        
        return EcologicalBenefits(
            tree_id=tree_id,
            species_kr=species_kr,
            dbh_cm=dbh_cm,
            canopy_area_m2=canopy_area,
            stormwater_liters=stormwater_liters,
            stormwater_value_krw=stormwater_value,
            energy_kwh=energy_kwh,
            energy_value_krw=energy_value,
            air_pollution_kg=air_pollution_kg,
            air_pollution_value_krw=air_pollution_value,
            carbon_storage_kg=carbon_storage_kg,
            carbon_storage_value_krw=carbon_storage_value,
            total_annual_value_krw=total_value
        )
    
    def process_geojson(self, geojson_path: str, output_path: str = None) -> pd.DataFrame:
        """GeoJSON 파일의 모든 나무에 대한 편익 계산"""
        logger.info(f"GeoJSON 파일 처리 시작: {geojson_path}")
        
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        results = []
        total_trees = len(geojson_data['features'])
        
        for i, feature in enumerate(geojson_data['features']):
            if i % 1000 == 0:
                logger.info(f"처리 진행률: {i}/{total_trees} ({i/total_trees*100:.1f}%)")
            
            properties = feature['properties']
            tree_id = properties.get('source_id', 'unknown')
            benefits = self.calculate_tree_benefits(properties)
            
            # 고유 식별자 생성 (tree_id + 좌표 + 수종 조합)
            unique_id = f"{tree_id}_{properties.get('tree_type', '')}_{feature['geometry']['coordinates'][0]:.6f}_{feature['geometry']['coordinates'][1]:.6f}"
            
            # 결과를 딕셔너리로 변환
            result_dict = {
                'unique_id': unique_id,
                'tree_id': benefits.tree_id,
                'species_kr': benefits.species_kr,
                'dbh_cm': benefits.dbh_cm,
                'canopy_area_m2': round(benefits.canopy_area_m2, 2),
                'borough': properties.get('borough', ''),
                'tree_type': properties.get('tree_type', ''),
                'longitude': feature['geometry']['coordinates'][0],
                'latitude': feature['geometry']['coordinates'][1],
                'stormwater_liters_year': round(benefits.stormwater_liters, 1),
                'stormwater_value_krw_year': round(benefits.stormwater_value_krw, 0),
                'energy_kwh_year': round(benefits.energy_kwh, 1),
                'energy_value_krw_year': round(benefits.energy_value_krw, 0),
                'air_pollution_kg_year': round(benefits.air_pollution_kg, 2),
                'air_pollution_value_krw_year': round(benefits.air_pollution_value_krw, 0),
                'carbon_storage_kg_year': round(benefits.carbon_storage_kg, 1),
                'carbon_value_krw_year': round(benefits.carbon_storage_value_krw, 0),
                'total_annual_value_krw': round(benefits.total_annual_value_krw, 0)
            }
            results.append(result_dict)
        
        # DataFrame 생성
        df = pd.DataFrame(results)
        
        # 결과 저장
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"결과 저장 완료: {output_path}")
        
        # 요약 통계
        self.print_summary_statistics(df)
        
        return df
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """요약 통계 출력"""
        logger.info("=== 서울시 나무 생태적 편익 요약 (수정버전) ===")
        
        total_trees = len(df)
        
        # 전체 연간 편익
        total_stormwater = df['stormwater_liters_year'].sum()
        total_stormwater_value = df['stormwater_value_krw_year'].sum()
        total_energy = df['energy_kwh_year'].sum()
        total_energy_value = df['energy_value_krw_year'].sum()
        total_air_pollution = df['air_pollution_kg_year'].sum()
        total_air_pollution_value = df['air_pollution_value_krw_year'].sum()
        total_carbon = df['carbon_storage_kg_year'].sum()
        total_carbon_value = df['carbon_value_krw_year'].sum()
        total_value = df['total_annual_value_krw'].sum()
        
        # 평균 수관면적
        avg_canopy_area = df['canopy_area_m2'].mean()
        
        logger.info(f"총 나무 수: {total_trees:,}그루")
        logger.info(f"평균 수관면적: {avg_canopy_area:.1f}㎡/그루")
        logger.info(f"")
        logger.info(f"연간 우수차집: {total_stormwater:,.0f}L (가치: {total_stormwater_value:,.0f}원)")
        logger.info(f"연간 에너지절약: {total_energy:,.0f}kWh (가치: {total_energy_value:,.0f}원)")
        logger.info(f"연간 대기정화: {total_air_pollution:,.0f}kg (가치: {total_air_pollution_value:,.0f}원)")
        logger.info(f"연간 탄소저장: {total_carbon:,.0f}kg (가치: {total_carbon_value:,.0f}원)")
        logger.info(f"")
        logger.info(f"총 연간 생태적 편익: {total_value:,.0f}원 ({total_value/100000000:.1f}억원)")
        logger.info(f"나무 1그루당 평균 편익: {total_value/total_trees:,.0f}원/년")
        
        # 편익별 비중
        logger.info(f"")
        logger.info("편익별 구성비:")
        logger.info(f"  우수차집: {total_stormwater_value/total_value*100:.1f}%")
        logger.info(f"  에너지절약: {total_energy_value/total_value*100:.1f}%") 
        logger.info(f"  대기정화: {total_air_pollution_value/total_value*100:.1f}%")
        logger.info(f"  탄소저장: {total_carbon_value/total_value*100:.1f}%")
        
        # 구별 통계
        borough_stats = df.groupby('borough')['total_annual_value_krw'].agg(['count', 'sum']).round(0)
        borough_stats['avg_per_tree'] = (borough_stats['sum'] / borough_stats['count']).round(0)
        borough_stats = borough_stats.sort_values('sum', ascending=False)
        
        logger.info(f"")
        logger.info("구별 상위 5개 지역:")
        for borough, row in borough_stats.head().iterrows():
            logger.info(f"  {borough}: {row['count']:,}그루, {row['sum']:,.0f}원/년 (평균 {row['avg_per_tree']:,.0f}원/그루)")
        
        # 수종별 통계
        species_stats = df.groupby('species_kr')['total_annual_value_krw'].agg(['count', 'sum']).round(0)
        species_stats['avg_per_tree'] = (species_stats['sum'] / species_stats['count']).round(0)
        species_stats = species_stats.sort_values('sum', ascending=False)
        
        logger.info(f"")
        logger.info("수종별 상위 5개:")
        for species, row in species_stats.head().iterrows():
            logger.info(f"  {species}: {row['count']:,}그루, {row['sum']:,.0f}원/년 (평균 {row['avg_per_tree']:,.0f}원/그루)")
            
        # 크기별 통계 (DBH 구간별)
        df['dbh_class'] = pd.cut(df['dbh_cm'], 
                                bins=[0, 20, 40, 60, 80, 100, float('inf')],
                                labels=['~20cm', '21-40cm', '41-60cm', '61-80cm', '81-100cm', '100cm~'])
        
        dbh_stats = df.groupby('dbh_class', observed=True)['total_annual_value_krw'].agg(['count', 'mean']).round(0)
        
        logger.info(f"")
        logger.info("크기별 평균 편익:")
        for dbh_class, row in dbh_stats.iterrows():
            logger.info(f"  {dbh_class}: {row['count']:,}그루, 평균 {row['mean']:,.0f}원/그루")

    def create_benefits_geojson(self, original_geojson_path: str, benefits_df: pd.DataFrame, 
                               output_path: str):
        """편익 정보가 추가된 GeoJSON 생성"""
        logger.info("편익 정보가 포함된 GeoJSON 생성 중...")
        
        with open(original_geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # unique_id 기준으로 딕셔너리 생성
        benefits_dict = benefits_df.set_index('unique_id').to_dict('index')
        
        # 각 feature에 편익 정보 추가
        for feature in geojson_data['features']:
            tree_id = feature['properties'].get('source_id', 'unknown')
            tree_type = feature['properties'].get('tree_type', '')
            coords = feature['geometry']['coordinates']
            
            # 동일한 고유 식별자 생성
            unique_id = f"{tree_id}_{tree_type}_{coords[0]:.6f}_{coords[1]:.6f}"
            
            if unique_id in benefits_dict:
                benefits_data = benefits_dict[unique_id]
                
                # 편익 정보 추가
                feature['properties'].update({
                    'canopy_area_m2': benefits_data['canopy_area_m2'],
                    'stormwater_liters_year': benefits_data['stormwater_liters_year'],
                    'stormwater_value_krw_year': benefits_data['stormwater_value_krw_year'],
                    'energy_kwh_year': benefits_data['energy_kwh_year'],
                    'energy_value_krw_year': benefits_data['energy_value_krw_year'],
                    'air_pollution_kg_year': benefits_data['air_pollution_kg_year'],
                    'air_pollution_value_krw_year': benefits_data['air_pollution_value_krw_year'],
                    'carbon_storage_kg_year': benefits_data['carbon_storage_kg_year'],
                    'carbon_value_krw_year': benefits_data['carbon_value_krw_year'],
                    'total_annual_value_krw': benefits_data['total_annual_value_krw']
                })
        
        # 새로운 GeoJSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"편익 정보가 포함된 GeoJSON 저장 완료: {output_path}")

    def validate_calculation_results(self, df: pd.DataFrame) -> Dict[str, bool]:
        """계산 결과 검증"""
        logger.info("=== 계산 결과 검증 ===")
        
        validation_results = {}
        
        # 1. 탄소 편익 검증 (DBH 95cm 나무 기준)
        large_trees = df[df['dbh_cm'] >= 90]
        if not large_trees.empty:
            max_carbon = large_trees['carbon_storage_kg_year'].max()
            avg_carbon = large_trees['carbon_storage_kg_year'].mean()
            
            # 대형 나무도 연간 200kg 미만이어야 함
            carbon_check = max_carbon < 200
            validation_results['carbon_realistic'] = carbon_check
            
            logger.info(f"대형나무(DBH≥90cm) 탄소흡수: 최대 {max_carbon:.1f}kg/년, 평균 {avg_carbon:.1f}kg/년")
            logger.info(f"탄소 편익 현실성: {'✓ 양호' if carbon_check else '✗ 과대'}")
        
        # 2. 수관면적 검증
        avg_canopy = df['canopy_area_m2'].mean()
        large_canopy = df[df['dbh_cm'] >= 90]['canopy_area_m2'].mean()
        
        canopy_check = large_canopy > 20  # 대형나무는 20㎡ 이상이어야 함
        validation_results['canopy_reasonable'] = canopy_check
        
        logger.info(f"평균 수관면적: {avg_canopy:.1f}㎡")
        logger.info(f"대형나무 수관면적: {large_canopy:.1f}㎡")
        logger.info(f"수관면적 적정성: {'✓ 양호' if canopy_check else '✗ 과소'}")
        
        # 3. 총 편익 분포 검증
        median_benefit = df['total_annual_value_krw'].median()
        q75_benefit = df['total_annual_value_krw'].quantile(0.75)
        
        benefit_check = median_benefit > 5000  # 중위값이 5천원 이상
        validation_results['benefit_distribution'] = benefit_check
        
        logger.info(f"편익 중위값: {median_benefit:,.0f}원/년")
        logger.info(f"편익 75분위: {q75_benefit:,.0f}원/년")
        logger.info(f"편익 분포 적정성: {'✓ 양호' if benefit_check else '✗ 부족'}")
        
        return validation_results

# 사용 예시
if __name__ == "__main__":
    calculator = SeoulTreeBenefitsCalculator()
    
    # GeoJSON 파일 경로 (기존 ETL 결과물)
    geojson_path = "seoul_trees_output/trees_clean.geojson"
    
    # 편익 계산 실행
    benefits_df = calculator.process_geojson(
        geojson_path=geojson_path,
        output_path="seoul_trees_output/tree_benefits_fixed.csv"
    )
    
    # 계산 결과 검증
    validation_results = calculator.validate_calculation_results(benefits_df)
    
    # 편익 정보가 포함된 GeoJSON 생성
    calculator.create_benefits_geojson(
        original_geojson_path=geojson_path,
        benefits_df=benefits_df,
        output_path="seoul_trees_output/trees_with_benefits_fixed.geojson"
    )
    
    print("\n🌳 서울시 나무 생태적 편익 계산 완료 (수정버전)!")
    print("개선사항:")
    print("✓ 탄소 편익 과대 계산 문제 해결 (계수 1/10~1/20 수준 조정)")
    print("✓ 에너지 계산 함수 중복 코드 제거")
    print("✓ 수관면적 추정 공식 개선 (현실적 크기 반영)")
    print("✓ 종별 보정계수 이중 반영 방지 (공기정화만 적용)")
    print("✓ 계산 결과 검증 기능 추가")
    print("\n생성된 파일:")
    print("- tree_benefits_fixed.csv: 수정된 상세 편익 데이터")
    print("- trees_with_benefits_fixed.geojson: 수정된 편익 정보가 포함된 지도 데이터")
    print("\n다음 단계:")
    print("1. 시각화 대시보드 생성")
    print("2. 구별/수종별 편익 분석")
    print("3. 정책 제안서 작성")