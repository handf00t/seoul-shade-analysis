#!/usr/bin/env python3
"""
서울시 나무 데이터에서 수종 목록만 추출하는 스크립트
"""

import requests
import xml.etree.ElementTree as ET
import os
import time
from collections import Counter

API_KEY = os.getenv('SEOUL_API_KEY')
if not API_KEY:
    print("❌ 환경변수 SEOUL_API_KEY를 설정해주세요.")
    exit(1)

# 서울시 25개 구
DISTRICTS = [
    '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구',
    '성북구', '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구',
    '양천구', '강서구', '구로구', '금천구', '영등포구', '동작구', '관악구',
    '서초구', '강남구', '송파구', '강동구'
]

def safe_get_text(element, tag_name, default=""):
    """XML에서 안전하게 텍스트 추출"""
    elem = element.find(tag_name)
    if elem is not None and elem.text is not None:
        return elem.text.strip()
    return default

def fetch_xml(url):
    """API에서 XML 가져오기"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return ET.fromstring(response.content)
    except Exception as e:
        print(f"❌ API 요청 실패: {e}")
        return None

def extract_species_from_protected():
    """보호수 수종 추출"""
    print("📦 보호수 수종 추출 중...")
    species_set = set()

    url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoNurseTreeOldTreeWGS/1/1000"
    root = fetch_xml(url)

    if root:
        for row in root.findall('.//row'):
            species = safe_get_text(row, 'TRE_SOM', "미상")
            if species and species != "미상":
                species_set.add(species)

    print(f"  ✅ 보호수: {len(species_set)}개 수종")
    return species_set

def extract_species_from_roadside():
    """가로수 수종 추출 (전체 구)"""
    print("📦 가로수 수종 추출 중...")
    species_set = set()

    for district in DISTRICTS:
        url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoOfRoadsideTreeW/1/1000/{district}"
        root = fetch_xml(url)

        if root:
            for row in root.findall('.//row'):
                species = safe_get_text(row, 'WDPT_NM', "미상")
                if species and species != "미상":
                    species_set.add(species)

        time.sleep(0.2)  # API 부하 방지
        print(f"  {district}: {len(species_set)}개 수종 (누적)")

    print(f"  ✅ 가로수: {len(species_set)}개 수종")
    return species_set

def extract_species_from_park():
    """공원수목 수종 추출"""
    print("📦 공원수목 수종 추출 중...")
    species_set = set()

    url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoParkAndPrivateLandWGS/1/1000"
    root = fetch_xml(url)

    if root:
        for row in root.findall('.//row'):
            species = safe_get_text(row, 'WDPT_NM', "미상")
            if species and species != "미상":
                species_set.add(species)

    print(f"  ✅ 공원수목: {len(species_set)}개 수종")
    return species_set

def count_species_occurrences():
    """각 수종의 출현 빈도 계산"""
    print("\n📊 수종별 개체 수 집계 중...")
    species_counter = Counter()

    # 보호수
    url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoNurseTreeOldTreeWGS/1/1000"
    root = fetch_xml(url)
    if root:
        for row in root.findall('.//row'):
            species = safe_get_text(row, 'TRE_SOM', "미상")
            if species and species != "미상":
                species_counter[species] += 1

    # 가로수
    for district in DISTRICTS:
        url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoOfRoadsideTreeW/1/1000/{district}"
        root = fetch_xml(url)
        if root:
            for row in root.findall('.//row'):
                species = safe_get_text(row, 'WDPT_NM', "미상")
                if species and species != "미상":
                    species_counter[species] += 1
        time.sleep(0.2)

    # 공원수목
    url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoParkAndPrivateLandWGS/1/1000"
    root = fetch_xml(url)
    if root:
        for row in root.findall('.//row'):
            species = safe_get_text(row, 'WDPT_NM', "미상")
            if species and species != "미상":
                species_counter[species] += 1

    return species_counter

if __name__ == "__main__":
    print("🌳 서울시 나무 수종 목록 추출 시작\n")

    # 각 소스별 수종 추출
    protected_species = extract_species_from_protected()
    roadside_species = extract_species_from_roadside()
    park_species = extract_species_from_park()

    # 전체 수종 통합
    all_species = protected_species | roadside_species | park_species

    print(f"\n{'='*60}")
    print(f"전체 고유 수종 개수: {len(all_species)}개")
    print(f"{'='*60}")

    # 알파벳순 정렬
    sorted_species = sorted(all_species)

    print("\n📋 전체 수종 목록 (가나다순):")
    print("-" * 60)
    for i, species in enumerate(sorted_species, 1):
        print(f"{i:3d}. {species}")

    # 빈도수 계산
    species_counts = count_species_occurrences()

    print(f"\n📊 상위 20개 수종 (개체 수 기준):")
    print("-" * 60)
    for i, (species, count) in enumerate(species_counts.most_common(20), 1):
        print(f"{i:2d}. {species:20s} : {count:,}개")

    # 파일로 저장
    with open('/Users/ashleyson/Downloads/Archive/species_list.txt', 'w', encoding='utf-8') as f:
        f.write("=== 서울시 나무 전체 수종 목록 ===\n\n")
        f.write(f"총 {len(all_species)}개 수종\n\n")
        f.write("수종 목록 (가나다순):\n")
        for species in sorted_species:
            f.write(f"- {species}\n")

        f.write("\n\n상위 20개 수종 (개체 수 기준):\n")
        for i, (species, count) in enumerate(species_counts.most_common(20), 1):
            f.write(f"{i:2d}. {species:20s} : {count:,}개\n")

    print(f"\n✅ 결과가 /Users/ashleyson/Downloads/Archive/species_list.txt 에 저장되었습니다!")
