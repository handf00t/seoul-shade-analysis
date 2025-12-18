#!/usr/bin/env python3
"""
서울시 나무 데이터에서 수종 목록 추출 (올바른 필드명 사용)
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
        root = ET.fromstring(response.content)

        # 에러 체크
        result_code = root.find('.//CODE')
        if result_code is not None and result_code.text not in ['INFO-000']:
            return None

        return root
    except Exception as e:
        return None

def extract_species_from_protected():
    """보호수 수종 추출 (올바른 필드명: TRSPC_KORN)"""
    print("📦 보호수 수종 추출 중...")
    species_counter = Counter()

    # 전체 개수 확인
    count_url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoNurseTreeOldTreeWGS/1/1"
    count_root = fetch_xml(count_url)

    if not count_root:
        return species_counter

    total_count_elem = count_root.find('.//list_total_count')
    total_count = int(total_count_elem.text) if total_count_elem is not None else 1000

    print(f"  전체 보호수: {total_count}건")

    # 배치로 수집
    batch_size = 1000
    for start_idx in range(1, total_count + 1, batch_size):
        end_idx = min(start_idx + batch_size - 1, total_count)
        url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoNurseTreeOldTreeWGS/{start_idx}/{end_idx}"

        root = fetch_xml(url)
        if root:
            for row in root.findall('.//row'):
                species = safe_get_text(row, 'TRSPC_KORN', "")  # 올바른 필드명!
                if species and species != "미상":
                    species_counter[species] += 1

        time.sleep(0.3)
        if start_idx % 1000 == 1:
            print(f"  처리 중: {start_idx}/{total_count}")

    print(f"  ✅ 보호수: {len(species_counter)}개 수종, {sum(species_counter.values())}개체")
    return species_counter

def extract_species_from_park():
    """공원수목 수종 추출 (올바른 필드명: TREE_NM)"""
    print("📦 공원수목 수종 추출 중...")
    species_counter = Counter()

    # 전체 개수 확인
    count_url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoParkAndPrivateLandWGS/1/1"
    count_root = fetch_xml(count_url)

    if not count_root:
        return species_counter

    total_count_elem = count_root.find('.//list_total_count')
    total_count = int(total_count_elem.text) if total_count_elem is not None else 1000

    print(f"  전체 공원수목: {total_count}건")

    # 배치로 수집 (전체 데이터가 많으므로 샘플링)
    batch_size = 1000
    max_samples = min(total_count, 10000)  # 최대 10000건만 샘플링

    for start_idx in range(1, max_samples + 1, batch_size):
        end_idx = min(start_idx + batch_size - 1, max_samples)
        url = f"http://openAPI.seoul.go.kr:8088/{API_KEY}/xml/GeoInfoParkAndPrivateLandWGS/{start_idx}/{end_idx}"

        root = fetch_xml(url)
        if root:
            for row in root.findall('.//row'):
                species = safe_get_text(row, 'TREE_NM', "")  # 올바른 필드명!
                if species and species != "미상":
                    species_counter[species] += 1

        time.sleep(0.3)
        if start_idx % 1000 == 1:
            print(f"  처리 중: {start_idx}/{max_samples}")

    print(f"  ✅ 공원수목: {len(species_counter)}개 수종, {sum(species_counter.values())}개체 (샘플링)")
    return species_counter

if __name__ == "__main__":
    print("🌳 서울시 나무 수종 목록 추출 시작\n")

    # 각 소스별 수종 추출
    protected_counter = extract_species_from_protected()
    park_counter = extract_species_from_park()

    # 통합
    all_species_counter = protected_counter + park_counter
    all_species = set(all_species_counter.keys())

    print(f"\n{'='*60}")
    print(f"전체 고유 수종 개수: {len(all_species)}개")
    print(f"전체 개체 수: {sum(all_species_counter.values()):,}개")
    print(f"{'='*60}")

    # 알파벳순 정렬
    sorted_species = sorted(all_species)

    print("\n📋 전체 수종 목록 (가나다순):")
    print("-" * 60)
    for i, species in enumerate(sorted_species, 1):
        count = all_species_counter[species]
        print(f"{i:3d}. {species:20s} : {count:,}개")

    print(f"\n📊 상위 30개 수종 (개체 수 기준):")
    print("-" * 60)
    for i, (species, count) in enumerate(all_species_counter.most_common(30), 1):
        print(f"{i:2d}. {species:20s} : {count:,}개")

    # 파일로 저장
    output_path = '/Users/ashleyson/Downloads/Archive/species_list.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== 서울시 나무 전체 수종 목록 ===\n\n")
        f.write(f"총 {len(all_species)}개 수종\n")
        f.write(f"전체 개체 수: {sum(all_species_counter.values()):,}개\n\n")

        f.write("전체 수종 목록 (가나다순):\n")
        f.write("-" * 60 + "\n")
        for i, species in enumerate(sorted_species, 1):
            count = all_species_counter[species]
            f.write(f"{i:3d}. {species:20s} : {count:,}개\n")

        f.write("\n\n상위 30개 수종 (개체 수 기준):\n")
        f.write("-" * 60 + "\n")
        for i, (species, count) in enumerate(all_species_counter.most_common(30), 1):
            f.write(f"{i:2d}. {species:20s} : {count:,}개\n")

    print(f"\n✅ 결과가 {output_path} 에 저장되었습니다!")
