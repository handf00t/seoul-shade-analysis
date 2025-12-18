#!/usr/bin/env python3
"""
Road Name Extractor for Seoul Street Trees
Extracts road names from Korean addresses for shade connectivity analysis.

Patterns:
- Primary: 로 (ro), 대로 (daero), 길 (gil)
- Fallback: 동 (dong)

Usage:
    python extract_road_names.py

    Or import:
    from extract_road_names import extract_road_name, add_road_names_to_df
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Primary pattern: Road names ending in 로, 대로, 길
# Captures: 테헤란로, 강남대로, 논현로, 봉은사로98길, 강남대로 98길
# Pattern breakdown:
#   [가-힣]+  : One or more Korean characters (road name prefix)
#   (?:대로|로|길) : Road type suffix (대로, 로, or 길)
#   (?:\s*\d+)?   : Optional space + numbers (e.g., " 98" or "98")
#   (?:길)?       : Optional 길 suffix for numbered streets (e.g., "98길")
ROAD_PATTERN = re.compile(
    r'([가-힣]+(?:대로|로|길)(?:\s*\d+)?(?:길)?)'
)

# Alternative pattern for roads with numbers like "강남대로98길"
ROAD_WITH_NUMBER_PATTERN = re.compile(
    r'([가-힣]+(?:대로|로)\s*\d+길)'
)

# Fallback pattern: Dong names (동)
# Captures: 역삼동, 삼성동, 논현동
DONG_PATTERN = re.compile(
    r'([가-힣]+동)(?:\s|$|\d)'
)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_road_name(address: Optional[str]) -> str:
    """
    Extract road name from Korean address string.

    Priority:
    1. Road names with numbers (e.g., 강남대로98길)
    2. Standard road names (e.g., 테헤란로, 강남대로)
    3. Dong names as fallback (e.g., 역삼동)
    4. 'Unknown' if nothing found

    Args:
        address: Korean address string

    Returns:
        Extracted road name or 'Unknown'

    Examples:
        >>> extract_road_name("서울시 강남구 테헤란로 123")
        '테헤란로'
        >>> extract_road_name("강남대로 98길 45-6")
        '강남대로98길'
        >>> extract_road_name("역삼동 456-7")
        '역삼동'
        >>> extract_road_name(None)
        'Unknown'
    """
    # Handle missing/invalid input
    if address is None or pd.isna(address):
        return 'Unknown'

    if not isinstance(address, str):
        address = str(address)

    # Clean up: normalize whitespace
    address = ' '.join(address.split())

    if not address.strip():
        return 'Unknown'

    # Try 1: Roads with numbers (e.g., 강남대로98길, 봉은사로 114길)
    match = ROAD_WITH_NUMBER_PATTERN.search(address)
    if match:
        road_name = match.group(1)
        # Remove internal spaces for consistency
        return road_name.replace(' ', '')

    # Try 2: Standard road names (e.g., 테헤란로, 강남대로, 논현길)
    match = ROAD_PATTERN.search(address)
    if match:
        road_name = match.group(1)
        # Clean up spaces
        return road_name.replace(' ', '')

    # Try 3: Fallback to dong name
    match = DONG_PATTERN.search(address)
    if match:
        return match.group(1)

    return 'Unknown'


def add_road_names_to_df(
    df: pd.DataFrame,
    address_column: str = 'address',
    output_column: str = 'road_name'
) -> pd.DataFrame:
    """
    Add road_name column to DataFrame by extracting from addresses.

    Args:
        df: Input DataFrame
        address_column: Name of column containing addresses
        output_column: Name of new column for road names

    Returns:
        DataFrame with new road_name column

    Example:
        >>> df = add_road_names_to_df(df)
        >>> df['road_name'].value_counts().head()
    """
    logger.info(f"Extracting road names from '{address_column}' column...")

    # Apply extraction
    df = df.copy()
    df[output_column] = df[address_column].apply(extract_road_name)

    # Log statistics
    total = len(df)
    unknown_count = (df[output_column] == 'Unknown').sum()
    extracted_count = total - unknown_count

    logger.info(f"Extraction complete:")
    logger.info(f"  - Total records: {total:,}")
    logger.info(f"  - Extracted: {extracted_count:,} ({extracted_count/total*100:.1f}%)")
    logger.info(f"  - Unknown: {unknown_count:,} ({unknown_count/total*100:.1f}%)")

    # Show top road names
    top_roads = df[output_column].value_counts().head(10)
    logger.info(f"Top 10 road names:")
    for road, count in top_roads.items():
        logger.info(f"    {road}: {count:,}")

    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_road_distribution(
    df: pd.DataFrame,
    road_column: str = 'road_name'
) -> pd.DataFrame:
    """
    Analyze road name distribution with tree counts and canopy stats.

    Args:
        df: DataFrame with road_name column
        road_column: Name of road name column

    Returns:
        DataFrame with road statistics
    """
    logger.info("Analyzing road distribution...")

    # Group by road name
    road_stats = df.groupby(road_column).agg(
        tree_count=('source_id', 'count'),
        avg_canopy=('canopy_width_m', 'mean'),
        total_canopy=('canopy_width_m', 'sum'),
        avg_dbh=('dbh_cm', 'mean')
    ).round(2)

    # Sort by tree count
    road_stats = road_stats.sort_values('tree_count', ascending=False)

    # Add percentage
    road_stats['pct_of_total'] = (road_stats['tree_count'] / road_stats['tree_count'].sum() * 100).round(2)

    logger.info(f"Found {len(road_stats):,} unique road names")

    return road_stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_gangnam_trees(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Process Gangnam street trees and extract road names.

    Args:
        input_path: Path to input CSV
        output_path: Path for output CSV

    Returns:
        Processed DataFrame
    """
    if input_path is None:
        input_path = Path("seoul_trees_output/강남구_roadside_trees.csv")

    if output_path is None:
        output_path = Path("seoul_trees_output/강남구_roadside_trees_with_roads.csv")

    logger.info("=" * 50)
    logger.info("Road Name Extraction for Gangnam Street Trees")
    logger.info("=" * 50)

    # Load data
    logger.info(f"Loading: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    logger.info(f"Loaded {len(df):,} records")

    # Extract road names
    df = add_road_names_to_df(df)

    # Analyze distribution
    logger.info("-" * 50)
    road_stats = analyze_road_distribution(df)

    # Show summary
    logger.info("-" * 50)
    logger.info("ROAD STATISTICS SUMMARY")
    logger.info("-" * 50)

    # Roads with most trees
    logger.info("\nRoads with most trees:")
    for road, row in road_stats.head(10).iterrows():
        logger.info(f"  {road}: {int(row['tree_count']):,} trees, avg canopy {row['avg_canopy']:.1f}m")

    # Unknown analysis
    unknown_df = df[df['road_name'] == 'Unknown']
    if len(unknown_df) > 0:
        logger.info(f"\nSample of 'Unknown' addresses ({len(unknown_df)} total):")
        for addr in unknown_df['address'].head(5):
            logger.info(f"  - {addr}")

    # Save results
    logger.info("-" * 50)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved to: {output_path}")

    # Save road statistics
    stats_path = output_path.parent / "강남구_road_statistics.csv"
    road_stats.to_csv(stats_path, encoding='utf-8-sig')
    logger.info(f"Road stats saved to: {stats_path}")

    return df


# ============================================================================
# TEST CASES
# ============================================================================

def run_tests():
    """Run test cases for road name extraction."""
    test_cases = [
        # (input, expected)
        ("서울시 강남구 테헤란로 123", "테헤란로"),
        ("강남대로 456", "강남대로"),
        ("논현로 85길 12-3", "논현로85길"),
        ("봉은사로 114길", "봉은사로114길"),
        ("강남대로98길 45-6", "강남대로98길"),
        ("역삼동 456-7", "역삼동"),
        ("삼성동 123-45 현대아파트", "삼성동"),
        ("", "Unknown"),
        (None, "Unknown"),
        ("서울시 강남구", "Unknown"),
        ("도곡로 25길 17", "도곡로25길"),
        ("선릉로 93길", "선릉로93길"),
        ("언주로30길 23-4", "언주로30길"),
        ("영동대로 513", "영동대로"),
    ]

    logger.info("=" * 50)
    logger.info("Running test cases...")
    logger.info("=" * 50)

    passed = 0
    failed = 0

    for address, expected in test_cases:
        result = extract_road_name(address)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1
            logger.warning(f"{status}: '{address}' → '{result}' (expected: '{expected}')")

    logger.info(f"\nTest results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    # Run tests first
    print("\n" + "=" * 50)
    print("RUNNING TESTS")
    print("=" * 50)
    run_tests()

    # Process Gangnam data
    print("\n")
    df = process_gangnam_trees()

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)
    print(f"Total records: {len(df):,}")
    print(f"Unique roads: {df['road_name'].nunique()}")
    print(f"Unknown: {(df['road_name'] == 'Unknown').sum():,}")
