#!/usr/bin/env python3
"""
District Street Tree Extractor
Extracts and filters street trees by district from Seoul tree dataset.

Usage:
    python extract_district_trees.py

    Or import as module:
    from extract_district_trees import extract_street_trees
    df = extract_street_trees("강남구")
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SeoulBounds:
    """Seoul geographic coordinate bounds (WGS84)."""
    lon_min: float = 126.7
    lon_max: float = 127.3
    lat_min: float = 37.4
    lat_max: float = 37.7


@dataclass
class FilterStats:
    """Statistics from filtering operations."""
    original_count: int = 0
    after_district: int = 0
    after_tree_type: int = 0
    after_canopy: int = 0
    after_coords: int = 0
    final_count: int = 0

    def summary(self) -> dict:
        """Return summary as dictionary."""
        return {
            'original': self.original_count,
            'after_district_filter': self.after_district,
            'after_tree_type_filter': self.after_tree_type,
            'after_canopy_filter': self.after_canopy,
            'after_coord_validation': self.after_coords,
            'final': self.final_count,
            'removed_total': self.original_count - self.final_count,
            'retention_rate': f"{self.final_count / self.original_count * 100:.2f}%" if self.original_count > 0 else "0%"
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_tree_data(filepath: Path) -> pd.DataFrame:
    """
    Load tree data from GeoJSON file into DataFrame.

    Args:
        filepath: Path to GeoJSON file

    Returns:
        DataFrame with tree data including coordinates

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    logger.info(f"Loading data from: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    records = []
    for feature in geojson['features']:
        coords = feature['geometry']['coordinates']
        record = {
            'longitude': coords[0],
            'latitude': coords[1],
            **feature['properties']
        }
        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} total records")

    return df


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def filter_by_district(
    df: pd.DataFrame,
    district: str,
    district_column: str = 'borough'
) -> pd.DataFrame:
    """
    Filter DataFrame for specific district.

    Args:
        df: Input DataFrame
        district: District name (e.g., '강남구')
        district_column: Column containing district info

    Returns:
        Filtered DataFrame
    """
    before = len(df)
    filtered = df[df[district_column] == district].copy()
    after = len(filtered)

    logger.info(f"District filter [{district}]: {before:,} → {after:,} ({after:,} retained)")

    return filtered


def filter_by_tree_type(
    df: pd.DataFrame,
    tree_type: str = 'roadside',
    type_column: str = 'tree_type'
) -> pd.DataFrame:
    """
    Filter DataFrame for specific tree type.

    Args:
        df: Input DataFrame
        tree_type: Type of tree ('roadside', 'park', 'protected')
        type_column: Column containing tree type info

    Returns:
        Filtered DataFrame
    """
    before = len(df)
    filtered = df[df[type_column] == tree_type].copy()
    after = len(filtered)

    logger.info(f"Tree type filter [{tree_type}]: {before:,} → {after:,} ({after:,} retained)")

    return filtered


def filter_missing_canopy(
    df: pd.DataFrame,
    canopy_column: str = 'canopy_width_m',
    min_canopy: float = 0.0
) -> pd.DataFrame:
    """
    Remove records with missing or invalid canopy width.

    Args:
        df: Input DataFrame
        canopy_column: Column containing canopy width
        min_canopy: Minimum valid canopy width (default 0)

    Returns:
        Filtered DataFrame
    """
    before = len(df)
    filtered = df[
        (df[canopy_column].notna()) &
        (df[canopy_column] > min_canopy)
    ].copy()
    after = len(filtered)

    removed = before - after
    logger.info(f"Canopy filter (> {min_canopy}m): {before:,} → {after:,} (removed {removed:,})")

    return filtered


def validate_coordinates(
    df: pd.DataFrame,
    bounds: Optional[SeoulBounds] = None,
    lon_column: str = 'longitude',
    lat_column: str = 'latitude'
) -> pd.DataFrame:
    """
    Validate coordinates are within bounds and not missing.

    Args:
        df: Input DataFrame
        bounds: Coordinate bounds (default: Seoul bounds)
        lon_column: Longitude column name
        lat_column: Latitude column name

    Returns:
        Filtered DataFrame with valid coordinates
    """
    if bounds is None:
        bounds = SeoulBounds()

    before = len(df)

    # Check for missing coordinates
    has_coords = (
        df[lon_column].notna() &
        df[lat_column].notna()
    )

    # Check bounds
    in_bounds = (
        (df[lon_column] >= bounds.lon_min) &
        (df[lon_column] <= bounds.lon_max) &
        (df[lat_column] >= bounds.lat_min) &
        (df[lat_column] <= bounds.lat_max)
    )

    filtered = df[has_coords & in_bounds].copy()
    after = len(filtered)

    removed = before - after
    logger.info(f"Coordinate validation: {before:,} → {after:,} (removed {removed:,})")

    return filtered


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_street_trees(
    district: str,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    tree_type: str = 'roadside',
    min_canopy: float = 0.0,
    save_csv: bool = True
) -> tuple[pd.DataFrame, FilterStats]:
    """
    Extract street trees for a specific district.

    Args:
        district: District name (e.g., '강남구', '서초구')
        input_path: Path to input GeoJSON (default: trees_cleaned.geojson)
        output_path: Path for output CSV (default: auto-generated)
        tree_type: Tree type to filter ('roadside', 'park', 'protected')
        min_canopy: Minimum canopy width to include
        save_csv: Whether to save results to CSV

    Returns:
        Tuple of (filtered DataFrame, FilterStats)

    Example:
        >>> df, stats = extract_street_trees("강남구")
        >>> print(f"Extracted {len(df)} trees")
        >>> print(stats.summary())
    """
    # Set default paths
    if input_path is None:
        input_path = Path("seoul_trees_output/trees_cleaned.geojson")

    if output_path is None:
        safe_district = district.replace(' ', '_')
        output_path = Path(f"seoul_trees_output/{safe_district}_{tree_type}_trees.csv")

    logger.info("=" * 50)
    logger.info(f"Extracting {tree_type} trees for {district}")
    logger.info("=" * 50)

    # Initialize stats
    stats = FilterStats()

    # Load data
    df = load_tree_data(input_path)
    stats.original_count = len(df)

    # Apply filters sequentially
    logger.info("-" * 50)

    # 1. Filter by district
    df = filter_by_district(df, district)
    stats.after_district = len(df)

    if len(df) == 0:
        logger.warning(f"No trees found for district: {district}")
        return df, stats

    # 2. Filter by tree type
    df = filter_by_tree_type(df, tree_type)
    stats.after_tree_type = len(df)

    if len(df) == 0:
        logger.warning(f"No {tree_type} trees found in {district}")
        return df, stats

    # 3. Filter missing canopy
    df = filter_missing_canopy(df, min_canopy=min_canopy)
    stats.after_canopy = len(df)

    # 4. Validate coordinates
    df = validate_coordinates(df)
    stats.after_coords = len(df)

    stats.final_count = len(df)

    # Log summary statistics
    logger.info("-" * 50)
    logger.info("EXTRACTION SUMMARY")
    logger.info("-" * 50)
    logger.info(f"District: {district}")
    logger.info(f"Tree type: {tree_type}")
    logger.info(f"Final count: {stats.final_count:,}")

    if len(df) > 0:
        logger.info(f"Canopy width: {df['canopy_width_m'].min():.1f}m ~ {df['canopy_width_m'].max():.1f}m (mean: {df['canopy_width_m'].mean():.2f}m)")
        logger.info(f"DBH: {df['dbh_cm'].min():.1f}cm ~ {df['dbh_cm'].max():.1f}cm (mean: {df['dbh_cm'].mean():.2f}cm)")

        # Top species
        top_species = df['species_kr'].value_counts().head(5)
        logger.info("Top 5 species:")
        for species, count in top_species.items():
            logger.info(f"  - {species}: {count:,}")

    # Save to CSV
    if save_csv and len(df) > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved to: {output_path}")

    return df, stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_available_districts(input_path: Optional[Path] = None) -> list[str]:
    """
    List all available districts in the dataset.

    Args:
        input_path: Path to input GeoJSON

    Returns:
        List of district names sorted alphabetically
    """
    if input_path is None:
        input_path = Path("seoul_trees_output/trees_cleaned.geojson")

    df = load_tree_data(input_path)
    districts = sorted(df['borough'].unique())

    logger.info(f"Available districts ({len(districts)}):")
    for d in districts:
        count = len(df[df['borough'] == d])
        logger.info(f"  - {d}: {count:,} trees")

    return districts


def extract_multiple_districts(
    districts: list[str],
    **kwargs
) -> dict[str, tuple[pd.DataFrame, FilterStats]]:
    """
    Extract street trees for multiple districts.

    Args:
        districts: List of district names
        **kwargs: Additional arguments passed to extract_street_trees

    Returns:
        Dictionary mapping district name to (DataFrame, FilterStats)
    """
    results = {}

    for district in districts:
        df, stats = extract_street_trees(district, **kwargs)
        results[district] = (df, stats)

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Extract Gangnam-gu roadside trees
    df, stats = extract_street_trees(
        district="강남구",
        tree_type="roadside",
        min_canopy=0.0,
        save_csv=True
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Total trees extracted: {len(df):,}")
    print(f"\nFilter statistics:")
    for key, value in stats.summary().items():
        print(f"  {key}: {value}")

    if len(df) > 0:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
