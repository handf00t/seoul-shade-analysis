#!/usr/bin/env python3
"""
서울시 가로수 데이터 품질 분석 스크립트
Seoul Street Trees Data Quality Analysis

Purpose: Analyze shade connectivity for pedestrian walking routes
- Check data quality (missing values, outliers, coordinate validity)
- Visualize distributions and geographic coverage
- Flag suspicious records
- Generate cleaning recommendations

Author: Data Quality Analysis Script
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data source path (output from seoul_trees_unified.py)
DATA_PATH = Path("seoul_trees_output/trees_unified.geojson")

# Seoul geographical bounds (WGS84)
SEOUL_BOUNDS = {
    'lon_min': 126.7,
    'lon_max': 127.3,
    'lat_min': 37.4,
    'lat_max': 37.7
}

# Outlier thresholds for canopy_width (meters)
CANOPY_WIDTH_MIN = 1.0   # Trees smaller than 1m are suspicious
CANOPY_WIDTH_MAX = 20.0  # Trees larger than 20m are suspicious

# Outlier thresholds for DBH (cm)
DBH_MIN = 5.0    # DBH smaller than 5cm is suspicious for street trees
DBH_MAX = 300.0  # DBH larger than 300cm is suspicious

# Output directory
OUTPUT_DIR = Path("data_quality_reports")


# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_geojson_to_dataframe(filepath: Path) -> pd.DataFrame:
    """
    Load GeoJSON file and convert to pandas DataFrame.

    GeoJSON structure:
    - features[].geometry.coordinates: [longitude, latitude]
    - features[].properties: all other attributes
    """
    print(f"Loading data from: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    records = []
    for feature in geojson['features']:
        # Extract coordinates
        coords = feature['geometry']['coordinates']

        # Combine with properties
        record = {
            'longitude': coords[0],
            'latitude': coords[1],
            **feature['properties']
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} records")

    return df


# ============================================================================
# STEP 2: DATA QUALITY CHECKS
# ============================================================================

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values per column.
    Returns a summary DataFrame with counts and percentages.
    """
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)

    # Calculate missing statistics
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    total_rows = len(df)

    missing_summary = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_pct.values,
        'Valid_Count': total_rows - missing_count.values,
        'Data_Type': df.dtypes.values
    })

    # Sort by missing percentage descending
    missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)
    missing_summary = missing_summary.reset_index(drop=True)

    print(f"\nTotal Records: {total_rows:,}")
    print("\nMissing Values by Column:")
    print("-" * 60)

    for _, row in missing_summary.iterrows():
        status = "OK" if row['Missing_Percent'] == 0 else "MISSING"
        print(f"  {row['Column']:20s}: {row['Missing_Count']:>7,} ({row['Missing_Percent']:>5.1f}%) [{status}]")

    return missing_summary


def check_coordinate_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if coordinates fall within Seoul bounds.
    Flags records outside the expected range.
    """
    print("\n" + "="*60)
    print("COORDINATE VALIDITY CHECK")
    print("="*60)

    # Create validity flags
    df['coord_valid'] = (
        (df['longitude'] >= SEOUL_BOUNDS['lon_min']) &
        (df['longitude'] <= SEOUL_BOUNDS['lon_max']) &
        (df['latitude'] >= SEOUL_BOUNDS['lat_min']) &
        (df['latitude'] <= SEOUL_BOUNDS['lat_max'])
    )

    valid_count = df['coord_valid'].sum()
    invalid_count = (~df['coord_valid']).sum()

    print(f"\nSeoul Bounds: Lon [{SEOUL_BOUNDS['lon_min']}, {SEOUL_BOUNDS['lon_max']}]")
    print(f"              Lat [{SEOUL_BOUNDS['lat_min']}, {SEOUL_BOUNDS['lat_max']}]")
    print(f"\nValid coordinates:   {valid_count:>8,} ({valid_count/len(df)*100:.2f}%)")
    print(f"Invalid coordinates: {invalid_count:>8,} ({invalid_count/len(df)*100:.2f}%)")

    # Show coordinate statistics
    print(f"\nCoordinate Statistics:")
    print(f"  Longitude: min={df['longitude'].min():.4f}, max={df['longitude'].max():.4f}")
    print(f"  Latitude:  min={df['latitude'].min():.4f}, max={df['latitude'].max():.4f}")

    # Show invalid records if any
    if invalid_count > 0:
        print(f"\nSample of invalid coordinate records:")
        invalid_df = df[~df['coord_valid']][['source_id', 'longitude', 'latitude', 'borough']].head(5)
        print(invalid_df.to_string(index=False))

    return df


def check_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers in canopy_width and dbh using specified thresholds.
    Also calculates IQR-based outliers for reference.
    """
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS")
    print("="*60)

    outlier_summary = {}

    # -------------------------------------------------------------------------
    # Canopy Width Analysis
    # -------------------------------------------------------------------------
    print("\n[Canopy Width (canopy_width_m)]")
    print("-" * 40)

    canopy_data = df['canopy_width_m'].dropna()

    if len(canopy_data) > 0:
        # Basic statistics
        stats = {
            'count': len(canopy_data),
            'mean': canopy_data.mean(),
            'std': canopy_data.std(),
            'min': canopy_data.min(),
            'q25': canopy_data.quantile(0.25),
            'median': canopy_data.median(),
            'q75': canopy_data.quantile(0.75),
            'max': canopy_data.max()
        }

        print(f"  Count:  {stats['count']:,}")
        print(f"  Mean:   {stats['mean']:.2f} m")
        print(f"  Std:    {stats['std']:.2f} m")
        print(f"  Min:    {stats['min']:.2f} m")
        print(f"  25%:    {stats['q25']:.2f} m")
        print(f"  Median: {stats['median']:.2f} m")
        print(f"  75%:    {stats['q75']:.2f} m")
        print(f"  Max:    {stats['max']:.2f} m")

        # Threshold-based outliers (for shade analysis context)
        too_small = (df['canopy_width_m'] < CANOPY_WIDTH_MIN).sum()
        too_large = (df['canopy_width_m'] > CANOPY_WIDTH_MAX).sum()

        print(f"\n  Threshold-based outliers:")
        print(f"    < {CANOPY_WIDTH_MIN}m (too small): {too_small:,}")
        print(f"    > {CANOPY_WIDTH_MAX}m (too large): {too_large:,}")

        # Flag suspicious records in DataFrame
        df['canopy_suspicious'] = (
            (df['canopy_width_m'] < CANOPY_WIDTH_MIN) |
            (df['canopy_width_m'] > CANOPY_WIDTH_MAX)
        )

        # IQR-based outliers
        iqr = stats['q75'] - stats['q25']
        lower_bound = stats['q25'] - 1.5 * iqr
        upper_bound = stats['q75'] + 1.5 * iqr
        iqr_outliers = ((canopy_data < lower_bound) | (canopy_data > upper_bound)).sum()

        print(f"\n  IQR-based outliers (reference):")
        print(f"    Lower bound: {lower_bound:.2f} m")
        print(f"    Upper bound: {upper_bound:.2f} m")
        print(f"    Outlier count: {iqr_outliers:,}")

        outlier_summary['canopy_width'] = {
            'stats': stats,
            'threshold_small': too_small,
            'threshold_large': too_large,
            'iqr_outliers': iqr_outliers
        }
    else:
        print("  No canopy width data available!")
        df['canopy_suspicious'] = False

    # -------------------------------------------------------------------------
    # DBH Analysis
    # -------------------------------------------------------------------------
    print("\n[DBH (dbh_cm)]")
    print("-" * 40)

    dbh_data = df['dbh_cm'].dropna()

    if len(dbh_data) > 0:
        stats = {
            'count': len(dbh_data),
            'mean': dbh_data.mean(),
            'std': dbh_data.std(),
            'min': dbh_data.min(),
            'q25': dbh_data.quantile(0.25),
            'median': dbh_data.median(),
            'q75': dbh_data.quantile(0.75),
            'max': dbh_data.max()
        }

        print(f"  Count:  {stats['count']:,}")
        print(f"  Mean:   {stats['mean']:.2f} cm")
        print(f"  Std:    {stats['std']:.2f} cm")
        print(f"  Min:    {stats['min']:.2f} cm")
        print(f"  25%:    {stats['q25']:.2f} cm")
        print(f"  Median: {stats['median']:.2f} cm")
        print(f"  75%:    {stats['q75']:.2f} cm")
        print(f"  Max:    {stats['max']:.2f} cm")

        # Threshold-based outliers
        too_small = (df['dbh_cm'] < DBH_MIN).sum()
        too_large = (df['dbh_cm'] > DBH_MAX).sum()

        print(f"\n  Threshold-based outliers:")
        print(f"    < {DBH_MIN}cm (too small): {too_small:,}")
        print(f"    > {DBH_MAX}cm (too large): {too_large:,}")

        # Flag suspicious records
        df['dbh_suspicious'] = (
            (df['dbh_cm'] < DBH_MIN) |
            (df['dbh_cm'] > DBH_MAX)
        )

        # IQR-based outliers
        iqr = stats['q75'] - stats['q25']
        lower_bound = max(0, stats['q25'] - 1.5 * iqr)
        upper_bound = stats['q75'] + 1.5 * iqr
        iqr_outliers = ((dbh_data < lower_bound) | (dbh_data > upper_bound)).sum()

        print(f"\n  IQR-based outliers (reference):")
        print(f"    Lower bound: {lower_bound:.2f} cm")
        print(f"    Upper bound: {upper_bound:.2f} cm")
        print(f"    Outlier count: {iqr_outliers:,}")

        outlier_summary['dbh'] = {
            'stats': stats,
            'threshold_small': too_small,
            'threshold_large': too_large,
            'iqr_outliers': iqr_outliers
        }
    else:
        print("  No DBH data available!")
        df['dbh_suspicious'] = False

    return outlier_summary


def generate_distribution_summary(df: pd.DataFrame) -> dict:
    """
    Generate distribution summary for key metrics.
    """
    print("\n" + "="*60)
    print("DISTRIBUTION SUMMARY")
    print("="*60)

    summary = {}

    # Tree type distribution
    if 'tree_type' in df.columns:
        print("\n[Tree Type Distribution]")
        type_dist = df['tree_type'].value_counts()
        for tree_type, count in type_dist.items():
            pct = count / len(df) * 100
            print(f"  {tree_type:15s}: {count:>8,} ({pct:>5.1f}%)")
        summary['tree_type'] = type_dist.to_dict()

    # Borough (District) distribution
    if 'borough' in df.columns:
        print("\n[Top 10 Boroughs by Tree Count]")
        borough_dist = df['borough'].value_counts().head(10)
        for borough, count in borough_dist.items():
            pct = count / len(df) * 100
            print(f"  {borough:15s}: {count:>8,} ({pct:>5.1f}%)")
        summary['borough'] = df['borough'].value_counts().to_dict()

    # Species distribution
    if 'species_kr' in df.columns:
        print("\n[Top 10 Species]")
        species_dist = df['species_kr'].value_counts().head(10)
        for species, count in species_dist.items():
            pct = count / len(df) * 100
            print(f"  {species:15s}: {count:>8,} ({pct:>5.1f}%)")
        summary['species'] = df['species_kr'].value_counts().to_dict()

    # Size class distribution (if available)
    if 'size_class' in df.columns:
        print("\n[Size Class Distribution]")
        size_dist = df['size_class'].value_counts()
        for size_class, count in size_dist.items():
            pct = count / len(df) * 100
            print(f"  {size_class:15s}: {count:>8,} ({pct:>5.1f}%)")
        summary['size_class'] = size_dist.to_dict()

    return summary


# ============================================================================
# STEP 3: VISUALIZATIONS
# ============================================================================

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create all required visualizations:
    1. Histogram of canopy width
    2. Geographic distribution scatter plot
    3. Missing data heatmap
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: Canopy Width Histogram
    # -------------------------------------------------------------------------
    print("\n[1] Creating canopy width histogram...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    canopy_data = df['canopy_width_m'].dropna()

    if len(canopy_data) > 0:
        # Left: Full distribution
        ax1 = axes[0]
        ax1.hist(canopy_data, bins=50, edgecolor='white', alpha=0.7, color='forestgreen')
        ax1.axvline(x=CANOPY_WIDTH_MIN, color='red', linestyle='--', linewidth=2,
                    label=f'Min threshold ({CANOPY_WIDTH_MIN}m)')
        ax1.axvline(x=CANOPY_WIDTH_MAX, color='red', linestyle='--', linewidth=2,
                    label=f'Max threshold ({CANOPY_WIDTH_MAX}m)')
        ax1.axvline(x=canopy_data.median(), color='orange', linestyle='-', linewidth=2,
                    label=f'Median ({canopy_data.median():.1f}m)')
        ax1.set_xlabel('Canopy Width (m)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Canopy Width Distribution (Full Range)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')

        # Right: Zoomed view (1-20m range)
        ax2 = axes[1]
        valid_range = canopy_data[(canopy_data >= CANOPY_WIDTH_MIN) & (canopy_data <= CANOPY_WIDTH_MAX)]
        ax2.hist(valid_range, bins=40, edgecolor='white', alpha=0.7, color='forestgreen')
        ax2.axvline(x=valid_range.median(), color='orange', linestyle='-', linewidth=2,
                    label=f'Median ({valid_range.median():.1f}m)')
        ax2.set_xlabel('Canopy Width (m)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Canopy Width Distribution ({CANOPY_WIDTH_MIN}-{CANOPY_WIDTH_MAX}m)',
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
    else:
        axes[0].text(0.5, 0.5, 'No canopy width data', ha='center', va='center', fontsize=14)
        axes[1].text(0.5, 0.5, 'No canopy width data', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'canopy_width_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'canopy_width_histogram.png'}")

    # -------------------------------------------------------------------------
    # Figure 2: Geographic Distribution Scatter Plot
    # -------------------------------------------------------------------------
    print("\n[2] Creating geographic distribution scatter plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Sample data for better visualization (if dataset is large)
    plot_df = df if len(df) <= 50000 else df.sample(n=50000, random_state=42)

    # Color by tree type if available
    if 'tree_type' in plot_df.columns:
        tree_types = plot_df['tree_type'].unique()
        colors = {'protected': 'red', 'roadside': 'green', 'park': 'blue'}

        for tree_type in tree_types:
            subset = plot_df[plot_df['tree_type'] == tree_type]
            ax.scatter(subset['longitude'], subset['latitude'],
                      c=colors.get(tree_type, 'gray'),
                      alpha=0.3, s=5, label=f'{tree_type} ({len(subset):,})')
    else:
        ax.scatter(plot_df['longitude'], plot_df['latitude'],
                  alpha=0.3, s=5, c='forestgreen')

    # Add Seoul bounds rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((SEOUL_BOUNDS['lon_min'], SEOUL_BOUNDS['lat_min']),
                     SEOUL_BOUNDS['lon_max'] - SEOUL_BOUNDS['lon_min'],
                     SEOUL_BOUNDS['lat_max'] - SEOUL_BOUNDS['lat_min'],
                     linewidth=2, edgecolor='black', facecolor='none',
                     linestyle='--', label='Seoul Bounds')
    ax.add_patch(rect)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Geographic Distribution of Street Trees in Seoul', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_dir / 'geographic_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'geographic_distribution.png'}")

    # -------------------------------------------------------------------------
    # Figure 3: Missing Data Heatmap
    # -------------------------------------------------------------------------
    print("\n[3] Creating missing data heatmap...")

    # Select key columns for heatmap
    key_columns = ['species_kr', 'dbh_cm', 'height_m', 'canopy_width_m',
                   'borough', 'district', 'longitude', 'latitude']
    available_columns = [col for col in key_columns if col in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart of missing values
    ax1 = axes[0]
    missing_pct = df[available_columns].isnull().mean() * 100
    colors = ['red' if pct > 50 else 'orange' if pct > 10 else 'green' for pct in missing_pct]
    bars = ax1.barh(available_columns, missing_pct, color=colors, edgecolor='white')
    ax1.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
    ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax1.legend(loc='lower right')

    # Add percentage labels
    for bar, pct in zip(bars, missing_pct):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)

    # Right: Heatmap matrix (sample for large datasets)
    ax2 = axes[1]
    sample_size = min(1000, len(df))
    sample_df = df[available_columns].sample(n=sample_size, random_state=42)

    # Create binary missing matrix
    missing_matrix = sample_df.isnull().astype(int)

    sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing (1) / Present (0)'},
                ax=ax2, yticklabels=available_columns)
    ax2.set_xlabel(f'Sample Records (n={sample_size})', fontsize=12)
    ax2.set_title('Missing Data Pattern (Sample)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([])

    plt.tight_layout()
    fig.savefig(output_dir / 'missing_data_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'missing_data_heatmap.png'}")

    # -------------------------------------------------------------------------
    # Figure 4 (Bonus): DBH vs Canopy Width Relationship
    # -------------------------------------------------------------------------
    print("\n[4] Creating DBH vs Canopy Width scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to records with both values
    plot_data = df.dropna(subset=['dbh_cm', 'canopy_width_m'])

    if len(plot_data) > 0:
        # Sample for visualization if needed
        if len(plot_data) > 10000:
            plot_data = plot_data.sample(n=10000, random_state=42)

        scatter = ax.scatter(plot_data['dbh_cm'], plot_data['canopy_width_m'],
                            alpha=0.3, s=10, c='forestgreen')

        # Add threshold lines
        ax.axhline(y=CANOPY_WIDTH_MIN, color='red', linestyle='--', alpha=0.7,
                   label=f'Canopy min ({CANOPY_WIDTH_MIN}m)')
        ax.axhline(y=CANOPY_WIDTH_MAX, color='red', linestyle='--', alpha=0.7,
                   label=f'Canopy max ({CANOPY_WIDTH_MAX}m)')
        ax.axvline(x=DBH_MIN, color='blue', linestyle='--', alpha=0.7,
                   label=f'DBH min ({DBH_MIN}cm)')

        # Add trend line
        z = np.polyfit(plot_data['dbh_cm'], plot_data['canopy_width_m'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data['dbh_cm'].min(), plot_data['dbh_cm'].max(), 100)
        ax.plot(x_line, p(x_line), 'orange', linewidth=2, label='Trend line')

        ax.set_xlabel('DBH (cm)', fontsize=12)
        ax.set_ylabel('Canopy Width (m)', fontsize=12)
        ax.set_title('DBH vs Canopy Width Relationship', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No data with both DBH and canopy width',
                ha='center', va='center', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'dbh_vs_canopy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'dbh_vs_canopy.png'}")


# ============================================================================
# STEP 4: FLAG SUSPICIOUS RECORDS
# ============================================================================

def flag_suspicious_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag all suspicious records and create a summary.
    """
    print("\n" + "="*60)
    print("SUSPICIOUS RECORDS SUMMARY")
    print("="*60)

    # Initialize flags if not already present
    if 'canopy_suspicious' not in df.columns:
        df['canopy_suspicious'] = (
            (df['canopy_width_m'] < CANOPY_WIDTH_MIN) |
            (df['canopy_width_m'] > CANOPY_WIDTH_MAX)
        )

    if 'dbh_suspicious' not in df.columns:
        df['dbh_suspicious'] = (
            (df['dbh_cm'] < DBH_MIN) |
            (df['dbh_cm'] > DBH_MAX)
        )

    if 'coord_valid' not in df.columns:
        df['coord_valid'] = (
            (df['longitude'] >= SEOUL_BOUNDS['lon_min']) &
            (df['longitude'] <= SEOUL_BOUNDS['lon_max']) &
            (df['latitude'] >= SEOUL_BOUNDS['lat_min']) &
            (df['latitude'] <= SEOUL_BOUNDS['lat_max'])
        )

    # Create overall suspicious flag
    df['is_suspicious'] = (
        df['canopy_suspicious'].fillna(False) |
        df['dbh_suspicious'].fillna(False) |
        (~df['coord_valid'])
    )

    suspicious_count = df['is_suspicious'].sum()
    clean_count = len(df) - suspicious_count

    print(f"\nTotal records:      {len(df):>10,}")
    print(f"Suspicious records: {suspicious_count:>10,} ({suspicious_count/len(df)*100:.2f}%)")
    print(f"Clean records:      {clean_count:>10,} ({clean_count/len(df)*100:.2f}%)")

    print("\nBreakdown of suspicious flags:")
    print(f"  - Invalid coordinates:      {(~df['coord_valid']).sum():>8,}")
    print(f"  - Suspicious canopy width:  {df['canopy_suspicious'].sum():>8,}")
    print(f"  - Suspicious DBH:           {df['dbh_suspicious'].sum():>8,}")

    # Show sample suspicious records
    suspicious_df = df[df['is_suspicious']].head(10)
    if len(suspicious_df) > 0:
        print("\nSample of suspicious records:")
        print(suspicious_df[['source_id', 'canopy_width_m', 'dbh_cm', 'longitude', 'latitude',
                             'canopy_suspicious', 'dbh_suspicious', 'coord_valid']].to_string(index=False))

    return df


# ============================================================================
# STEP 5: GENERATE SUMMARY REPORT
# ============================================================================

def generate_summary_report(df: pd.DataFrame, missing_summary: pd.DataFrame,
                           outlier_summary: dict, output_dir: Path):
    """
    Generate a comprehensive summary report with cleaning recommendations.
    """
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)

    report_path = output_dir / 'data_quality_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("서울시 가로수 데이터 품질 분석 보고서\n")
        f.write("Seoul Street Trees Data Quality Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {DATA_PATH}\n")
        f.write("\n")

        # 1. Overview
        f.write("-" * 80 + "\n")
        f.write("1. DATA OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        f.write(f"Column Names: {', '.join(df.columns)}\n\n")

        # Tree type breakdown
        if 'tree_type' in df.columns:
            f.write("Tree Type Distribution:\n")
            for tree_type, count in df['tree_type'].value_counts().items():
                f.write(f"  - {tree_type}: {count:,} ({count/len(df)*100:.1f}%)\n")
            f.write("\n")

        # 2. Missing Values
        f.write("-" * 80 + "\n")
        f.write("2. MISSING VALUES ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for _, row in missing_summary.iterrows():
            f.write(f"  {row['Column']:20s}: {row['Missing_Count']:>7,} missing ({row['Missing_Percent']:>5.1f}%)\n")
        f.write("\n")

        # 3. Outlier Summary
        f.write("-" * 80 + "\n")
        f.write("3. OUTLIER ANALYSIS\n")
        f.write("-" * 80 + "\n")

        if 'canopy_width' in outlier_summary:
            cw = outlier_summary['canopy_width']
            f.write("\nCanopy Width (canopy_width_m):\n")
            f.write(f"  Valid Range: {CANOPY_WIDTH_MIN}m - {CANOPY_WIDTH_MAX}m\n")
            f.write(f"  Records with data: {cw['stats']['count']:,}\n")
            f.write(f"  Mean: {cw['stats']['mean']:.2f}m, Median: {cw['stats']['median']:.2f}m\n")
            f.write(f"  Too small (< {CANOPY_WIDTH_MIN}m): {cw['threshold_small']:,}\n")
            f.write(f"  Too large (> {CANOPY_WIDTH_MAX}m): {cw['threshold_large']:,}\n")

        if 'dbh' in outlier_summary:
            dbh = outlier_summary['dbh']
            f.write("\nDBH (dbh_cm):\n")
            f.write(f"  Valid Range: {DBH_MIN}cm - {DBH_MAX}cm\n")
            f.write(f"  Records with data: {dbh['stats']['count']:,}\n")
            f.write(f"  Mean: {dbh['stats']['mean']:.2f}cm, Median: {dbh['stats']['median']:.2f}cm\n")
            f.write(f"  Too small (< {DBH_MIN}cm): {dbh['threshold_small']:,}\n")
            f.write(f"  Too large (> {DBH_MAX}cm): {dbh['threshold_large']:,}\n")
        f.write("\n")

        # 4. Coordinate Validity
        f.write("-" * 80 + "\n")
        f.write("4. COORDINATE VALIDITY\n")
        f.write("-" * 80 + "\n")
        valid_coords = df['coord_valid'].sum()
        invalid_coords = (~df['coord_valid']).sum()
        f.write(f"Seoul Bounds: Lon [{SEOUL_BOUNDS['lon_min']}, {SEOUL_BOUNDS['lon_max']}], ")
        f.write(f"Lat [{SEOUL_BOUNDS['lat_min']}, {SEOUL_BOUNDS['lat_max']}]\n")
        f.write(f"Valid coordinates: {valid_coords:,} ({valid_coords/len(df)*100:.2f}%)\n")
        f.write(f"Invalid coordinates: {invalid_coords:,} ({invalid_coords/len(df)*100:.2f}%)\n\n")

        # 5. Suspicious Records Summary
        f.write("-" * 80 + "\n")
        f.write("5. SUSPICIOUS RECORDS SUMMARY\n")
        f.write("-" * 80 + "\n")
        suspicious_count = df['is_suspicious'].sum()
        f.write(f"Total suspicious records: {suspicious_count:,} ({suspicious_count/len(df)*100:.2f}%)\n")
        f.write(f"  - Invalid coordinates: {invalid_coords:,}\n")
        f.write(f"  - Suspicious canopy width: {df['canopy_suspicious'].sum():,}\n")
        f.write(f"  - Suspicious DBH: {df['dbh_suspicious'].sum():,}\n\n")

        # 6. Data Cleaning Recommendations
        f.write("-" * 80 + "\n")
        f.write("6. DATA CLEANING RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")

        recommendations = []

        # Missing canopy_width recommendation
        canopy_missing = df['canopy_width_m'].isnull().sum()
        canopy_missing_pct = canopy_missing / len(df) * 100
        if canopy_missing_pct > 50:
            recommendations.append(
                f"CRITICAL: {canopy_missing_pct:.1f}% of records missing canopy_width data.\n"
                f"  -> For shade analysis, consider imputing canopy width based on species and DBH.\n"
                f"  -> Use species-specific average canopy width as fallback.\n"
                f"  -> Alternatively, exclude records without canopy data from shade analysis.\n"
            )
        elif canopy_missing_pct > 10:
            recommendations.append(
                f"WARNING: {canopy_missing_pct:.1f}% of records missing canopy_width data.\n"
                f"  -> Consider imputation using DBH-canopy relationship for same species.\n"
            )

        # Coordinate issues
        if invalid_coords > 0:
            recommendations.append(
                f"COORDINATE ISSUES: {invalid_coords:,} records have coordinates outside Seoul bounds.\n"
                f"  -> Remove these records from analysis.\n"
                f"  -> Investigate data source for systematic errors.\n"
            )

        # Canopy width outliers
        if 'canopy_width' in outlier_summary:
            cw = outlier_summary['canopy_width']
            if cw['threshold_small'] > 0 or cw['threshold_large'] > 0:
                recommendations.append(
                    f"CANOPY WIDTH OUTLIERS: {cw['threshold_small'] + cw['threshold_large']:,} records flagged.\n"
                    f"  -> Records < {CANOPY_WIDTH_MIN}m: May be young trees or data errors - verify or exclude.\n"
                    f"  -> Records > {CANOPY_WIDTH_MAX}m: Likely data entry errors - cap at {CANOPY_WIDTH_MAX}m or exclude.\n"
                    f"  -> For shade connectivity analysis, focus on trees with canopy width 3-15m.\n"
                )

        # DBH outliers
        if 'dbh' in outlier_summary:
            dbh = outlier_summary['dbh']
            if dbh['threshold_small'] > 0 or dbh['threshold_large'] > 0:
                recommendations.append(
                    f"DBH OUTLIERS: {dbh['threshold_small'] + dbh['threshold_large']:,} records flagged.\n"
                    f"  -> Records < {DBH_MIN}cm: May be very young trees - consider separate analysis.\n"
                    f"  -> Records > {DBH_MAX}cm: Verify against protected tree records.\n"
                )

        # Missing species
        species_missing = df['species_kr'].isnull().sum() if 'species_kr' in df.columns else 0
        if species_missing > 0:
            recommendations.append(
                f"SPECIES DATA: {species_missing:,} records missing species information.\n"
                f"  -> Species is important for shade coefficient estimation.\n"
                f"  -> Consider neighborhood-based imputation or use generic shade model.\n"
            )

        if not recommendations:
            recommendations.append("Data quality is good! No critical issues found.\n")

        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")

        # 7. Shade Connectivity Analysis Readiness
        f.write("-" * 80 + "\n")
        f.write("7. SHADE CONNECTIVITY ANALYSIS READINESS\n")
        f.write("-" * 80 + "\n")

        # Calculate analysis-ready records
        analysis_ready = df[
            (df['coord_valid'] == True) &
            (df['canopy_width_m'].notna()) &
            (df['canopy_width_m'] >= CANOPY_WIDTH_MIN) &
            (df['canopy_width_m'] <= CANOPY_WIDTH_MAX)
        ]

        f.write(f"\nRecords ready for shade connectivity analysis: {len(analysis_ready):,} ({len(analysis_ready)/len(df)*100:.1f}%)\n")
        f.write(f"Records excluded: {len(df) - len(analysis_ready):,}\n\n")

        f.write("Exclusion reasons:\n")
        f.write(f"  - Invalid coordinates: {(~df['coord_valid']).sum():,}\n")
        f.write(f"  - Missing canopy width: {df['canopy_width_m'].isnull().sum():,}\n")
        f.write(f"  - Canopy width out of range: {((df['canopy_width_m'] < CANOPY_WIDTH_MIN) | (df['canopy_width_m'] > CANOPY_WIDTH_MAX)).sum():,}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport saved to: {report_path}")

    # Also save flagged records for review
    suspicious_records = df[df['is_suspicious']]
    if len(suspicious_records) > 0:
        suspicious_path = output_dir / 'suspicious_records.csv'
        suspicious_records.to_csv(suspicious_path, index=False, encoding='utf-8-sig')
        print(f"Suspicious records saved to: {suspicious_path}")

    return report_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs complete data quality analysis.
    """
    print("\n")
    print("*" * 60)
    print("  서울시 가로수 데이터 품질 분석")
    print("  Seoul Street Trees Data Quality Analysis")
    print("*" * 60)
    print(f"\nData source: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Check if data file exists
    if not DATA_PATH.exists():
        print(f"\nERROR: Data file not found: {DATA_PATH}")
        print("Please run seoul_trees_unified.py first to collect tree data.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    df = load_geojson_to_dataframe(DATA_PATH)

    # Step 2: Data quality checks
    missing_summary = check_missing_values(df)
    df = check_coordinate_validity(df)
    outlier_summary = check_outliers(df)
    distribution_summary = generate_distribution_summary(df)

    # Step 3: Flag suspicious records
    df = flag_suspicious_records(df)

    # Step 4: Create visualizations
    create_visualizations(df, OUTPUT_DIR)

    # Step 5: Generate summary report
    report_path = generate_summary_report(df, missing_summary, outlier_summary, OUTPUT_DIR)

    # Final summary
    print("\n")
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated outputs in {OUTPUT_DIR}/:")
    print("  - canopy_width_histogram.png")
    print("  - geographic_distribution.png")
    print("  - missing_data_heatmap.png")
    print("  - dbh_vs_canopy.png")
    print("  - data_quality_report.txt")
    print("  - suspicious_records.csv (if suspicious records exist)")

    # Quick summary for shade connectivity analysis
    analysis_ready = df[
        (df['coord_valid'] == True) &
        (df['canopy_width_m'].notna()) &
        (df['canopy_width_m'] >= CANOPY_WIDTH_MIN) &
        (df['canopy_width_m'] <= CANOPY_WIDTH_MAX)
    ]
    print(f"\n*** SHADE ANALYSIS READINESS ***")
    print(f"Records ready for shade connectivity analysis: {len(analysis_ready):,} / {len(df):,} ({len(analysis_ready)/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
