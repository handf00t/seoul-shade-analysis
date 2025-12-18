#!/usr/bin/env python3
"""
Shade Walking Network Builder
Creates a graph where trees are connected if they provide continuous shade coverage.

Connection formula:
    connected if: distance < (shade_radius1 + shade_radius2 + max_gap)
    where: shade_radius = (canopy_width / 2) × shade_factor

Optimization:
    - KDTree for O(n log n) spatial indexing
    - Two-stage filtering: fast candidates → exact check
    - Sparse matrix representation

Author: Shade Network Analysis
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# CONFIGURATION
# ============================================================================

@dataclass
class ShadeNetworkConfig:
    """Configuration for shade network construction."""
    shade_factor: float = 0.65       # Effective shade coverage factor
    max_gap: float = 5.0             # Maximum gap between shade zones (meters)
    min_canopy: float = 1.0          # Minimum canopy width to include

    # Quality thresholds
    strong_quality_threshold: float = 0.7
    weak_quality_threshold: float = 0.3

    # Coordinate conversion (Seoul approximate)
    meters_per_degree_lon: float = 88740  # at ~37.5° latitude
    meters_per_degree_lat: float = 111000


# ============================================================================
# COORDINATE UTILITIES
# ============================================================================

def coords_to_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    config: ShadeNetworkConfig
) -> np.ndarray:
    """
    Convert WGS84 coordinates to local meter coordinates.

    Uses simple equirectangular projection centered on data centroid.
    Sufficient accuracy for distances < 50km.

    Args:
        lon: Longitude array
        lat: Latitude array
        config: Network configuration

    Returns:
        Nx2 array of [x_meters, y_meters]
    """
    # Center point
    lon_center = lon.mean()
    lat_center = lat.mean()

    # Convert to meters from center
    x_meters = (lon - lon_center) * config.meters_per_degree_lon
    y_meters = (lat - lat_center) * config.meters_per_degree_lat

    return np.column_stack([x_meters, y_meters])


# ============================================================================
# NETWORK BUILDER
# ============================================================================

class ShadeNetworkBuilder:
    """
    Builds shade connectivity network from tree data.

    Uses two-stage filtering:
    1. KDTree query for fast candidate search
    2. Exact threshold check based on individual canopy widths
    """

    def __init__(self, config: Optional[ShadeNetworkConfig] = None):
        """Initialize builder with configuration."""
        self.config = config or ShadeNetworkConfig()
        self.G: Optional[nx.Graph] = None
        self.stats: dict = {}

    def calculate_shade_radius(self, canopy_width: float) -> float:
        """
        Calculate effective shade radius from canopy width.

        shade_radius = (canopy_width / 2) × shade_factor

        Args:
            canopy_width: Tree canopy width in meters

        Returns:
            Effective shade radius in meters
        """
        return (canopy_width / 2) * self.config.shade_factor

    def build_network(
        self,
        df: pd.DataFrame,
        lon_col: str = 'longitude',
        lat_col: str = 'latitude',
        canopy_col: str = 'canopy_width_m',
        id_col: str = 'source_id'
    ) -> nx.Graph:
        """
        Build shade connectivity network from tree DataFrame.

        Args:
            df: DataFrame with tree data
            lon_col: Longitude column name
            lat_col: Latitude column name
            canopy_col: Canopy width column name
            id_col: Tree ID column name

        Returns:
            NetworkX graph with shade connections
        """
        logger.info("=" * 60)
        logger.info("BUILDING SHADE NETWORK")
        logger.info("=" * 60)

        start_time = time.time()

        # ----- Step 0: Prepare data -----
        logger.info("\n[Step 0] Preparing data...")

        # Filter valid records
        valid_mask = (
            df[canopy_col].notna() &
            (df[canopy_col] >= self.config.min_canopy) &
            df[lon_col].notna() &
            df[lat_col].notna()
        )
        df_valid = df[valid_mask].reset_index(drop=True)
        n_trees = len(df_valid)

        logger.info(f"  Valid trees: {n_trees:,}")

        # Extract arrays
        lons = df_valid[lon_col].values
        lats = df_valid[lat_col].values
        canopy_widths = df_valid[canopy_col].values
        tree_ids = df_valid[id_col].values if id_col in df_valid.columns else np.arange(n_trees)

        # Calculate shade radii for all trees
        shade_radii = np.array([
            self.calculate_shade_radius(cw) for cw in canopy_widths
        ])

        logger.info(f"  Shade radii: min={shade_radii.min():.2f}m, max={shade_radii.max():.2f}m, mean={shade_radii.mean():.2f}m")

        # ----- Step 1: Convert to meters -----
        logger.info("\n[Step 1] Converting coordinates to meters...")

        coords_meters = coords_to_meters(lons, lats, self.config)
        logger.info(f"  Coordinate range: X=[{coords_meters[:,0].min():.0f}, {coords_meters[:,0].max():.0f}]m")
        logger.info(f"                    Y=[{coords_meters[:,1].min():.0f}, {coords_meters[:,1].max():.0f}]m")

        # ----- Step 2: Build KDTree -----
        logger.info("\n[Step 2] Building KDTree spatial index...")

        kdtree_start = time.time()
        kdtree = KDTree(coords_meters)
        kdtree_time = time.time() - kdtree_start

        logger.info(f"  KDTree built in {kdtree_time:.3f}s")

        # Maximum search radius (conservative upper bound)
        max_shade_radius = shade_radii.max()
        max_search_radius = 2 * max_shade_radius + self.config.max_gap
        logger.info(f"  Max search radius: {max_search_radius:.2f}m")

        # ----- Step 3: Build graph with two-stage filtering -----
        logger.info("\n[Step 3] Building graph with two-stage filtering...")

        # Initialize graph
        G = nx.Graph()

        # Add all nodes with attributes
        for i in range(n_trees):
            G.add_node(i,
                tree_id=str(tree_ids[i]),
                longitude=lons[i],
                latitude=lats[i],
                canopy_width=canopy_widths[i],
                shade_radius=shade_radii[i],
                x_meters=coords_meters[i, 0],
                y_meters=coords_meters[i, 1]
            )

        # Add additional columns as node attributes
        extra_cols = ['species_kr', 'dbh_cm', 'road_name', 'borough']
        for col in extra_cols:
            if col in df_valid.columns:
                for i, val in enumerate(df_valid[col].values):
                    G.nodes[i][col] = val

        logger.info(f"  Added {n_trees:,} nodes")

        # Find edges using two-stage filtering
        edge_start = time.time()

        edges_added = 0
        candidates_checked = 0

        # Progress tracking
        progress_interval = max(1, n_trees // 10)

        for i in range(n_trees):
            # Stage 1: Fast KDTree query for candidates
            tree_coord = coords_meters[i]

            # Use maximum possible threshold for this tree
            search_radius = shade_radii[i] + max_shade_radius + self.config.max_gap

            # Query candidates (returns indices)
            candidates = kdtree.query_ball_point(tree_coord, r=search_radius)

            # Stage 2: Exact check based on individual canopy widths
            for j in candidates:
                if j <= i:  # Skip self and already-processed pairs
                    continue

                candidates_checked += 1

                # Calculate exact threshold for this pair
                threshold = shade_radii[i] + shade_radii[j] + self.config.max_gap

                # Calculate actual distance
                actual_distance = np.linalg.norm(coords_meters[i] - coords_meters[j])

                # Check connection
                if actual_distance < threshold:
                    # Calculate edge attributes
                    shade_overlap = threshold - actual_distance
                    connection_quality = min(1.0, shade_overlap / self.config.max_gap)

                    # Add edge
                    G.add_edge(i, j,
                        distance=round(actual_distance, 2),
                        threshold=round(threshold, 2),
                        shade_overlap=round(shade_overlap, 2),
                        connection_quality=round(connection_quality, 3)
                    )
                    edges_added += 1

            # Progress logging
            if (i + 1) % progress_interval == 0:
                pct = (i + 1) / n_trees * 100
                logger.info(f"    Progress: {pct:.0f}% ({i+1:,}/{n_trees:,} trees, {edges_added:,} edges)")

        edge_time = time.time() - edge_start

        logger.info(f"  Edge construction completed in {edge_time:.2f}s")
        logger.info(f"  Candidates checked: {candidates_checked:,}")
        logger.info(f"  Edges added: {edges_added:,}")

        self.G = G

        # ----- Step 4: Calculate statistics -----
        logger.info("\n[Step 4] Calculating network statistics...")

        self.stats = self._calculate_statistics()

        total_time = time.time() - start_time
        self.stats['build_time_seconds'] = round(total_time, 2)

        logger.info(f"\nTotal build time: {total_time:.2f}s")

        return G

    def _calculate_statistics(self) -> dict:
        """Calculate comprehensive network statistics."""
        G = self.G

        if G is None or len(G) == 0:
            return {}

        # Basic stats
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Degree statistics
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees)
        max_degree = max(degrees)
        isolated_nodes = sum(1 for d in degrees if d == 0)

        # Edge quality statistics
        qualities = [G.edges[e]['connection_quality'] for e in G.edges()]
        distances = [G.edges[e]['distance'] for e in G.edges()]
        overlaps = [G.edges[e]['shade_overlap'] for e in G.edges()]

        strong_connections = sum(1 for q in qualities if q >= self.config.strong_quality_threshold)
        weak_connections = sum(1 for q in qualities if q < self.config.weak_quality_threshold)
        medium_connections = n_edges - strong_connections - weak_connections

        # Connected components
        n_components = nx.number_connected_components(G)
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc)

        stats = {
            # Basic
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': round(nx.density(G), 6),

            # Degree
            'avg_degree': round(avg_degree, 2),
            'max_degree': max_degree,
            'isolated_nodes': isolated_nodes,
            'connected_nodes': n_nodes - isolated_nodes,

            # Quality distribution
            'strong_connections': strong_connections,
            'medium_connections': medium_connections,
            'weak_connections': weak_connections,
            'strong_pct': round(strong_connections / n_edges * 100, 1) if n_edges > 0 else 0,
            'weak_pct': round(weak_connections / n_edges * 100, 1) if n_edges > 0 else 0,

            # Edge metrics
            'avg_distance': round(np.mean(distances), 2) if distances else 0,
            'avg_overlap': round(np.mean(overlaps), 2) if overlaps else 0,
            'avg_quality': round(np.mean(qualities), 3) if qualities else 0,

            # Connectivity
            'n_components': n_components,
            'largest_component_size': largest_cc_size,
            'largest_component_pct': round(largest_cc_size / n_nodes * 100, 1),
        }

        return stats

    def print_statistics(self):
        """Print network statistics."""
        if not self.stats:
            logger.warning("No statistics available. Build network first.")
            return

        print("\n" + "=" * 60)
        print("SHADE NETWORK STATISTICS")
        print("=" * 60)

        print("\n[Basic Metrics]")
        print(f"  Nodes (trees):     {self.stats['n_nodes']:,}")
        print(f"  Edges (connections): {self.stats['n_edges']:,}")
        print(f"  Network density:   {self.stats['density']:.6f}")

        print("\n[Degree Statistics]")
        print(f"  Average degree:    {self.stats['avg_degree']:.2f}")
        print(f"  Maximum degree:    {self.stats['max_degree']}")
        print(f"  Isolated nodes:    {self.stats['isolated_nodes']:,}")
        print(f"  Connected nodes:   {self.stats['connected_nodes']:,}")

        print("\n[Connection Quality]")
        print(f"  Strong (≥{self.config.strong_quality_threshold}): {self.stats['strong_connections']:,} ({self.stats['strong_pct']}%)")
        print(f"  Medium:            {self.stats['medium_connections']:,}")
        print(f"  Weak (<{self.config.weak_quality_threshold}):   {self.stats['weak_connections']:,} ({self.stats['weak_pct']}%)")

        print("\n[Edge Metrics]")
        print(f"  Avg distance:      {self.stats['avg_distance']:.2f}m")
        print(f"  Avg shade overlap: {self.stats['avg_overlap']:.2f}m")
        print(f"  Avg quality:       {self.stats['avg_quality']:.3f}")

        print("\n[Connectivity]")
        print(f"  Connected components: {self.stats['n_components']:,}")
        print(f"  Largest component:    {self.stats['largest_component_size']:,} nodes ({self.stats['largest_component_pct']}%)")

        if 'build_time_seconds' in self.stats:
            print(f"\n[Performance]")
            print(f"  Build time: {self.stats['build_time_seconds']:.2f}s")

        print("=" * 60)

    def get_quality_distribution(self) -> pd.DataFrame:
        """Get edge quality distribution as DataFrame."""
        if self.G is None:
            return pd.DataFrame()

        edges_data = []
        for u, v, data in self.G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'distance': data['distance'],
                'threshold': data['threshold'],
                'shade_overlap': data['shade_overlap'],
                'connection_quality': data['connection_quality']
            })

        return pd.DataFrame(edges_data)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_network_sample(
    G: nx.Graph,
    sample_size: int = 500,
    output_path: Optional[Path] = None,
    title: str = "Shade Connectivity Network"
):
    """
    Visualize a sample of the network with edges colored by quality.

    Args:
        G: NetworkX graph
        sample_size: Number of nodes to sample
        output_path: Path to save figure
        title: Plot title
    """
    logger.info("Creating network visualization...")

    if len(G) == 0:
        logger.warning("Empty graph, nothing to visualize")
        return

    # Sample nodes from largest component for better visualization
    largest_cc = max(nx.connected_components(G), key=len)

    if len(largest_cc) > sample_size:
        # Sample nodes
        sample_nodes = list(largest_cc)[:sample_size]
    else:
        sample_nodes = list(largest_cc)

    # Create subgraph
    G_sub = G.subgraph(sample_nodes).copy()

    logger.info(f"  Visualizing {len(G_sub.nodes()):,} nodes, {len(G_sub.edges()):,} edges")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ----- Left: Network with quality-colored edges -----
    ax1 = axes[0]

    # Get positions from coordinates
    pos = {n: (G_sub.nodes[n]['x_meters'], G_sub.nodes[n]['y_meters']) for n in G_sub.nodes()}

    # Color edges by quality
    edge_colors = []
    edge_widths = []

    for u, v in G_sub.edges():
        quality = G_sub.edges[u, v]['connection_quality']
        edge_colors.append(quality)
        edge_widths.append(0.5 + quality * 1.5)

    # Draw edges
    edges = nx.draw_networkx_edges(
        G_sub, pos, ax=ax1,
        edge_color=edge_colors,
        edge_cmap=plt.cm.RdYlGn,
        edge_vmin=0, edge_vmax=1,
        width=edge_widths,
        alpha=0.6
    )

    # Draw nodes
    node_sizes = [G_sub.nodes[n]['canopy_width'] * 5 for n in G_sub.nodes()]
    nx.draw_networkx_nodes(
        G_sub, pos, ax=ax1,
        node_size=node_sizes,
        node_color='forestgreen',
        alpha=0.7
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label('Connection Quality', fontsize=10)

    ax1.set_title(f'{title}\n({len(G_sub.nodes())} trees, {len(G_sub.edges())} connections)', fontsize=12)
    ax1.set_xlabel('X (meters)', fontsize=10)
    ax1.set_ylabel('Y (meters)', fontsize=10)
    ax1.set_aspect('equal')

    # ----- Right: Quality distribution histogram -----
    ax2 = axes[1]

    # Get all qualities from full graph
    all_qualities = [G.edges[e]['connection_quality'] for e in G.edges()]

    # Create histogram
    bins = np.linspace(0, 1, 21)
    counts, bins, patches = ax2.hist(all_qualities, bins=bins, edgecolor='white', alpha=0.7)

    # Color bins by quality
    cmap = plt.cm.RdYlGn
    for i, (count, patch) in enumerate(zip(counts, patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        patch.set_facecolor(cmap(bin_center))

    # Add threshold lines
    ax2.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Weak threshold (0.3)')
    ax2.axvline(x=0.7, color='green', linestyle='--', linewidth=2, label='Strong threshold (0.7)')

    # Calculate percentages
    weak_pct = sum(1 for q in all_qualities if q < 0.3) / len(all_qualities) * 100
    strong_pct = sum(1 for q in all_qualities if q >= 0.7) / len(all_qualities) * 100

    ax2.set_xlabel('Connection Quality', fontsize=12)
    ax2.set_ylabel('Number of Connections', fontsize=12)
    ax2.set_title(f'Connection Quality Distribution\n(Strong: {strong_pct:.1f}%, Weak: {weak_pct:.1f}%)', fontsize=12)
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved to: {output_path}")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution: build shade network for Gangnam street trees."""

    # Configuration
    config = ShadeNetworkConfig(
        shade_factor=0.65,
        max_gap=5.0,
        min_canopy=1.0
    )

    # Input path
    input_path = Path("seoul_trees_output/강남구_roadside_trees_with_roads.csv")

    if not input_path.exists():
        # Try without road names
        input_path = Path("seoul_trees_output/강남구_roadside_trees.csv")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Load data
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    logger.info(f"Loaded {len(df):,} trees")

    # Build network
    builder = ShadeNetworkBuilder(config)
    G = builder.build_network(df)

    # Print statistics
    builder.print_statistics()

    # Visualize
    output_dir = Path("shade_network_output")
    output_dir.mkdir(exist_ok=True)

    visualize_network_sample(
        G,
        sample_size=500,
        output_path=output_dir / "shade_network_sample.png",
        title="Gangnam Shade Walking Network"
    )

    # Save quality distribution
    quality_df = builder.get_quality_distribution()
    quality_df.to_csv(output_dir / "edge_quality_distribution.csv", index=False)
    logger.info(f"Edge data saved to: {output_dir / 'edge_quality_distribution.csv'}")

    # Save graph (for later analysis)
    nx.write_gexf(G, output_dir / "shade_network.gexf")
    logger.info(f"Graph saved to: {output_dir / 'shade_network.gexf'}")

    # Summary
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  - {output_dir}/shade_network_sample.png")
    print(f"  - {output_dir}/edge_quality_distribution.csv")
    print(f"  - {output_dir}/shade_network.gexf")


if __name__ == "__main__":
    main()
