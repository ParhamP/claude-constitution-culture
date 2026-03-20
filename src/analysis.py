"""
Analysis for Claude Constitution Culture Eval

Core analyses:
1. Cultural clustering — where does Claude sit among WVS countries?
2. Steerability — how much does cultural context shift Claude's responses?
3. Refusal patterns — where does Claude refuse to engage?
"""

import json
import ast
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_country_means() -> pd.DataFrame:
    """Load WVS country-level means for selected items."""
    return pd.read_csv(DATA_DIR / "processed" / "country_means.csv")


def load_format_a_results() -> pd.DataFrame:
    """Load parsed Format A results."""
    return pd.read_csv(RESULTS_DIR / "coded" / "format_a_parsed.csv")


def load_format_b_coded() -> pd.DataFrame:
    """Load LLM-coded Format B results."""
    return pd.read_csv(RESULTS_DIR / "coded" / "format_b_coded.csv")


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "processed" / "selected_items.csv")


# ============================================================
# Analysis 1: Cultural Clustering
# ============================================================

def cultural_clustering(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame):
    """
    Compare Claude's Format A response vector to each country's mean vector.
    Produce correlation ranking and hierarchical clustering.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build Claude's response vector
    claude_responses = format_a_df[format_a_df['parsed_value'].notna()].copy()
    claude_vector = claude_responses.groupby('item_id')['parsed_value'].mean()

    # Build country response matrix
    # Pivot: rows=countries, columns=items
    country_pivot = country_means_df.pivot_table(
        index='country', columns='item_id', values='mean_response'
    )

    # Only use items that Claude answered
    common_items = sorted(set(claude_vector.index) & set(country_pivot.columns))
    if len(common_items) < 5:
        print(f"WARNING: Only {len(common_items)} common items between Claude and WVS")
        return None

    claude_vec = claude_vector[common_items].values
    country_mat = country_pivot[common_items]

    # Drop countries with too many NaNs
    valid_countries = country_mat.dropna(thresh=len(common_items) * 0.7)
    # Fill remaining NaNs with column means for clustering
    valid_countries = valid_countries.fillna(valid_countries.mean())

    print(f"\n=== CULTURAL CLUSTERING ===")
    print(f"Items used: {len(common_items)}")
    print(f"Countries: {len(valid_countries)}")

    # Standardize for Euclidean distance (z-score each item across countries)
    # IMPORTANT: fit on countries only, then transform Claude separately
    scaler_corr = StandardScaler()
    countries_z_corr = pd.DataFrame(
        scaler_corr.fit_transform(valid_countries.values),
        index=valid_countries.index, columns=valid_countries.columns
    )
    claude_vec_z = (claude_vec - scaler_corr.mean_) / scaler_corr.scale_
    countries_z = countries_z_corr

    # Compute correlations + Euclidean distance with Claude
    correlations = {}
    for country in valid_countries.index:
        country_vec = valid_countries.loc[country].values
        r, p = stats.pearsonr(claude_vec, country_vec)
        rho, p_spearman = stats.spearmanr(claude_vec, country_vec)
        # Euclidean distance on standardized data (like Tao et al.)
        eucl_dist = np.sqrt(((claude_vec_z - countries_z.loc[country].values) ** 2).sum())
        correlations[country] = {'r': r, 'p': p, 'rho': rho, 'p_spearman': p_spearman,
                                 'euclidean_dist': eucl_dist}

    corr_df = pd.DataFrame(correlations).T.sort_values('r', ascending=False)
    corr_df.index.name = 'country'

    # Apply FDR correction for multiple comparisons (Pearson and Spearman)
    _, p_fdr, _, _ = multipletests(corr_df['p'].values, method='fdr_bh')
    corr_df['p_fdr'] = p_fdr
    _, p_fdr_spearman, _, _ = multipletests(corr_df['p_spearman'].values, method='fdr_bh')
    corr_df['p_spearman_fdr'] = p_fdr_spearman

    # Add confidence intervals (Fisher z-transformation)
    n_items = len(common_items)
    from scipy.stats import norm as norm_dist
    for country in corr_df.index:
        r = corr_df.loc[country, 'r']
        z = np.arctanh(r)
        se = 1 / np.sqrt(n_items - 3)
        z_crit = norm_dist.ppf(0.975)
        corr_df.loc[country, 'ci_lo'] = np.tanh(z - z_crit * se)
        corr_df.loc[country, 'ci_hi'] = np.tanh(z + z_crit * se)

    print(f"\nTop 15 countries most similar to Claude:")
    for country, row in corr_df.head(15).iterrows():
        sig = "***" if row['p_fdr'] < 0.001 else "**" if row['p_fdr'] < 0.01 else "*" if row['p_fdr'] < 0.05 else ""
        print(f"  {country:25s}  r={row['r']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]{sig}")

    print(f"\nBottom 15 countries least similar to Claude:")
    for country, row in corr_df.tail(15).iterrows():
        sig = "***" if row['p_fdr'] < 0.001 else "**" if row['p_fdr'] < 0.01 else "*" if row['p_fdr'] < 0.05 else ""
        print(f"  {country:25s}  r={row['r']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]{sig}")

    # Save correlations
    corr_df.to_csv(RESULTS_DIR / "tables" / "country_correlations.csv")

    # --- Figure 2: Correlation bar chart with CI error bars ---
    fig, ax = plt.subplots(figsize=(10, 18))
    y_pos = np.arange(len(corr_df))
    bar_colors = []
    for r_val in corr_df['r']:
        if r_val > 0.7:
            bar_colors.append('#1a7a3a')  # dark green
        elif r_val > 0.5:
            bar_colors.append('#5cb85c')  # medium green
        elif r_val > 0.3:
            bar_colors.append('#f0ad4e')  # amber
        else:
            bar_colors.append('#d9534f')  # red
    xerr_lo = corr_df['r'].values - corr_df['ci_lo'].values
    xerr_hi = corr_df['ci_hi'].values - corr_df['r'].values
    ax.barh(y_pos, corr_df['r'], color=bar_colors, alpha=0.85, height=0.75,
            xerr=[xerr_lo, xerr_hi],
            error_kw=dict(ecolor='#333333', lw=0.8, capsize=2, capthick=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_df.index, fontsize=8)
    ax.set_xlabel('Pearson Correlation with Claude (95% CI)', fontsize=11, fontweight='medium')
    ax.set_title("Cultural Similarity to Claude's Value Positions", fontsize=13, fontweight='bold', pad=12)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-0.05, 1.0)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Highlight US
    us_idx = list(corr_df.index).index('United States')
    ax.get_yticklabels()[us_idx].set_fontweight('bold')
    ax.get_yticklabels()[us_idx].set_color('#d9534f')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "country_correlations.png", dpi=200)
    plt.close()
    print(f"\nSaved correlation chart to {FIGURES_DIR / 'country_correlations.png'}")

    # --- Figure 2: Hierarchical clustering dendrogram ---
    # Standardize on countries only, then transform Claude
    scaler_clust = StandardScaler()
    countries_z = pd.DataFrame(
        scaler_clust.fit_transform(valid_countries.values),
        index=valid_countries.index, columns=valid_countries.columns
    )
    claude_z = pd.DataFrame(
        scaler_clust.transform(claude_vec.reshape(1, -1)),
        index=['CLAUDE'], columns=valid_countries.columns
    )
    combined_z = pd.concat([countries_z, claude_z])

    # Ward linkage on Euclidean distance of standardized data
    condensed = pdist(combined_z.values, metric='euclidean')
    Z = linkage(condensed, method='ward')

    fig, ax = plt.subplots(figsize=(20, 11))
    labels = list(combined_z.index)

    dn = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8.5,
                    color_threshold=25)

    # Color and size Claude's label
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        if lbl.get_text() == 'CLAUDE':
            lbl.set_color('#1a1a1a')
            lbl.set_fontweight('bold')
            lbl.set_fontsize(12)
        else:
            lbl.set_fontsize(8)

    ax.set_title("Hierarchical Clustering of Value Positions",
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel("Ward Distance (Euclidean, standardized)", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES_DIR / "clustering_dendrogram.png", dpi=200)
    plt.close()
    print(f"Saved dendrogram to {FIGURES_DIR / 'clustering_dendrogram.png'}")

    return corr_df


# ============================================================
# Analysis 1b: Cultural Map (Inglehart-Welzel style)
# ============================================================

# Inglehart-Welzel cultural zones for coloring
CULTURAL_ZONES = {
    'Protestant Europe': [
        'Sweden', 'Denmark', 'Norway', 'Finland', 'Iceland',
        'Germany', 'Netherlands', 'Switzerland', 'Estonia', 'Latvia',
    ],
    'English-Speaking': [
        'United States', 'Australia', 'New Zealand', 'Canada',
        'Great Britain', 'United Kingdom', 'Ireland', 'Northern Ireland',
    ],
    'Catholic Europe': [
        'France', 'Spain', 'Italy', 'Portugal', 'Belgium',
        'Austria', 'Poland', 'Czech Republic', 'Czechia', 'Slovakia',
        'Hungary', 'Croatia', 'Slovenia', 'Lithuania', 'Andorra',
    ],
    'Confucian': [
        'Japan', 'South Korea', 'China', 'Taiwan', 'Taiwan ROC',
        'Hong Kong SAR', 'Macau SAR', 'Vietnam', 'Singapore',
        'Thailand', 'Philippines', 'Mongolia',
    ],
    'South Asia': [
        'India', 'Bangladesh', 'Pakistan', 'Sri Lanka',
        'Nepal', 'Myanmar', 'Maldives',
    ],
    'African-Islamic': [
        'Nigeria', 'Egypt', 'Morocco', 'Ethiopia', 'Kenya',
        'Tanzania', 'Ghana', 'Zimbabwe', 'Tunisia', 'Libya',
        'Algeria', 'Iran', 'Iraq', 'Jordan', 'Lebanon',
        'Turkey', 'Indonesia', 'Malaysia', 'Saudi Arabia',
        'Tajikistan', 'Kyrgyzstan', 'Kazakhstan', 'Azerbaijan',
    ],
    'Orthodox Europe': [
        'Russia', 'Ukraine', 'Serbia', 'Romania', 'Bulgaria',
        'Georgia', 'Armenia', 'Belarus', 'Moldova', 'Albania',
        'Greece', 'Cyprus', 'North Macedonia', 'Montenegro', 'Bosnia Herzegovina',
    ],
    'Latin America': [
        'Brazil', 'Mexico', 'Argentina', 'Colombia', 'Chile',
        'Peru', 'Venezuela', 'Ecuador', 'Bolivia', 'Guatemala',
        'Nicaragua', 'Puerto Rico', 'Dominican Republic', 'Uruguay',
    ],
}

ZONE_COLORS = {
    'Protestant Europe': '#1f77b4',
    'English-Speaking': '#ff7f0e',
    'Catholic Europe': '#2ca02c',
    'Confucian': '#d62728',
    'South Asia': '#9467bd',
    'African-Islamic': '#8c564b',
    'Orthodox Europe': '#e377c2',
    'Latin America': '#bcbd22',
    'Other': '#7f7f7f',
    'CLAUDE': '#000000',
}


def get_zone(country: str) -> str:
    """Get the Inglehart-Welzel cultural zone for a country."""
    for zone, countries in CULTURAL_ZONES.items():
        if country in countries:
            return zone
    return 'Other'


def cultural_map(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame):
    """
    Create an Inglehart-Welzel style 2D cultural map using MDS
    (like Atari et al. Figure 2), with Claude plotted among countries.
    """
    from sklearn.manifold import MDS
    from matplotlib.patches import Ellipse, Patch

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    claude_responses = format_a_df[format_a_df['parsed_value'].notna()].copy()
    claude_vector = claude_responses.groupby('item_id')['parsed_value'].mean()

    country_pivot = country_means_df.pivot_table(
        index='country', columns='item_id', values='mean_response'
    )

    common_items = sorted(set(claude_vector.index) & set(country_pivot.columns))
    if len(common_items) < 5:
        print("WARNING: Too few common items for cultural map")
        return

    claude_vec = claude_vector[common_items]
    country_mat = country_pivot[common_items]

    valid = country_mat.dropna(thresh=len(common_items) * 0.7)
    valid = valid.fillna(valid.mean())

    scaler = StandardScaler()
    valid_z = pd.DataFrame(
        scaler.fit_transform(valid.values),
        index=valid.index, columns=valid.columns
    )
    claude_z_vals = scaler.transform(claude_vec.values.reshape(1, -1))

    combined = pd.concat([valid_z, pd.DataFrame(claude_z_vals, index=['CLAUDE'], columns=valid_z.columns)])

    # MDS like Atari et al.
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, n_init=10, max_iter=1000, normalized_stress='auto')
    coords = mds.fit_transform(combined.values)
    coords_df = pd.DataFrame(coords, index=combined.index, columns=['Dim1', 'Dim2'])
    coords_df['zone'] = coords_df.index.map(lambda x: 'CLAUDE' if x == 'CLAUDE' else get_zone(x))

    print(f"\n=== CULTURAL MAP (MDS) ===")
    print(f"Stress: {mds.stress_:.2f}")
    print(f"Claude position: Dim1={coords_df.loc['CLAUDE', 'Dim1']:.3f}, Dim2={coords_df.loc['CLAUDE', 'Dim2']:.3f}")

    claude_pos = coords_df.loc['CLAUDE', ['Dim1', 'Dim2']].values
    distances = {}
    for country in coords_df.index:
        if country == 'CLAUDE':
            continue
        c = coords_df.loc[country, ['Dim1', 'Dim2']].values
        distances[country] = np.sqrt(((claude_pos - c) ** 2).sum())

    nearest = sorted(distances.items(), key=lambda x: x[1])[:10]
    print(f"\nNearest countries to Claude in MDS space:")
    for country, dist in nearest:
        print(f"  {country:25s}  dist={dist:.3f}  zone={get_zone(country)}")

    # --- Figure: MDS Cultural Map ---
    fig, ax = plt.subplots(figsize=(14, 11))

    zone_order = ['Protestant Europe', 'English-Speaking', 'Catholic Europe',
                  'Confucian', 'South Asia', 'African-Islamic', 'Orthodox Europe',
                  'Latin America']

    # Colors matching Atari style — warm for Western, cool/neutral for others
    zp = {
        'Protestant Europe': '#c44e52',
        'English-Speaking': '#dd8452',
        'Catholic Europe': '#4c72b0',
        'Confucian': '#55a868',
        'South Asia': '#8172b3',
        'African-Islamic': '#937860',
        'Orthodox Europe': '#da8bc3',
        'Latin America': '#ccb974',
        'Other': '#aaaaaa',
    }

    # 1) Cluster ellipses — outline only like Atari
    for zone in zone_order:
        zone_data = coords_df[coords_df['zone'] == zone]
        if len(zone_data) < 3:
            continue
        color = zp.get(zone, '#999')
        pts = zone_data[['Dim1', 'Dim2']].values

        mean_x, mean_y = pts[:, 0].mean(), pts[:, 1].mean()
        cov = np.cov(pts[:, 0], pts[:, 1])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(5.991 * eigenvalues[0]) * 1.15
        height = 2 * np.sqrt(5.991 * eigenvalues[1]) * 1.15

        ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                          facecolor=color, alpha=0.07, edgecolor=color,
                          linewidth=1.3, linestyle='-', zorder=0)
        ax.add_patch(ellipse)

    # 2) Country labels
    skip_countries = {
        'Macao SAR', 'Hong Kong SAR', 'Andorra', 'Taiwan ROC', 'Maldives',
        'Montenegro', 'North Macedonia', 'Moldova', 'Belarus',
        'Bosnia Herzegovina', 'Dominican Republic', 'Nicaragua',
    }

    for country in coords_df.index:
        if country == 'CLAUDE' or country in skip_countries:
            continue
        x, y = coords_df.loc[country, ['Dim1', 'Dim2']]
        zone = get_zone(country)
        color = zp.get(zone, '#999')

        ax.plot(x, y, '.', color=color, markersize=2.5, zorder=2, alpha=0.4)
        ax.text(x, y, country, fontsize=6.5, ha='center', va='center',
                color=color, fontweight='medium', zorder=5,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=color, linewidth=0.5, alpha=0.92))

    # 3) Claude
    cx, cy = coords_df.loc['CLAUDE', ['Dim1', 'Dim2']]
    ax.plot(cx, cy, '.', color='#1a1a1a', markersize=8, zorder=9)
    ax.text(cx, cy, '  Claude  ', fontsize=13, ha='center', va='center',
            color='white', fontweight='bold', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                      edgecolor='#000000', linewidth=1.5, alpha=0.95))

    # 4) Legend
    legend_handles = []
    for zone in zone_order:
        color = zp.get(zone, '#999')
        if len(coords_df[coords_df['zone'] == zone]) > 0:
            legend_handles.append(Patch(facecolor=color, alpha=0.25, edgecolor=color,
                                        linewidth=1, label=zone))
    legend_handles.append(Patch(facecolor='#1a1a1a', edgecolor='#000000',
                                linewidth=1, label='Claude'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8.5, framealpha=0.95,
              edgecolor='#cccccc', fancybox=True)

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Claude on the Global Cultural Map',
                 fontsize=14, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cultural_map_mds.png", dpi=200)
    plt.savefig(FIGURES_DIR / "cultural_map_mds.pdf", format='pdf')
    plt.close()
    print(f"\nSaved cultural map to {FIGURES_DIR / 'cultural_map_mds.png'}")

    return coords_df


# ============================================================
# Analysis 1c: Distribution Overlap Plots
# ============================================================

def distribution_overlap_plots(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame):
    """
    For the most culturally divisive items, show Claude's response
    overlaid on distributions from contrasting countries.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    items_df = load_items()

    # Pick the 6 highest-variance items
    top_items = items_df.nlargest(6, 'cross_country_var')

    # Use 4 countries for cleaner bars
    contrast_countries = ['Sweden', 'United States', 'Nigeria', 'Egypt']
    country_colors = {'Sweden': '#2166ac', 'United States': '#4393c3',
                      'Nigeria': '#d6604d', 'Egypt': '#b2182b'}

    # Load full selections once
    full_items = json.loads(
        (Path(__file__).parent.parent / "data" / "processed" / "selected_items_full.json").read_text()
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for idx, (_, item) in enumerate(top_items.iterrows()):
        ax = axes[idx // 3, idx % 3]
        item_id = item['item_id']

        options = ast.literal_eval(item['options'])
        skip = {"Don't know", "No answer", "Missing; Not available",
                "Missing; Not applicable for other reasons", "No answer/refused",
                "Other missing; Multiple answers Mail (EVS)", "Missing; Unknown",
                "Missing; Not asked in survey", "DK/Refused"}
        substantive = [o for o in options if o not in skip]
        n_opts = len(substantive)

        claude_resp = format_a_df[
            (format_a_df['item_id'] == item_id) & (format_a_df['parsed_value'].notna())
        ]['parsed_value'].values

        full_item = [fi for fi in full_items if fi['item_id'] == item_id]
        if not full_item:
            continue
        selections = ast.literal_eval(
            full_item[0]['selections'].replace("defaultdict(<class 'list'>, ", "").rstrip(")")
        ) if isinstance(full_item[0]['selections'], str) else full_item[0]['selections']

        x = np.arange(1, n_opts + 1)
        n_countries = len(contrast_countries)
        width = 0.8 / n_countries

        for c_idx, country in enumerate(contrast_countries):
            if country in selections:
                dist = selections[country][:n_opts]
                s = sum(dist)
                if s > 0:
                    normalized = [p / s for p in dist]
                    offset = (c_idx - n_countries / 2 + 0.5) * width
                    ax.bar(x + offset, normalized, width, label=country,
                           color=country_colors[country], alpha=0.75, edgecolor='white', linewidth=0.3)

        # Claude marker — mean only, no range band for visual consistency
        if len(claude_resp) > 0:
            claude_mean = np.mean(claude_resp)
            ax.axvline(x=claude_mean, color='#1a1a1a', linewidth=2.5, linestyle='--',
                       label='Claude', zorder=10)

        # Topic label
        topic = item['question'].strip().split('\n')[-1].strip()
        if len(topic) > 50:
            topic = topic[:47] + '...'
        ax.set_title(topic, fontsize=9, fontweight='bold', pad=6)

        # X-axis: use scale numbers for 10-point scales, short text for others
        if n_opts >= 8:
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in x], fontsize=7)
            # Label endpoints — full text, no truncation
            left_label = substantive[0].replace('justifiable', 'just.')
            right_label = substantive[-1].replace('justifiable', 'just.')
            ax.text(1, -0.1, left_label, transform=ax.get_xaxis_transform(),
                    fontsize=6, ha='left', va='top', color='#444', style='italic')
            ax.text(n_opts, -0.1, right_label, transform=ax.get_xaxis_transform(),
                    fontsize=6, ha='right', va='top', color='#444', style='italic')
        else:
            ax.set_xticks(x)
            short_labels = []
            for opt in substantive:
                if len(opt) > 20:
                    short_labels.append(opt[:18] + '..')
                else:
                    short_labels.append(opt)
            ax.set_xticklabels(short_labels, fontsize=6.5, rotation=30, ha='right')

        ax.set_ylabel('Proportion', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=7)

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Claude's Position on Culturally Divisive Items",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(FIGURES_DIR / "distribution_overlap.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved distribution overlap plots to {FIGURES_DIR / 'distribution_overlap.png'}")


# ============================================================
# Analysis 2: Steerability
# ============================================================

def steerability_analysis(format_b_coded: pd.DataFrame, format_b_responses: pd.DataFrame,
                          country_means_df: pd.DataFrame):
    """
    Measure how much cultural context in Format B shifts Claude's implied value position.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Merge coded values with response metadata
    merged = format_b_responses.merge(format_b_coded[['prompt_id', 'coded_value', 'strategy']],
                                       on='prompt_id', how='left')

    # Separate baseline and country-contextualized
    baseline = merged[merged['condition'] == 'baseline'].copy()
    contextualized = merged[merged['condition'] == 'country_context'].copy()

    if len(baseline) == 0 or len(contextualized) == 0:
        print("WARNING: Missing baseline or contextualized data")
        return None

    # Compute baseline mean per item
    baseline_means = baseline.groupby('item_id')['coded_value'].mean()

    # Compute contextualized mean per item x country
    context_means = contextualized.groupby(['item_id', 'country'])['coded_value'].mean()

    # Compute shift from baseline
    shifts = []
    for (item_id, country), ctx_mean in context_means.items():
        bl_mean = baseline_means.get(item_id)
        if bl_mean is not None and not np.isnan(ctx_mean) and not np.isnan(bl_mean):
            shifts.append({
                'item_id': item_id,
                'country': country,
                'baseline_value': bl_mean,
                'contextualized_value': ctx_mean,
                'shift': ctx_mean - bl_mean,
            })

    shifts_df = pd.DataFrame(shifts)
    if len(shifts_df) == 0:
        print("WARNING: No valid shifts computed")
        return None

    print(f"\n=== STEERABILITY ANALYSIS ===")
    print(f"Valid shift observations: {len(shifts_df)}")

    # Average shift per country
    country_shifts = shifts_df.groupby('country').agg(
        mean_shift=('shift', 'mean'),
        abs_mean_shift=('shift', lambda x: np.abs(x).mean()),
        std_shift=('shift', 'std'),
        n_items=('shift', 'count'),
    ).sort_values('abs_mean_shift', ascending=False)

    # Wilcoxon signed-rank test per country: is shift distribution != 0?
    from scipy.stats import wilcoxon
    wilcoxon_results = {}
    for country in shifts_df['country'].unique():
        country_data = shifts_df[shifts_df['country'] == country]['shift'].dropna()
        if len(country_data) >= 5:
            try:
                stat, p_val = wilcoxon(country_data, alternative='two-sided')
                wilcoxon_results[country] = p_val
            except ValueError:
                wilcoxon_results[country] = 1.0  # all zeros
        else:
            wilcoxon_results[country] = float('nan')

    # FDR correct the Wilcoxon p-values
    w_countries = [c for c in wilcoxon_results if not np.isnan(wilcoxon_results[c])]
    w_pvals = [wilcoxon_results[c] for c in w_countries]
    if len(w_pvals) > 0:
        _, w_fdr, _, _ = multipletests(w_pvals, method='fdr_bh')
        for c, p_corr in zip(w_countries, w_fdr):
            wilcoxon_results[c] = p_corr

    country_shifts['wilcoxon_p_fdr'] = country_shifts.index.map(wilcoxon_results)

    print(f"\nMean absolute shift by country (Wilcoxon signed-rank, FDR-corrected):")
    for country, row in country_shifts.iterrows():
        p = row.get('wilcoxon_p_fdr', float('nan'))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {country:20s}  |shift|={row['abs_mean_shift']:.3f}  mean={row['mean_shift']:+.3f}  p={p:.4f}{sig}")

    # Save
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    country_shifts.to_csv(RESULTS_DIR / "tables" / "steerability_by_country.csv")
    shifts_df.to_csv(RESULTS_DIR / "tables" / "steerability_all_shifts.csv", index=False)

    # --- Figure: Steerability heatmap with SEM ---
    items_df = load_items()
    item_domains = items_df.set_index('item_id')['domain'].to_dict()
    shifts_df['domain'] = shifts_df['item_id'].map(item_domains)

    # Clean domain names
    domain_rename = {
        'moral_justifiability': 'Moral Justifiability',
        'gender_family': 'Gender & Family',
        'religion_values': 'Religion & Values',
        'tolerance_neighbors': 'Tolerance',
        'political_authority': 'Political Authority',
        'economic_values': 'Economic Values',
    }
    shifts_df['domain_clean'] = shifts_df['domain'].map(domain_rename)

    domain_country_shift = shifts_df.groupby(['domain_clean', 'country'])['shift'].mean().unstack()
    domain_country_sem = shifts_df.groupby(['domain_clean', 'country'])['shift'].sem().unstack()

    # Build annotation strings: mean ± SEM
    annot_strings = domain_country_shift.copy().astype(str)
    for dom in domain_country_shift.index:
        for country in domain_country_shift.columns:
            mean_val = domain_country_shift.loc[dom, country]
            sem_val = domain_country_sem.loc[dom, country]
            if pd.notna(mean_val) and pd.notna(sem_val):
                annot_strings.loc[dom, country] = f"{mean_val:+.2f}\n(±{sem_val:.2f})"
            else:
                annot_strings.loc[dom, country] = ""

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(domain_country_shift, cmap='RdBu_r', center=0, annot=annot_strings.values,
                fmt='', ax=ax, cbar_kws={'label': 'Mean Shift from Baseline', 'shrink': 0.8},
                annot_kws={'fontsize': 7.5}, linewidths=0.5, linecolor='white')
    ax.set_title("Cultural Steerability of Claude's Advice",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Country Context", fontsize=11)
    ax.set_ylabel("Value Domain", fontsize=11)
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "steerability_heatmap.png", dpi=200)
    plt.close()
    print(f"\nSaved steerability heatmap to {FIGURES_DIR / 'steerability_heatmap.png'}")

    return shifts_df


# ============================================================
# Analysis 3: Refusal & Rhetorical Strategy Patterns
# ============================================================

def refusal_analysis(format_a_df: pd.DataFrame, format_b_coded: pd.DataFrame,
                     format_b_responses: pd.DataFrame):
    """Analyze where Claude refuses to engage or hedges."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    items_df = load_items()
    item_domains = items_df.set_index('item_id')['domain'].to_dict()

    print(f"\n=== REFUSAL & STRATEGY ANALYSIS ===")

    # Format A refusals
    format_a_df['domain'] = format_a_df['item_id'].map(item_domains)
    refusal_by_domain = format_a_df.groupby('domain')['is_refusal'].mean()
    print(f"\nFormat A refusal rate by domain:")
    for domain, rate in refusal_by_domain.sort_values(ascending=False).items():
        print(f"  {domain:25s}  {rate:.1%}")

    # Format B strategy distribution
    merged_b = format_b_responses.merge(format_b_coded[['prompt_id', 'strategy']],
                                         on='prompt_id', how='left')
    merged_b['domain'] = merged_b['item_id'].map(item_domains)

    # Overall strategy distribution
    print(f"\nFormat B rhetorical strategy distribution:")
    strategy_counts = merged_b['strategy'].value_counts(normalize=True)
    for strategy, pct in strategy_counts.items():
        print(f"  {strategy:20s}  {pct:.1%}")

    # Strategy by domain
    strategy_by_domain = pd.crosstab(merged_b['domain'], merged_b['strategy'], normalize='index')
    print(f"\nStrategy by domain:")
    print(strategy_by_domain.round(3).to_string())

    # Strategy by country (for contextualized only)
    ctx_only = merged_b[merged_b['condition'] == 'country_context']
    if len(ctx_only) > 0:
        strategy_by_country = pd.crosstab(ctx_only['country'], ctx_only['strategy'],
                                           normalize='index')
        # Reorder columns for consistent visual
        col_order = ['DIRECTIVE', 'BALANCED_LEAN', 'PURE_BALANCE', 'DEFERRAL']
        col_order = [c for c in col_order if c in strategy_by_country.columns]
        strategy_by_country = strategy_by_country[col_order]

        # Better labels
        label_rename = {'DIRECTIVE': 'Directive', 'BALANCED_LEAN': 'Balanced Lean',
                        'PURE_BALANCE': 'Pure Balance', 'DEFERRAL': 'Deferral'}
        strategy_by_country.columns = [label_rename.get(c, c) for c in strategy_by_country.columns]

        # Publication palette
        strat_colors = {'Directive': '#c0392b', 'Balanced Lean': '#2980b9',
                        'Pure Balance': '#27ae60', 'Deferral': '#f39c12'}
        colors_ordered = [strat_colors.get(c, '#999') for c in strategy_by_country.columns]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        strategy_by_country.plot(kind='barh', stacked=True, ax=ax, color=colors_ordered,
                                  edgecolor='white', linewidth=0.5)
        ax.set_title("Claude's Rhetorical Strategy by Country Context",
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Proportion", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.legend(title="Strategy", fontsize=9, title_fontsize=10,
                  loc='lower right', frameon=True, fancybox=True)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "strategy_by_country.png", dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        print(f"\nSaved strategy chart to {FIGURES_DIR / 'strategy_by_country.png'}")

    return strategy_by_domain


# ============================================================
# Domain-level analysis
# ============================================================

def domain_analysis(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame):
    """Compare Claude's position to WVS data, broken down by domain."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    items_df = load_items()
    item_domains = items_df.set_index('item_id')['domain'].to_dict()

    # Claude's mean per item
    claude_means = format_a_df[format_a_df['parsed_value'].notna()].groupby('item_id')['parsed_value'].mean()

    # US mean per item
    us_means = country_means_df[country_means_df['country'] == 'United States'].set_index('item_id')['mean_response']

    # Global mean per item
    global_means = country_means_df.groupby('item_id')['mean_response'].mean()

    # Build comparison
    comparison = pd.DataFrame({
        'claude': claude_means,
        'us': us_means,
        'global_mean': global_means,
    })
    n_before = len(comparison)
    comparison = comparison.dropna()
    n_dropped = n_before - len(comparison)
    comparison['domain'] = comparison.index.map(item_domains)

    print(f"\n=== DOMAIN-LEVEL COMPARISON ===")
    if n_dropped > 0:
        print(f"  NOTE: {n_dropped} items dropped due to missing Claude, US, or global mean data")
    for domain in sorted(comparison['domain'].unique()):
        d = comparison[comparison['domain'] == domain]
        if len(d) < 2:
            continue
        r_us, _ = stats.pearsonr(d['claude'], d['us']) if len(d) > 2 else (float('nan'), 1)
        r_global, _ = stats.pearsonr(d['claude'], d['global_mean']) if len(d) > 2 else (float('nan'), 1)
        mean_diff_us = (d['claude'] - d['us']).mean()
        mean_diff_global = (d['claude'] - d['global_mean']).mean()
        print(f"\n  {domain}:")
        print(f"    Correlation with US: {r_us:.3f}")
        print(f"    Correlation with global mean: {r_global:.3f}")
        print(f"    Mean diff from US: {mean_diff_us:+.3f}")
        print(f"    Mean diff from global: {mean_diff_global:+.3f}")

    # --- Figure: Claude vs US vs Global by domain ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    domains = sorted(comparison['domain'].unique())

    for idx, domain in enumerate(domains):
        ax = axes[idx // 3, idx % 3]
        d = comparison[comparison['domain'] == domain].copy()
        d = d.sort_values('claude')

        x = range(len(d))
        ax.scatter(x, d['claude'], label='Claude', color='#1a1a1a', s=60, zorder=3)
        ax.scatter(x, d['us'], label='US', color='blue', s=40, alpha=0.6, zorder=2)
        ax.scatter(x, d['global_mean'], label='Global Mean', color='gray', s=40, alpha=0.4, zorder=1)
        ax.set_title(domain.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Mean Response Value')
        if idx == 0:
            ax.legend(fontsize=8)

    # Remove any empty subplots
    for idx in range(len(domains), 6):
        axes[idx // 3, idx % 3].set_visible(False)

    fig.suptitle("Claude's Value Positions vs. US and Global Mean\n(by Domain, Format A)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "domain_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved domain comparison to {FIGURES_DIR / 'domain_comparison.png'}")


# ============================================================
# Analysis 5: Steerability Direction
# ============================================================

def steerability_direction_analysis(format_b_coded: pd.DataFrame,
                                     format_b_responses: pd.DataFrame,
                                     country_means_df: pd.DataFrame):
    """
    Does Claude shift TOWARD the actual WVS values of the named country,
    or does it shift randomly / in some other direction?
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    merged = format_b_responses.merge(
        format_b_coded[['prompt_id', 'coded_value']],
        on='prompt_id', how='left'
    )

    baseline = merged[merged['condition'] == 'baseline'].copy()
    contextualized = merged[merged['condition'] == 'country_context'].copy()

    baseline_means = baseline.groupby('item_id')['coded_value'].mean()
    context_means = contextualized.groupby(['item_id', 'country'])['coded_value'].mean()

    wvs_pivot = country_means_df.pivot_table(
        index='country', columns='item_id', values='mean_response'
    )

    records = []
    for (item_id, country), ctx_val in context_means.items():
        bl_val = baseline_means.get(item_id)
        if bl_val is None or np.isnan(ctx_val) or np.isnan(bl_val):
            continue
        if country not in wvs_pivot.index or item_id not in wvs_pivot.columns:
            continue
        wvs_val = wvs_pivot.loc[country, item_id]
        if np.isnan(wvs_val):
            continue

        dist_before = abs(bl_val - wvs_val)
        dist_after = abs(ctx_val - wvs_val)

        records.append({
            'item_id': item_id,
            'country': country,
            'baseline_value': bl_val,
            'contextualized_value': ctx_val,
            'wvs_value': wvs_val,
            'shift': ctx_val - bl_val,
            'dist_before': dist_before,
            'dist_after': dist_after,
            'moved_toward': dist_after < dist_before,
            'moved_away': dist_after > dist_before,
            'no_change': dist_after == dist_before,
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("WARNING: No valid direction observations")
        return None

    print(f"\n=== STEERABILITY DIRECTION ANALYSIS ===")
    print(f"Valid observations: {len(df)}")
    print(f"\nOverall:")
    print(f"  Moved toward WVS value: {df['moved_toward'].mean():.1%}")
    print(f"  Moved away from WVS value: {df['moved_away'].mean():.1%}")
    print(f"  No change: {df['no_change'].mean():.1%}")
    print(f"  Mean distance before context: {df['dist_before'].mean():.3f}")
    print(f"  Mean distance after context: {df['dist_after'].mean():.3f}")

    print(f"\nBy country:")
    country_dir = df.groupby('country').agg(
        toward_pct=('moved_toward', 'mean'),
        away_pct=('moved_away', 'mean'),
        mean_dist_before=('dist_before', 'mean'),
        mean_dist_after=('dist_after', 'mean'),
        n=('item_id', 'count'),
    ).sort_values('toward_pct', ascending=False)

    for country, row in country_dir.iterrows():
        delta = row['mean_dist_before'] - row['mean_dist_after']
        direction = "closer" if delta > 0 else "further"
        print(f"  {country:20s}  toward={row['toward_pct']:.0%}  away={row['away_pct']:.0%}  "
              f"dist_delta={delta:+.3f} ({direction})")

    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    country_dir.to_csv(RESULTS_DIR / "tables" / "steerability_direction.csv")
    df.to_csv(RESULTS_DIR / "tables" / "steerability_direction_all.csv", index=False)

    # Binomial test: is the proportion moving toward significantly > 50%?
    from scipy.stats import binomtest
    n_toward = df['moved_toward'].sum()
    n_with_change = len(df[~df['no_change']])
    if n_with_change > 0:
        result = binomtest(int(n_toward), n_with_change, 0.5, alternative='greater')
        print(f"\nBinomial test (toward > 50%): p={result.pvalue:.4f}")
        print(f"  {int(n_toward)}/{n_with_change} shifts moved toward WVS value")

    return df


# ============================================================
# Analysis 6: Jensen-Shannon Divergence
# ============================================================

def jsd_analysis(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame):
    """
    Compute Jensen-Shannon Divergence between Claude's response distribution
    and each country's distribution. More rigorous than Pearson on means.

    Note: scipy.spatial.distance.jensenshannon returns the JS *distance*
    (square root of JS divergence), using log base 2 by default.
    Values are bounded in [0, 1], not [0, ln(2)].
    """
    from scipy.spatial.distance import jensenshannon

    items_df = load_items()

    full_items = json.loads(
        (Path(__file__).parent.parent / "data" / "processed" / "selected_items_full.json").read_text()
    )

    SKIP_OPTIONS = {
        "Don't know", "No answer", "Missing; Not available",
        "Missing; Not applicable for other reasons", "No answer/refused",
        "Other missing; Multiple answers Mail (EVS)", "Missing; Unknown",
        "Missing; Not asked in survey", "DK/Refused",
    }

    records = []
    for fi in full_items:
        item_id = fi['item_id']
        options = ast.literal_eval(fi['options']) if isinstance(fi['options'], str) else fi['options']
        n_substantive = len([o for o in options if o not in SKIP_OPTIONS])

        claude_responses = format_a_df[
            (format_a_df['item_id'] == item_id) & (format_a_df['parsed_value'].notna())
        ]['parsed_value'].values

        if len(claude_responses) == 0:
            continue

        # Build Claude's empirical distribution
        claude_dist = np.zeros(n_substantive)
        for v in claude_responses:
            idx = int(round(v)) - 1
            if 0 <= idx < n_substantive:
                claude_dist[idx] += 1
        if claude_dist.sum() == 0:
            continue
        claude_dist = claude_dist / claude_dist.sum()

        # Parse country distributions
        sel_str = fi['selections']
        if isinstance(sel_str, str):
            sel_str = sel_str.replace("defaultdict(<class 'list'>, ", "").rstrip(")")
            selections = ast.literal_eval(sel_str)
        else:
            selections = sel_str

        for country, dist in selections.items():
            country_dist = np.array(dist[:n_substantive], dtype=float)
            s = country_dist.sum()
            if s < 0.5:
                continue
            country_dist = country_dist / s

            eps = 1e-10
            claude_smooth = claude_dist + eps
            claude_smooth = claude_smooth / claude_smooth.sum()
            country_smooth = country_dist + eps
            country_smooth = country_smooth / country_smooth.sum()

            jsd = jensenshannon(claude_smooth, country_smooth)

            records.append({
                'item_id': item_id,
                'country': country,
                'jsd': jsd,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("WARNING: No JSD values computed")
        return None

    print(f"\n=== JENSEN-SHANNON DIVERGENCE ANALYSIS ===")
    print(f"Total observations: {len(df)}")

    country_jsd = df.groupby('country')['jsd'].mean().sort_values()
    print(f"\nMost similar countries (lowest mean JSD):")
    for country, jsd_val in country_jsd.head(15).items():
        print(f"  {country:25s}  JSD={jsd_val:.4f}")

    print(f"\nLeast similar countries (highest mean JSD):")
    for country, jsd_val in country_jsd.tail(15).items():
        print(f"  {country:25s}  JSD={jsd_val:.4f}")

    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    country_jsd.to_csv(RESULTS_DIR / "tables" / "jsd_by_country.csv")

    # Check correlation between JSD and Pearson rankings
    try:
        pearson_df = pd.read_csv(RESULTS_DIR / "tables" / "country_correlations.csv", index_col=0)
        common = set(country_jsd.index) & set(pearson_df.index)
        if len(common) > 10:
            jsd_ranks = country_jsd[list(common)].rank()
            pearson_ranks = pearson_df.loc[list(common), 'r'].rank(ascending=False)
            rank_corr, rank_p = stats.spearmanr(jsd_ranks, pearson_ranks)
            print(f"\nSpearman correlation between JSD and Pearson rankings: rho={rank_corr:.3f}, p={rank_p:.4f}")
    except FileNotFoundError:
        pass

    # --- Figure: JSD bar chart ---
    fig, ax = plt.subplots(figsize=(12, 16))
    colors = ['#2ecc71' if j < country_jsd.median() else '#e74c3c'
              for j in country_jsd.values]
    ax.barh(range(len(country_jsd)), country_jsd.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(country_jsd)))
    ax.set_yticklabels(country_jsd.index, fontsize=7)
    ax.set_xlabel('Mean Jensen-Shannon Divergence from Claude', fontsize=12)
    ax.set_title("Distributional Distance from Claude\n(Jensen-Shannon Divergence, Format A)", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "jsd_by_country.png", dpi=150)
    plt.savefig(FIGURES_DIR / "jsd_by_country.pdf", format='pdf')
    plt.close()
    print(f"\nSaved JSD chart to {FIGURES_DIR / 'jsd_by_country.png'}")

    return df


# ============================================================
# Analysis 7: Bootstrap CIs on Extremity Claims
# ============================================================

def bootstrap_extremity_analysis(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame,
                                  n_bootstrap: int = 10000):
    """
    Bootstrap confidence intervals on Claude's mean per item (from 5 runs),
    then test whether Claude is significantly beyond the most extreme country.
    """
    items_df = load_items()
    country_pivot = country_means_df.pivot_table(
        index='country', columns='item_id', values='mean_response'
    )

    valid = format_a_df[format_a_df['parsed_value'].notna()].copy()
    rng = np.random.default_rng(seed=42)

    records = []
    for item_id in valid['item_id'].unique():
        runs = valid[valid['item_id'] == item_id]['parsed_value'].values
        if len(runs) < 2:
            continue

        if item_id not in country_pivot.columns:
            continue

        country_vals = country_pivot[item_id].dropna()
        country_min = country_vals.min()
        country_max = country_vals.max()

        # Bootstrap CI on Claude's mean
        boot_means = np.array([
            np.mean(rng.choice(runs, size=len(runs), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        claude_mean = np.mean(runs)

        # Is Claude significantly beyond all countries?
        beyond_above = ci_lower > country_max  # Even the lower bound exceeds all countries
        beyond_below = ci_upper < country_min   # Even the upper bound is below all countries
        significantly_beyond = beyond_above or beyond_below

        item_info = items_df[items_df['item_id'] == item_id]
        topic = ""
        domain = ""
        if len(item_info) > 0:
            topic = item_info.iloc[0]['question'].strip().split('\n')[-1].strip()[:60]
            domain = item_info.iloc[0]['domain']

        records.append({
            'item_id': item_id,
            'domain': domain,
            'topic': topic,
            'claude_mean': claude_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'country_min': country_min,
            'country_max': country_max,
            'beyond_above': beyond_above,
            'beyond_below': beyond_below,
            'significantly_beyond': significantly_beyond,
            'n_runs': len(runs),
            'run_std': np.std(runs),
        })

    df = pd.DataFrame(records)

    print(f"\n=== BOOTSTRAP EXTREMITY ANALYSIS (n={n_bootstrap}) ===")
    print(f"Items analyzed: {len(df)}")
    print(f"Items with 95% CI entirely beyond all countries: {df['significantly_beyond'].sum()}")
    print(f"Items with point estimate beyond all countries: {((df['claude_mean'] > df['country_max']) | (df['claude_mean'] < df['country_min'])).sum()}")
    print(f"\nResponse consistency (std across 5 runs):")
    print(f"  Mean std: {df['run_std'].mean():.3f}")
    print(f"  Items with zero variance (same answer every run): {(df['run_std'] == 0).sum()}/{len(df)}")

    print(f"\nSignificantly beyond all countries (CI doesn't overlap country range):")
    sig = df[df['significantly_beyond']].sort_values('claude_mean', ascending=False)
    for _, r in sig.iterrows():
        direction = "ABOVE" if r['beyond_above'] else "BELOW"
        print(f"  {r['topic']:<50s}  mean={r['claude_mean']:.1f} [{r['ci_lower']:.1f}, {r['ci_upper']:.1f}]  "
              f"range=[{r['country_min']:.2f}, {r['country_max']:.2f}]  {direction}")

    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tables" / "bootstrap_extremity.csv", index=False)

    return df


# ============================================================
# Analysis 8: Cohen's d on Steerability
# ============================================================

def steerability_effect_sizes(format_b_coded: pd.DataFrame,
                               format_b_responses: pd.DataFrame):
    """
    Compute Cohen's d for the effect of country context on Claude's coded values.
    """
    merged = format_b_responses.merge(
        format_b_coded[['prompt_id', 'coded_value']],
        on='prompt_id', how='left'
    )

    baseline = merged[merged['condition'] == 'baseline'].copy()
    contextualized = merged[merged['condition'] == 'country_context'].copy()

    baseline_values = baseline['coded_value'].dropna()
    bl_mean = baseline_values.mean()
    bl_std = baseline_values.std()

    print(f"\n=== STEERABILITY EFFECT SIZES (paired Cohen's d) ===")
    print(f"Baseline overall: mean={bl_mean:.3f}, std={bl_std:.3f}, n={len(baseline_values)}")

    records = []
    for country in sorted(contextualized['country'].dropna().unique()):
        ctx_country = contextualized[contextualized['country'] == country].copy()
        if len(ctx_country) < 5:
            continue

        # Paired Cohen's d: match items between baseline and country condition
        diffs = []
        common_items = set(baseline['item_id'].unique()) & set(ctx_country['item_id'].unique())
        for item_id in common_items:
            bl_item = baseline[baseline['item_id'] == item_id]['coded_value'].dropna()
            ctx_item = ctx_country[ctx_country['item_id'] == item_id]['coded_value'].dropna()
            if len(bl_item) == 0 or len(ctx_item) == 0:
                continue
            diffs.append(ctx_item.values[0] - bl_item.values[0])

        if len(diffs) == 0:
            continue
        diffs = np.array(diffs)
        # Cohen's d for paired data: mean(diffs) / SD(diffs)
        sd_diffs = np.std(diffs, ddof=1)
        d = np.mean(diffs) / sd_diffs if sd_diffs > 0 else 0

        ctx_mean = ctx_country['coded_value'].dropna().mean()
        ctx_std = ctx_country['coded_value'].dropna().std()

        # Interpretation
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        records.append({
            'country': country,
            'ctx_mean': ctx_mean,
            'ctx_std': ctx_std,
            'cohens_d': d,
            'interpretation': interpretation,
            'n': len(diffs),
        })

    df = pd.DataFrame(records).sort_values('cohens_d', key=abs, ascending=False)

    for _, r in df.iterrows():
        print(f"  {r['country']:20s}  d={r['cohens_d']:+.3f}  ({r['interpretation']})")

    print(f"\nOverall: mean |d| = {df['cohens_d'].abs().mean():.3f}")

    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tables" / "steerability_cohens_d.csv", index=False)

    return df


# ============================================================
# Analysis 9: Gap Statistic for Optimal Clusters
# ============================================================

def gap_statistic_analysis(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame,
                            max_k: int = 10, n_refs: int = 100):
    """
    Use the gap statistic (Tibshirani et al. 2001) with Monte Carlo bootstrapping
    to determine optimal number of clusters, following Atari et al.'s approach.
    """
    from sklearn.cluster import KMeans

    # Build data matrix (countries + Claude)
    claude_responses = format_a_df[format_a_df['parsed_value'].notna()].copy()
    claude_vector = claude_responses.groupby('item_id')['parsed_value'].mean()

    country_pivot = country_means_df.pivot_table(
        index='country', columns='item_id', values='mean_response'
    )

    common_items = sorted(set(claude_vector.index) & set(country_pivot.columns))
    country_mat = country_pivot[common_items].dropna(thresh=len(common_items) * 0.7)
    country_mat = country_mat.fillna(country_mat.mean())

    # Standardize
    scaler = StandardScaler()
    country_scaled = pd.DataFrame(
        scaler.fit_transform(country_mat),
        index=country_mat.index, columns=country_mat.columns
    )

    # Add Claude
    claude_vec_scaled = (claude_vector[common_items] - scaler.mean_) / scaler.scale_
    claude_z_df = claude_vec_scaled.to_frame().T
    claude_z_df.index = ['CLAUDE']
    combined = pd.concat([country_scaled, claude_z_df])
    X = combined.values

    print(f"\n=== GAP STATISTIC ANALYSIS ===")
    print(f"Data: {X.shape[0]} entities (countries + Claude), {X.shape[1]} items")

    def compute_wk(data, k):
        """Within-cluster sum of squares."""
        if k == 1:
            return np.sum((data - data.mean(axis=0))**2)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(data)
        wk = 0
        for label in range(k):
            cluster_data = data[km.labels_ == label]
            if len(cluster_data) > 0:
                wk += np.sum((cluster_data - cluster_data.mean(axis=0))**2)
        return wk

    # Compute gap statistic for k=1..max_k
    gap_results = []
    for k in range(1, max_k + 1):
        wk = compute_wk(X, k)
        log_wk = np.log(wk) if wk > 0 else 0

        # Reference distribution (uniform random in bounding box)
        ref_log_wks = []
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        rng = np.random.default_rng(seed=42)
        for _ in range(n_refs):
            ref_data = rng.uniform(mins, maxs, size=X.shape)
            ref_wk = compute_wk(ref_data, k)
            ref_log_wks.append(np.log(ref_wk) if ref_wk > 0 else 0)

        ref_mean = np.mean(ref_log_wks)
        ref_std = np.std(ref_log_wks, ddof=1) * np.sqrt(1 + 1/n_refs)
        gap = ref_mean - log_wk

        gap_results.append({
            'k': k,
            'log_wk': log_wk,
            'ref_mean': ref_mean,
            'gap': gap,
            'sk': ref_std,
        })

    gap_df = pd.DataFrame(gap_results)

    # Find optimal k: smallest k such that Gap(k) >= Gap(k+1) - s(k+1)
    optimal_k = 1
    for i in range(len(gap_df) - 1):
        if gap_df.iloc[i]['gap'] >= gap_df.iloc[i+1]['gap'] - gap_df.iloc[i+1]['sk']:
            optimal_k = int(gap_df.iloc[i]['k'])
            break

    print(f"\nGap statistic results:")
    for _, r in gap_df.iterrows():
        marker = " <-- optimal" if r['k'] == optimal_k else ""
        print(f"  k={int(r['k']):2d}  Gap={r['gap']:.3f}  sk={r['sk']:.3f}{marker}")

    print(f"\nOptimal number of clusters: {optimal_k}")

    # Which cluster is Claude in?
    km_opt = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    km_opt.fit(X)
    claude_idx = list(combined.index).index('CLAUDE')
    claude_cluster = km_opt.labels_[claude_idx]

    cluster_members = {}
    for i, label in enumerate(km_opt.labels_):
        name = combined.index[i]
        cluster_members.setdefault(label, []).append(name)

    print(f"\nClaude is in cluster {claude_cluster} with {len(cluster_members[claude_cluster])} members:")
    for member in sorted(cluster_members[claude_cluster]):
        print(f"  {member}")

    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    gap_df.to_csv(RESULTS_DIR / "tables" / "gap_statistic.csv", index=False)

    return gap_df, optimal_k, cluster_members


# ============================================================
# Analysis 10: Inglehart-Welzel Canonical Dimensions
# ============================================================

def inglehart_welzel_dimensions(format_a_df: pd.DataFrame, country_means_df: pd.DataFrame):
    """
    Compute scores on the two canonical Inglehart-Welzel dimensions:
    - Traditional vs. Secular-Rational values
    - Survival vs. Self-Expression values

    Uses items from our selection that map to the canonical IW factor items.
    """
    items_df = load_items()

    # Mapping our items to IW dimensions based on WVS methodology
    # Traditional vs Secular-Rational: religion importance, God importance,
    # national pride, authority/obedience, abortion justifiability
    # Survival vs Self-Expression: life satisfaction, political action,
    # homosexuality tolerance, trust, self-expression priorities

    # Items in our set that map to Traditional vs Secular-Rational
    # Keywords must exactly match topic strings (from get_topic()) or be substrings thereof
    trad_secular_items = {
        # Religion importance (high = traditional)
        'Religion': {'direction': -1},  # "Very important"=1 is traditional, so invert
        # Abortion justifiability (high = secular-rational)
        'Abortion': {'direction': 1},  # Higher = more justifiable = more secular
        # Obey rulers (high = traditional) - topic is "People obey their rulers"
        'People obey their rulers': {'direction': -1},
        # God belief (having belief = traditional) - item 24 (refused, will be skipped)
        'God': {'direction': -1},
    }

    # Items in our set that map to Survival vs Self-Expression
    surv_selfexp_items = {
        # Homosexuality justifiability (high = self-expression)
        'Homosexuality': {'direction': 1},
        # Homosexuals as neighbors tolerance (high = self-expression)
        'Homosexuals': {'direction': 1},
        # Sex before marriage (high = self-expression)
        'Sex before marriage': {'direction': 1},
        # Having casual sex (high = self-expression)
        'Having casual sex': {'direction': 1},
    }

    # Extract topics for matching
    def get_topic(question):
        parts = question.strip().split('\n')
        return parts[-1].strip() if len(parts) > 1 else question.strip()[:50]

    item_topics = {}
    for _, item in items_df.iterrows():
        topic = get_topic(item['question'])
        item_topics[item['item_id']] = topic

    # Build Claude's and countries' scores on each dimension
    claude_valid = format_a_df[format_a_df['parsed_value'].notna()].copy()
    claude_means = claude_valid.groupby('item_id')['parsed_value'].mean()

    country_pivot = country_means_df.pivot_table(
        index='country', columns='item_id', values='mean_response'
    )

    # Z-score items across countries (fit on countries only, then transform Claude)
    # This is critical because items have mixed scales (1-4, 1-10, 1-11)
    country_item_means = country_pivot.mean()
    country_item_stds = country_pivot.std()
    country_pivot_z = (country_pivot - country_item_means) / country_item_stds
    claude_means_z = (claude_means - country_item_means) / country_item_stds

    def compute_dimension_score(item_defs, entity_values, item_topics_map, items_df_local):
        """Compute a dimension score by averaging z-scored item values multiplied by direction."""
        matched_values = []
        for keyword, props in item_defs.items():
            # Use exact match first, then fall back to substring match
            # This prevents 'Religion' from matching 'The only acceptable religion is...'
            # when the exact 'Religion' topic exists
            matched_id = None
            fallback_id = None
            for item_id, topic in item_topics_map.items():
                if topic.lower() == keyword.lower():
                    matched_id = item_id
                    break
                elif fallback_id is None and keyword.lower() in topic.lower():
                    fallback_id = item_id
            item_id = matched_id if matched_id is not None else fallback_id
            if item_id is not None and item_id in entity_values.index and not np.isnan(entity_values[item_id]):
                val = entity_values[item_id] * props['direction']
                matched_values.append(val)
        return np.mean(matched_values) if matched_values else np.nan

    # Compute for Claude (using z-scored values)
    claude_trad = compute_dimension_score(trad_secular_items, claude_means_z, item_topics, items_df)
    claude_surv = compute_dimension_score(surv_selfexp_items, claude_means_z, item_topics, items_df)

    # Compute for all countries (using z-scored values)
    records = []
    for country in country_pivot_z.index:
        country_series = country_pivot_z.loc[country]
        trad = compute_dimension_score(trad_secular_items, country_series, item_topics, items_df)
        surv = compute_dimension_score(surv_selfexp_items, country_series, item_topics, items_df)
        if not np.isnan(trad) and not np.isnan(surv):
            records.append({
                'entity': country,
                'traditional_secular': trad,
                'survival_selfexpression': surv,
                'zone': get_zone(country),
            })

    # Add Claude
    records.append({
        'entity': 'CLAUDE',
        'traditional_secular': claude_trad,
        'survival_selfexpression': claude_surv,
        'zone': 'CLAUDE',
    })

    df = pd.DataFrame(records)

    print(f"\n=== INGLEHART-WELZEL CANONICAL DIMENSIONS ===")
    print(f"Entities: {len(df)}")
    claude_row = df[df['entity'] == 'CLAUDE'].iloc[0]
    print(f"Claude: Traditional/Secular={claude_row['traditional_secular']:.3f}, "
          f"Survival/SelfExpression={claude_row['survival_selfexpression']:.3f}")

    # Nearest countries on these dimensions
    countries_only = df[df['entity'] != 'CLAUDE'].copy()
    countries_only['dist_to_claude'] = np.sqrt(
        (countries_only['traditional_secular'] - claude_row['traditional_secular'])**2 +
        (countries_only['survival_selfexpression'] - claude_row['survival_selfexpression'])**2
    )
    nearest = countries_only.nsmallest(10, 'dist_to_claude')
    print(f"\nNearest countries on IW dimensions:")
    for _, r in nearest.iterrows():
        print(f"  {r['entity']:25s}  dist={r['dist_to_claude']:.3f}  "
              f"TS={r['traditional_secular']:.3f}  SS={r['survival_selfexpression']:.3f}")

    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tables" / "inglehart_welzel_scores.csv", index=False)

    return df


# ============================================================
# Run all analyses
# ============================================================

def run_all():
    """Run all analyses on existing results."""
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    country_means = load_country_means()

    # Format A
    try:
        format_a = load_format_a_results()
        print(f"Loaded {len(format_a)} Format A results")

        corr_df = cultural_clustering(format_a, country_means)
        cultural_map(format_a, country_means)
        distribution_overlap_plots(format_a, country_means)
        domain_analysis(format_a, country_means)

    except FileNotFoundError:
        print("Format A results not found. Run the experiment first.")
        format_a = None

    # Additional Format A analyses
    if format_a is not None:
        jsd_analysis(format_a, country_means)
        bootstrap_extremity_analysis(format_a, country_means)
        gap_statistic_analysis(format_a, country_means)
        inglehart_welzel_dimensions(format_a, country_means)

    # Format B — use ONLY the fixed-prompts file
    try:
        format_b_coded = load_format_b_coded()
        # Load raw Format B responses from the specific fixed-prompts file
        format_b_file = RESULTS_DIR / "raw_responses" / "responses_20260320_192138.jsonl"
        if format_b_file.exists():
            format_b_responses = []
            with open(format_b_file) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get('format') == 'B':
                        format_b_responses.append(r)
            format_b_responses_df = pd.DataFrame(format_b_responses)
            print(f"Loaded {len(format_b_responses_df)} Format B responses from {format_b_file.name}")

            steerability_analysis(format_b_coded, format_b_responses_df, country_means)
            steerability_direction_analysis(format_b_coded, format_b_responses_df, country_means)
            steerability_effect_sizes(format_b_coded, format_b_responses_df)
            refusal_analysis(format_a, format_b_coded, format_b_responses_df)
        else:
            print(f"Format B responses file not found: {format_b_file}")

    except FileNotFoundError:
        print("Format B coded results not found. Run LLM coding first.")


if __name__ == "__main__":
    run_all()
