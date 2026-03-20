"""
Publication-quality figures for Claude Constitution Culture Eval.

Style targets: clean, minimal, academic (comparable to Atari et al. 2023,
Tao et al. 2024 PNAS Nexus figures).
"""

import json
import ast
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# ── Style setup ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.2,
})

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Inglehart-Welzel zones and colors (muted, publication-friendly palette)
CULTURAL_ZONES = {
    'Protestant Europe': [
        'Sweden', 'Denmark', 'Norway', 'Finland', 'Iceland',
        'Germany', 'Netherlands', 'Switzerland',
    ],
    'English-Speaking': [
        'United States', 'Australia', 'New Zealand', 'Canada',
        'Great Britain', 'United Kingdom', 'Ireland',
    ],
    'Catholic Europe': [
        'France', 'Spain', 'Italy', 'Portugal', 'Belgium',
        'Austria', 'Poland', 'Czech Republic', 'Slovakia',
        'Hungary', 'Croatia', 'Slovenia', 'Lithuania',
    ],
    'Confucian': [
        'Japan', 'South Korea', 'China', 'Taiwan',
        'Hong Kong SAR', 'Vietnam', 'Singapore',
    ],
    'South Asia': [
        'India', 'Bangladesh', 'Pakistan', 'Sri Lanka',
        'Nepal', 'Myanmar',
    ],
    'African-Islamic': [
        'Nigeria', 'Egypt', 'Morocco', 'Ethiopia', 'Kenya',
        'Tanzania', 'Ghana', 'Zimbabwe', 'Tunisia', 'Libya',
        'Algeria', 'Iran', 'Iraq', 'Jordan', 'Lebanon',
        'Turkey', 'Indonesia', 'Malaysia', 'Saudi Arabia',
    ],
    'Orthodox Europe': [
        'Russia', 'Ukraine', 'Serbia', 'Romania', 'Bulgaria',
        'Georgia', 'Armenia', 'Belarus', 'Moldova',
        'Greece', 'Cyprus', 'North Macedonia', 'Montenegro', 'Bosnia Herzegovina',
    ],
    'Latin America': [
        'Brazil', 'Mexico', 'Argentina', 'Colombia', 'Chile',
        'Peru', 'Venezuela', 'Ecuador', 'Bolivia', 'Guatemala',
        'Nicaragua', 'Puerto Rico', 'Dominican Republic',
    ],
}

ZONE_COLORS = {
    'Protestant Europe': '#4477AA',
    'English-Speaking': '#EE6677',
    'Catholic Europe': '#228833',
    'Confucian': '#CCBB44',
    'South Asia': '#AA3377',
    'African-Islamic': '#66CCEE',
    'Orthodox Europe': '#BBBBBB',
    'Latin America': '#CC6633',
    'Other': '#999999',
}

ZONE_MARKERS = {
    'Protestant Europe': 'o',
    'English-Speaking': 's',
    'Catholic Europe': 'D',
    'Confucian': '^',
    'South Asia': 'v',
    'African-Islamic': 'P',
    'Orthodox Europe': 'X',
    'Latin America': 'p',
    'Other': 'h',
}


def get_zone(country: str) -> str:
    for zone, countries in CULTURAL_ZONES.items():
        if country in countries:
            return zone
    return 'Other'


def load_data():
    """Load all needed data."""
    country_means = pd.read_csv(DATA_DIR / "processed" / "country_means.csv")
    format_a = pd.read_csv(RESULTS_DIR / "coded" / "format_a_parsed.csv")
    items = pd.read_csv(DATA_DIR / "processed" / "selected_items.csv")
    return country_means, format_a, items


# ════════════════════════════════════════════════════════════
# Figure 1: Cultural Map (UMAP + k-means) — THE HERO FIGURE
# ════════════════════════════════════════════════════════════

def fig_cultural_map(format_a_df, country_means_df):
    """Inglehart-Welzel style cultural map using UMAP projection with k-means clusters in high-D space."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from matplotlib.patches import Ellipse

    # Build vectors
    claude_vector = format_a_df[format_a_df['parsed_value'].notna()].groupby('item_id')['parsed_value'].mean()
    country_pivot = country_means_df.pivot_table(index='country', columns='item_id', values='mean_response')
    common_items = sorted(set(claude_vector.index) & set(country_pivot.columns))

    claude_vec = claude_vector[common_items]
    valid = country_pivot[common_items].dropna(thresh=len(common_items) * 0.7)
    valid = valid.fillna(valid.mean())

    # Standardize on countries only
    scaler = StandardScaler()
    valid_z = pd.DataFrame(scaler.fit_transform(valid.values), index=valid.index, columns=valid.columns)
    claude_z_row = pd.DataFrame(
        scaler.transform(claude_vec.values.reshape(1, -1)),
        index=['CLAUDE'], columns=valid_z.columns
    )
    combined_z = pd.concat([valid_z, claude_z_row])

    # UMAP projection (fit on countries only, then transform Claude)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    country_coords = reducer.fit_transform(valid_z.values)
    claude_umap = reducer.transform(claude_z_row.values)

    coords_df = pd.DataFrame(
        np.vstack([country_coords, claude_umap]),
        index=list(valid_z.index) + ['CLAUDE'],
        columns=['Dim1', 'Dim2']
    )

    # k-means clusters in HIGH-DIMENSIONAL space (not 2D), k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    country_clusters = kmeans.fit_predict(valid_z.values)
    cluster_labels = {country: int(c) for country, c in zip(valid_z.index, country_clusters)}
    cluster_labels['CLAUDE'] = -1  # Claude gets no cluster

    # Assign cluster colors (map cluster IDs to distinguishable colors)
    CLUSTER_COLORS = {0: '#4477AA', 1: '#EE6677', 2: '#228833', 3: '#CCBB44'}
    coords_df['cluster'] = coords_df.index.map(lambda x: cluster_labels.get(x, -1))
    coords_df['zone'] = coords_df.index.map(lambda x: 'CLAUDE' if x == 'CLAUDE' else get_zone(x))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 10))

    zone_order = ['Protestant Europe', 'English-Speaking', 'Catholic Europe',
                  'Confucian', 'South Asia', 'African-Islamic', 'Orthodox Europe',
                  'Latin America']

    # 1) Clean ellipse outlines only (no fill) — by k-means cluster
    for cluster_id in range(4):
        zone_data = coords_df[coords_df['cluster'] == cluster_id]
        if len(zone_data) < 3:
            continue
        color = CLUSTER_COLORS.get(cluster_id, '#999')
        pts = zone_data[['Dim1', 'Dim2']].values

        mean_x, mean_y = pts[:, 0].mean(), pts[:, 1].mean()
        cov = np.cov(pts[:, 0], pts[:, 1])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        chi2_val = 5.991
        width = 2 * np.sqrt(chi2_val * max(eigenvalues[0], 0.01))
        height = 2 * np.sqrt(chi2_val * max(eigenvalues[1], 0.01))

        ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                          facecolor=color, alpha=0.06, edgecolor=color,
                          linewidth=2.0, linestyle='-', zorder=0)
        ax.add_patch(ellipse)

    # 2) Every country as a colored label box (no dots, like Atari et al.)
    #    Skip only a handful of very minor territories
    skip_countries = {
        'Macao SAR', 'Andorra', 'Maldives', 'Montenegro',
        'North Macedonia', 'Moldova', 'Belarus', 'Bosnia Herzegovina',
        'Northern Ireland', 'Puerto Rico', 'Taiwan ROC', 'Hong Kong SAR',
        'Dominican Republic', 'Nicaragua', 'Guatemala', 'Bolivia',
        'Ecuador', 'Venezuela', 'Mongolia', 'Cyprus', 'Armenia',
        'Iceland', 'Serbia', 'Slovakia', 'Croatia', 'Slovenia',
        'Lithuania', 'Sri Lanka', 'Nepal', 'Kenya', 'Tanzania',
        'Ghana', 'Zimbabwe', 'Algeria', 'Libya', 'Tajikistan',
        'Kyrgyzstan', 'Kazakhstan', 'Myanmar',
    }

    texts = []
    for country in coords_df.index:
        if country == 'CLAUDE' or country in skip_countries:
            continue
        x, y = coords_df.loc[country, ['Dim1', 'Dim2']]
        c_id = coords_df.loc[country, 'cluster']
        color = CLUSTER_COLORS.get(c_id, '#999')

        txt = ax.text(x, y, country, fontsize=8, ha='center', va='center',
                      color=color, fontweight='bold', zorder=5,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor=color, linewidth=0.6, alpha=0.9))
        texts.append(txt)

    # Note: adjustText disabled — causes canvas blowup with bbox labels.
    # Labels overlap slightly in dense areas; acceptable for publication.

    # 3) Claude — bold black label
    cx, cy = coords_df.loc['CLAUDE', ['Dim1', 'Dim2']]
    ax.text(cx, cy, ' Claude ', fontsize=13, ha='center', va='center',
            color='white', fontweight='bold', zorder=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#1a1a1a',
                      edgecolor='#000000', linewidth=2, alpha=0.95))

    # Set axis limits with padding
    all_x = coords_df['Dim1'].values
    all_y = coords_df['Dim2'].values
    pad = 2.5
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_clip_on(True)

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Claude on the Global Cultural Map', fontsize=14, fontweight='bold', pad=14)

    # Clip all artists to axes
    for txt in texts:
        txt.set_clip_on(True)

    plt.savefig(FIGURES_DIR / "fig1_cultural_map.pdf", format='pdf',
                dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig1_cultural_map.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Saved fig1_cultural_map")
    return coords_df


# ════════════════════════════════════════════════════════════
# Figure 2: Country Correlation Rankings
# ════════════════════════════════════════════════════════════

def fig_correlations(format_a_df, country_means_df):
    """Horizontal bar chart of country correlations with Claude, colored by zone."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    claude_vector = format_a_df[format_a_df['parsed_value'].notna()].groupby('item_id')['parsed_value'].mean()
    country_pivot = country_means_df.pivot_table(index='country', columns='item_id', values='mean_response')
    common_items = sorted(set(claude_vector.index) & set(country_pivot.columns))
    claude_vec = claude_vector[common_items].values
    valid = country_pivot[common_items].dropna(thresh=len(common_items) * 0.7)
    valid = valid.fillna(valid.mean())

    # Compute correlations
    records = []
    for country in valid.index:
        r, p = stats.pearsonr(claude_vec, valid.loc[country].values)
        records.append({'country': country, 'r': r, 'p': p, 'zone': get_zone(country)})
    corr_df = pd.DataFrame(records).sort_values('r', ascending=True)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6, 14))

    colors = [ZONE_COLORS.get(z, '#999') for z in corr_df['zone']]
    y_pos = range(len(corr_df))

    bars = ax.barh(y_pos, corr_df['r'], color=colors, alpha=0.85, height=0.75, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_df['country'], fontsize=6.5)
    ax.set_xlabel('Pearson $r$ with Claude', fontsize=11)
    ax.set_xlim(0, 1)
    ax.axvline(x=corr_df['r'].median(), color='black', linewidth=0.6, linestyle=':', alpha=0.5)

    # Add zone legend
    legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c,
                              markersize=8, label=z)
                       for z, c in ZONE_COLORS.items() if z != 'Other']
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              frameon=True, framealpha=0.95, edgecolor='#cccccc', ncol=1)

    ax.set_title("Cultural Similarity to Claude's Values", fontsize=12, fontweight='bold', pad=10)
    plt.savefig(FIGURES_DIR / "fig2_correlations.pdf", format='pdf')
    plt.savefig(FIGURES_DIR / "fig2_correlations.png", dpi=300)
    plt.close()
    print("Saved fig2_correlations")


# ════════════════════════════════════════════════════════════
# Figure 3: Hierarchical Clustering Dendrogram
# ════════════════════════════════════════════════════════════

def fig_dendrogram(format_a_df, country_means_df):
    """Dendrogram with Claude highlighted."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    claude_vector = format_a_df[format_a_df['parsed_value'].notna()].groupby('item_id')['parsed_value'].mean()
    country_pivot = country_means_df.pivot_table(index='country', columns='item_id', values='mean_response')
    common_items = sorted(set(claude_vector.index) & set(country_pivot.columns))
    claude_vec = claude_vector[common_items].values
    valid = country_pivot[common_items].dropna(thresh=len(common_items) * 0.7)
    valid = valid.fillna(valid.mean())

    # Standardize on countries only
    scaler = StandardScaler()
    countries_z = pd.DataFrame(scaler.fit_transform(valid.values), index=valid.index, columns=valid.columns)
    claude_z_val = scaler.transform(claude_vec.reshape(1, -1))
    combined_z = pd.concat([countries_z,
                            pd.DataFrame(claude_z_val, index=['Claude'], columns=valid.columns)])

    condensed = pdist(combined_z.values, metric='euclidean')
    Z = linkage(condensed, method='ward')

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(22, 9))

    # Let dendrogram auto-color branches by cluster
    # Choose a threshold that gives ~6-8 clusters for visual clarity
    max_d = Z[-4, 2]  # cut at 4th-from-last merge for ~5 clusters
    dn = dendrogram(Z, labels=list(combined_z.index), ax=ax,
                    leaf_rotation=90, leaf_font_size=8.5,
                    color_threshold=max_d,
                    above_threshold_color='#AAAAAA')

    # All labels black; bold Claude
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color('#1a1a1a')
        if lbl.get_text() == 'Claude':
            lbl.set_fontweight('bold')
            lbl.set_fontsize(11)
        else:
            lbl.set_fontsize(8.5)

    ax.set_ylabel('Euclidean distance (standardized)', fontsize=12)
    ax.set_title('Hierarchical Clustering of Value Positions', fontsize=13, fontweight='bold', pad=12)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES_DIR / "fig3_dendrogram.pdf", format='pdf')
    plt.savefig(FIGURES_DIR / "fig3_dendrogram.png", dpi=300)
    plt.close()
    print("Saved fig3_dendrogram")


# ════════════════════════════════════════════════════════════
# Figure 4: Distribution Overlap (most divisive items)
# ════════════════════════════════════════════════════════════

def fig_distributions(format_a_df, country_means_df, items_df):
    """Response distributions for the most culturally divisive items."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    top_items = items_df.nlargest(6, 'cross_country_var')

    # Use 4 maximally contrasting countries (wider bars, cleaner look)
    contrast_countries = ['Sweden', 'United States', 'Nigeria', 'Egypt']
    country_colors = {
        'Sweden': '#4477AA', 'United States': '#EE6677',
        'Nigeria': '#66CCEE', 'Egypt': '#228833',
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    full_items = json.loads(
        (Path(__file__).parent.parent / "data" / "processed" / "selected_items_full.json").read_text()
    )

    SKIP = {"Don't know", "No answer", "Missing; Not available",
            "Missing; Not applicable for other reasons", "No answer/refused",
            "Other missing; Multiple answers Mail (EVS)", "Missing; Unknown",
            "Missing; Not asked in survey", "DK/Refused"}

    for idx, (_, item) in enumerate(top_items.iterrows()):
        ax = axes[idx // 3, idx % 3]
        item_id = item['item_id']
        options = ast.literal_eval(item['options'])
        substantive = [o for o in options if o not in SKIP]
        n_opts = len(substantive)

        # Claude's response
        claude_resp = format_a_df[
            (format_a_df['item_id'] == item_id) & (format_a_df['parsed_value'].notna())
        ]['parsed_value'].values

        # Country distributions
        full_item = [fi for fi in full_items if fi['item_id'] == item_id]
        if not full_item:
            continue
        sel_str = full_item[0]['selections']
        selections = ast.literal_eval(
            sel_str.replace("defaultdict(<class 'list'>, ", "").rstrip(")")
        ) if isinstance(sel_str, str) and 'defaultdict' in sel_str else sel_str

        x = np.arange(n_opts)
        n_countries = len(contrast_countries)
        width = 0.18

        for c_idx, country in enumerate(contrast_countries):
            if country not in selections:
                continue
            dist = selections[country][:n_opts]
            s = sum(dist)
            if s > 0:
                normalized = [p / s for p in dist]
                offset = (c_idx - (n_countries - 1) / 2) * width
                ax.bar(x + offset, normalized, width,
                       label=country if idx == 0 else None,
                       color=country_colors[country], alpha=0.8,
                       edgecolor='white', linewidth=0.3)

        # Claude's response — shaded region instead of just a line
        if len(claude_resp) > 0:
            cv = claude_resp[0] - 1  # convert to 0-indexed
            ax.axvspan(cv - 0.4, cv + 0.4, alpha=0.12, color='#1a1a1a', zorder=0)
            ax.axvline(x=cv, color='#1a1a1a', linewidth=2.5, linestyle='--',
                       alpha=0.9, label='Claude' if idx == 0 else None, zorder=5)

        # Topic label
        topic = item['question'].strip().split('\n')[-1].strip()
        if len(topic) > 40:
            topic = topic[:37] + '...'
        ax.set_title(topic, fontsize=10, fontweight='bold', pad=6)

        # X-axis: for 10-point scales, just show endpoints
        if n_opts == 10:
            ax.set_xticks([0, 4, 9])
            ax.set_xticklabels([substantive[0], '5', substantive[-1]], fontsize=7)
        elif n_opts <= 5:
            ax.set_xticks(x)
            ax.set_xticklabels(substantive, fontsize=7, rotation=30, ha='right')
        else:
            ax.set_xticks([0, n_opts - 1])
            ax.set_xticklabels([substantive[0], substantive[-1]], fontsize=7)

        ax.set_ylim(0, max(0.75, ax.get_ylim()[1] * 1.05))
        if idx % 3 == 0:
            ax.set_ylabel('Proportion', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

    # Single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               frameon=True, framealpha=0.95, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Claude's Position on Culturally Divisive Items",
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout(h_pad=2.0, w_pad=1.5)
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(FIGURES_DIR / "fig4_distributions.pdf", format='pdf')
    plt.savefig(FIGURES_DIR / "fig4_distributions.png", dpi=300)
    plt.close()
    print("Saved fig4_distributions")


# ════════════════════════════════════════════════════════════
# Figure 5: Domain Comparison (Claude vs US vs Global)
# ════════════════════════════════════════════════════════════

def fig_domain_comparison(format_a_df, country_means_df, items_df):
    """Dot plot comparing Claude to US and global mean by domain."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    item_domains = items_df.set_index('item_id')['domain'].to_dict()
    claude_means = format_a_df[format_a_df['parsed_value'].notna()].groupby('item_id')['parsed_value'].mean()
    us_means = country_means_df[country_means_df['country'] == 'United States'].set_index('item_id')['mean_response']
    global_means = country_means_df.groupby('item_id')['mean_response'].mean()

    comparison = pd.DataFrame({
        'Claude': claude_means, 'United States': us_means, 'Global Mean': global_means,
    }).dropna()
    comparison['domain'] = comparison.index.map(item_domains)

    domains = ['moral_justifiability', 'gender_family', 'religion_values',
               'tolerance_neighbors', 'political_authority', 'economic_values']
    domain_labels = {
        'moral_justifiability': 'Moral\nJustifiability',
        'gender_family': 'Gender &\nFamily',
        'religion_values': 'Religion &\nValues',
        'tolerance_neighbors': 'Tolerance',
        'political_authority': 'Political\nAuthority',
        'economic_values': 'Economic\nValues',
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=False)

    for idx, domain in enumerate(domains):
        ax = axes[idx // 3, idx % 3]
        d = comparison[comparison['domain'] == domain].copy()
        if len(d) == 0:
            ax.set_visible(False)
            continue

        d = d.sort_values('Claude')
        items_in_domain = range(len(d))

        ax.scatter(d['Global Mean'], items_in_domain, c='#BBBBBB', s=40, zorder=2,
                   marker='o', alpha=0.8, label='Global Mean' if idx == 0 else None)
        ax.scatter(d['United States'], items_in_domain, c='#4477AA', s=40, zorder=3,
                   marker='s', alpha=0.8, label='United States' if idx == 0 else None)
        ax.scatter(d['Claude'], items_in_domain, c='#1a1a1a', s=70, zorder=4,
                   marker='*', label='Claude' if idx == 0 else None)

        # Connect with lines
        for j, (item_id, row) in enumerate(d.iterrows()):
            ax.plot([row['Global Mean'], row['Claude']], [j, j],
                    color='#dddddd', linewidth=0.8, zorder=1)

        ax.set_title(domain_labels.get(domain, domain), fontsize=10, fontweight='bold', pad=8)
        ax.set_yticks([])
        ax.set_xlabel('Mean Response', fontsize=9)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10,
               frameon=True, framealpha=0.95, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Claude's Value Positions by Domain",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(h_pad=2.0, w_pad=1.5)
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(FIGURES_DIR / "fig5_domain_comparison.pdf", format='pdf')
    plt.savefig(FIGURES_DIR / "fig5_domain_comparison.png", dpi=300)
    plt.close()
    print("Saved fig5_domain_comparison")


# ════════════════════════════════════════════════════════════
# Generate all figures
# ════════════════════════════════════════════════════════════

def generate_all():
    country_means, format_a, items = load_data()
    fig_cultural_map(format_a, country_means)
    fig_correlations(format_a, country_means)
    fig_dendrogram(format_a, country_means)
    fig_distributions(format_a, country_means, items)
    fig_domain_comparison(format_a, country_means, items)
    print("\n=== All publication figures generated ===")


if __name__ == "__main__":
    generate_all()
