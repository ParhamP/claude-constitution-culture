"""
Item Selection for Claude Constitution Culture Eval

Selects 55 WVS items from the Anthropic GlobalOpinionQA dataset,
ranked by cross-country variance and mapped to constitutional principle domains.
"""

import ast
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Constitutional principle domains and the WVS question keywords that map to them
DOMAIN_KEYWORDS = {
    "moral_justifiability": [
        "always be justified, never be justified",
    ],
    "gender_family": [
        "men make better political",
        "men make better business",
        "university education is more important for a boy",
        "problem if women have more income",
        "woman should not work" ,
        "duty towards society to have children",
        "child's duty to take care",
        "childs duty to take care",
        "homosexual couples are as good parents",
        "woman earns more",
        "pre-school child suffers",
    ],
    "religion_values": [
        "important it is in your life",  # importance of religion
        "only acceptable religion is my religion",
        "religion conflict, religion is always right",
        "believe in",  # hell, heaven, god
        "how important is God in your life",
    ],
    "tolerance_neighbors": [
        "would not like to have as neighbors",
    ],
    "political_authority": [
        "types of political systems",
        "essential you think it is as a characteristic of democracy",
        "how democratically is this country",
        "how important is it for you to live in a country that is governed democratically",
        "political system is functioning",
        "elections",
        "political action",
        "corruption",
        "bribe",
    ],
    "economic_values": [
        "incomes should be made more equal",
        "government should take more responsibility",
        "private ownership of business",
        "competition is good",
        "hard work usually brings a better life",
        "work should always come first",
        "people who don't work turn lazy",
        "financial situation of your household",
        "science and technology",
        "left\" and \"the right",
        "moral rules are the right ones",
        "immigrants",
        "immigration",
        "government should or should not have the right",
    ],
}


def parse_selections(s: str) -> dict:
    """Parse the defaultdict string representation from the dataset."""
    s = s.replace("defaultdict(<class 'list'>, ", "").rstrip(")")
    return ast.literal_eval(s)


def parse_options(s: str) -> list:
    return ast.literal_eval(s)


SKIP_OPTIONS = {
    "Don't know", "No answer", "Missing; Not available",
    "Missing; Not applicable for other reasons", "No answer/refused",
    "Other missing; Multiple answers Mail (EVS)", "Missing; Unknown",
    "Missing; Not asked in survey", "DK/Refused",
}


def compute_cross_country_variance(row: pd.Series) -> dict:
    """Compute cross-country variance of mean response for a WVS item."""
    sel = parse_selections(row['selections'])
    options = parse_options(row['options'])

    n_substantive = len([o for o in options if o not in SKIP_OPTIONS])
    countries = list(sel.keys())

    country_means = {}
    for country, dist in sel.items():
        sub_dist = dist[:n_substantive]
        s = sum(sub_dist)
        if s > 0.5:
            normalized = [p / s for p in sub_dist]
            mean = sum((j + 1) * p for j, p in enumerate(normalized))
            country_means[country] = mean

    if len(country_means) < 5:
        return None

    means = list(country_means.values())
    return {
        'cross_country_var': np.var(means),
        'cross_country_std': np.std(means),
        'n_countries': len(country_means),
        'n_options': n_substantive,
        'mean_of_means': np.mean(means),
        'country_means': country_means,
    }


def assign_domain(question: str) -> str:
    """Assign a WVS item to a constitutional principle domain."""
    q_lower = question.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in q_lower:
                return domain
    return "other"


def select_items(target_per_domain: int = 10, min_countries: int = 10) -> pd.DataFrame:
    """
    Select WVS items by:
    1. Assigning each to a constitutional principle domain
    2. Within each domain, ranking by cross-country variance
    3. Selecting top items per domain
    """
    print("Loading GlobalOpinionQA dataset...")
    ds = load_dataset('Anthropic/llm_global_opinions')
    df = ds['train'].to_pandas()
    wvs = df[df['source'] == 'WVS'].copy().reset_index(drop=True)
    print(f"Total WVS items: {len(wvs)}")

    # Compute variance and assign domains
    records = []
    for i, row in wvs.iterrows():
        stats = compute_cross_country_variance(row)
        if stats is None:
            continue
        if stats['n_countries'] < min_countries:
            continue

        domain = assign_domain(row['question'])

        records.append({
            'original_idx': i,
            'question': row['question'],
            'options': row['options'],
            'selections': row['selections'],
            'domain': domain,
            **{k: v for k, v in stats.items() if k != 'country_means'},
        })

    items_df = pd.DataFrame(records)

    # Print domain distribution
    print(f"\nItems by domain (before selection):")
    for domain in sorted(items_df['domain'].unique()):
        count = len(items_df[items_df['domain'] == domain])
        print(f"  {domain}: {count}")

    # Select top items per domain by variance
    selected = []
    for domain in DOMAIN_KEYWORDS.keys():
        domain_items = items_df[items_df['domain'] == domain].sort_values(
            'cross_country_var', ascending=False
        )
        n_select = min(target_per_domain, len(domain_items))
        selected.append(domain_items.head(n_select))
        print(f"\nSelected {n_select} items for domain '{domain}':")
        for _, item in domain_items.head(n_select).iterrows():
            print(f"  [var={item['cross_country_var']:.3f}] {item['question'][:100]}")

    selected_df = pd.concat(selected, ignore_index=True)
    selected_df['item_id'] = range(len(selected_df))

    print(f"\nTotal selected items: {len(selected_df)}")
    return selected_df


def save_selected_items(selected_df: pd.DataFrame):
    """Save selected items and their country-level distributions."""
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save item metadata (without the full selections blob)
    meta_cols = ['item_id', 'domain', 'question', 'options', 'n_countries',
                 'n_options', 'cross_country_var', 'mean_of_means']
    selected_df[meta_cols].to_csv(processed_dir / "selected_items.csv", index=False)

    # Save country-level means for each item
    country_means_records = []
    for _, row in selected_df.iterrows():
        stats = compute_cross_country_variance(row)
        if stats is None:
            continue
        for country, mean in stats['country_means'].items():
            country_means_records.append({
                'item_id': row['item_id'],
                'country': country,
                'mean_response': mean,
            })

    country_means_df = pd.DataFrame(country_means_records)
    country_means_df.to_csv(processed_dir / "country_means.csv", index=False)

    # Also save full selections for later distributional analysis
    selected_df.to_json(processed_dir / "selected_items_full.json", orient='records', indent=2)

    print(f"Saved {len(selected_df)} items to {processed_dir}")
    print(f"Saved {len(country_means_df)} country-mean records")

    return country_means_df


if __name__ == "__main__":
    selected = select_items()
    country_means = save_selected_items(selected)

    # Print summary statistics
    print("\n=== SELECTION SUMMARY ===")
    print(f"Items: {len(selected)}")
    print(f"Domains: {selected['domain'].nunique()}")
    print(f"Countries covered: {country_means['country'].nunique()}")
    print(f"\nVariance range: {selected['cross_country_var'].min():.3f} - {selected['cross_country_var'].max():.3f}")
    print(f"Mean variance: {selected['cross_country_var'].mean():.3f}")
