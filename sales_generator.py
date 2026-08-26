"""Weekly sales history generator for MobiMart.

This module simulates planning data rather than pretending to be a source of
actual sales. The formulas are deliberately visible so an interviewer can
change an assumption and explain how it affects allocation decisions.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from catalog import generate_catalog
from stores import generate_stores


START_DATE = pd.Timestamp("2025-09-01")
WEEKS_IN_HISTORY = 52
FESTIVE_DATES = {
    pd.Timestamp("2025-10-02"): "Dussehra 2025",
    pd.Timestamp("2025-10-20"): "Diwali 2025",
    pd.Timestamp("2026-10-20"): "Dussehra 2026",
    pd.Timestamp("2026-11-08"): "Diwali 2026",
}


def _week_contains(week_start: pd.Timestamp, event_date: pd.Timestamp) -> bool:
    """Return whether a Monday-starting week contains a festival date."""
    return week_start <= event_date <= week_start + pd.Timedelta(days=6)


def festive_multiplier(week_start: pd.Timestamp) -> float:
    """Return a 3-4x festive uplift for weeks containing Dussehra or Diwali."""
    if any(_week_contains(week_start, date) for date in FESTIVE_DATES):
        return 3.5
    return 1.0


def lifecycle_multiplier(
    week_start: pd.Timestamp,
    launch_date: pd.Timestamp,
    expected_lifecycle_weeks: int,
    successor_launch_date: Optional[pd.Timestamp] = None,
) -> float:
    """Estimate demand by model age, including a successor shock.

    Demand starts below its mature peak, reaches maximum around the configured
    lifecycle week, then decays. A replacement launched during the four-week
    post-launch window causes an additional sharp drop as shoppers and staff
    move attention to the newer model.
    """
    age_weeks = (week_start - launch_date).days / 7
    if age_weeks < 0:
        return 0.0

    peak_week = min(10, max(8, round(expected_lifecycle_weeks * 0.30)))
    if age_weeks <= peak_week:
        multiplier = 0.35 + (0.65 * age_weeks / peak_week)
    else:
        multiplier = float(np.exp(-0.055 * (age_weeks - peak_week)))

    if successor_launch_date is not None:
        weeks_after_successor = (week_start - successor_launch_date).days / 7
        if 0 <= weeks_after_successor < 4:
            multiplier *= max(0.08, 0.55 - (0.12 * weeks_after_successor))
    return multiplier


def _fit_multiplier(sales_mix_profile: str, price_tier: str) -> float:
    """Map the Stage 1 store profile to relative category demand."""
    fit = {
        "flagship-heavy": {"budget": 0.45, "mid": 0.85, "flagship": 1.55},
        "mid-range": {"budget": 0.75, "mid": 1.35, "flagship": 0.70},
        "budget-heavy": {"budget": 1.55, "mid": 0.80, "flagship": 0.30},
    }
    return fit[sales_mix_profile][price_tier]


def _base_weekly_demand(footfall_index: int, sales_mix_profile: str, price_tier: str) -> float:
    """Create a plausible store-model demand level before lifecycle effects."""
    category_baseline = {"budget": 4.0, "mid": 2.4, "flagship": 0.8}
    return (footfall_index / 100) * category_baseline[price_tier] * _fit_multiplier(
        sales_mix_profile, price_tier
    )


def _successor_launches(catalog: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    """Index model launches for fast predecessor lookups."""
    return dict(zip(catalog["model_id"], pd.to_datetime(catalog["launch_date"])))


def generate_sales_history(
    stores: Optional[pd.DataFrame] = None,
    catalog: Optional[pd.DataFrame] = None,
    start_date: pd.Timestamp = START_DATE,
    weeks: int = WEEKS_IN_HISTORY,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 52 weeks of store-model sales at weekly granularity.

    A seeded lognormal noise term captures local execution, promotions, and
    customer variation while keeping reruns reproducible. Poisson sampling
    then turns expected demand into integer unit sales, which is more realistic
    for low-volume flagship models than rounding a smooth formula.
    """
    stores = generate_stores() if stores is None else stores
    catalog = generate_catalog() if catalog is None else catalog
    rng = np.random.default_rng(seed)
    launch_dates = pd.to_datetime(catalog["launch_date"])
    launch_by_model = _successor_launches(catalog)
    weeks_to_generate = pd.date_range(pd.Timestamp(start_date), periods=weeks, freq="7D")
    records = []

    for store in stores.itertuples(index=False):
        for model in catalog.itertuples(index=False):
            successor_launch = (
                launch_by_model.get(model.successor_model_id)
                if pd.notna(model.successor_model_id)
                else None
            )
            base_demand = _base_weekly_demand(
                store.footfall_index, store.sales_mix_profile, model.price_tier
            )
            model_launch = pd.Timestamp(model.launch_date)
            for week_start in weeks_to_generate:
                lifecycle = lifecycle_multiplier(
                    week_start,
                    model_launch,
                    model.expected_lifecycle_weeks,
                    successor_launch,
                )
                festive = festive_multiplier(week_start)
                expected_units = base_demand * lifecycle * festive
                noisy_demand = expected_units * rng.lognormal(mean=0, sigma=0.16)
                units_sold = int(rng.poisson(noisy_demand)) if noisy_demand > 0 else 0
                records.append(
                    {
                        "week_start_date": week_start,
                        "store_id": store.store_id,
                        "model_id": model.model_id,
                        "units_sold": units_sold,
                            "unit_price": model.selling_price,
                            "revenue": units_sold * model.selling_price,
                    }
                )

    return pd.DataFrame(records)


def _festival_mask(sales_history: pd.DataFrame) -> pd.Series:
    """Identify generated rows whose week contains a configured festival."""
    return sales_history["week_start_date"].map(
        lambda date: festive_multiplier(pd.Timestamp(date)) > 1
    )


def print_sanity_checks(
    sales_history: pd.DataFrame,
    stores: Optional[pd.DataFrame] = None,
    catalog: Optional[pd.DataFrame] = None,
) -> None:
    """Print summaries that expose whether the simulation behaves realistically."""
    stores = generate_stores() if stores is None else stores
    catalog = generate_catalog() if catalog is None else catalog
    enriched = sales_history.merge(stores[["store_id", "city_tier"]], on="store_id")
    tier_summary = enriched.groupby("city_tier").agg(
        total_units=("units_sold", "sum"), total_revenue=("revenue", "sum")
    )
    print("\nTotal units and revenue by city tier:")
    print(tier_summary.to_string())

    flagship_catalog = catalog.loc[catalog["price_tier"].eq("flagship")].copy()
    flagship_catalog["launch_date"] = pd.to_datetime(flagship_catalog["launch_date"])
    in_window = flagship_catalog.loc[flagship_catalog["launch_date"] >= START_DATE]
    sample_model = in_window.sort_values("launch_date").iloc[0]
    sample = sales_history.loc[sales_history["model_id"].eq(sample_model.model_id)]
    sample_weekly = sample.groupby("week_start_date")["units_sold"].sum()
    print(f"\nSample flagship lifecycle ({sample_model.model_id} - {sample_model.model_name}):")
    print(sample_weekly.to_string())

    festive = _festival_mask(sales_history)
    festive_units = sales_history.loc[festive, "units_sold"].mean()
    normal_units = sales_history.loc[~festive, "units_sold"].mean()
    observed_ratio = festive_units / normal_units if normal_units else float("nan")
    print(
        f"\nFestive uplift: actual generated {observed_ratio:.2f}x normal "
        f"(coded multiplier {festive_multiplier(START_DATE + pd.Timedelta(days=31)):.1f}x)"
    )

    predecessor = catalog.loc[catalog["successor_model_id"].notna()].iloc[0]
    successor_launch = _successor_launches(catalog)[predecessor.successor_model_id]
    predecessor_sales = sales_history.loc[sales_history["model_id"].eq(predecessor.model_id)]
    before = predecessor_sales["week_start_date"].between(
        successor_launch - pd.Timedelta(weeks=4), successor_launch - pd.Timedelta(days=1)
    )
    after = predecessor_sales["week_start_date"].between(
        successor_launch, successor_launch + pd.Timedelta(weeks=3)
    )
    before_avg = predecessor_sales.loc[before, "units_sold"].mean()
    after_avg = predecessor_sales.loc[after, "units_sold"].mean()
    print(
        f"\nCannibalization ({predecessor.model_id} after {predecessor.successor_model_id} launch "
        f"on {successor_launch.date()}): {before_avg:.2f} units/week before vs "
        f"{after_avg:.2f} after"
    )


if __name__ == "__main__":
    stores = generate_stores()
    catalog = generate_catalog()
    history = generate_sales_history(stores, catalog)
    print("Generated rows:", len(history))
    print_sanity_checks(history, stores, catalog)