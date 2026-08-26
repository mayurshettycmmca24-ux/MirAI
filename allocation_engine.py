"""Budget-aware weekly stock allocation for MobiMart."""

from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sales_generator import festive_multiplier


REQUIRED_SALES_COLUMNS = {"week_start_date", "store_id", "model_id", "units_sold"}


def festive_precommitment_budget(
    week_start_date: pd.Timestamp, budget: int = 4_00_00_000
) -> int:
    """Apply a 50% distributor pre-commitment uplift in known festive weeks."""
    return int(budget * 1.5) if festive_multiplier(pd.Timestamp(week_start_date)) > 1 else budget


def _lifecycle_stage(
    week_start_date: pd.Timestamp,
    launch_date: pd.Timestamp,
    lifecycle_weeks: int,
    successor_launch_date: Optional[pd.Timestamp],
) -> Tuple[str, float]:
    """Return a named lifecycle stage and a conservative forecast adjustment."""
    age_weeks = (week_start_date - launch_date).days / 7
    if age_weeks < 0:
        return "pre-launch", 0.0
    if successor_launch_date is not None:
        weeks_to_successor = (successor_launch_date - week_start_date).days / 7
        if 0 <= weeks_to_successor <= 4:
            return "near end-of-life", 0.55
    if age_weeks > lifecycle_weeks:
        return "near end-of-life", 0.50
    if age_weeks <= min(10, max(8, round(lifecycle_weeks * 0.30))):
        if age_weeks < 8:
            return "ramping", 1.15
        return "peaking", 1.05
    return "declining", 0.85


def _normalise_current_stock(
    current_stock: Union[pd.DataFrame, Dict[str, int], Dict[Tuple[str, str], int]]
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    """Accept warehouse-by-model or store-model stock without hiding bad input."""
    by_model: Dict[str, int] = {}
    by_pair: Dict[Tuple[str, str], int] = {}
    if isinstance(current_stock, pd.DataFrame):
        required = {"model_id", "units_available"}
        missing = required - set(current_stock.columns)
        if missing:
            raise ValueError(f"current_stock is missing columns: {sorted(missing)}")
        for row in current_stock.itertuples(index=False):
            if hasattr(row, "store_id"):
                by_pair[(str(row.store_id), str(row.model_id))] = max(0, int(row.units_available))
            else:
                by_model[str(row.model_id)] = max(0, int(row.units_available))
    elif isinstance(current_stock, dict):
        for key, units in current_stock.items():
            if isinstance(key, tuple) and len(key) == 2:
                by_pair[(str(key[0]), str(key[1]))] = max(0, int(units))
            else:
                by_model[str(key)] = max(0, int(units))
    else:
        raise TypeError("current_stock must be a DataFrame or dictionary")
    return by_model, by_pair


def _weighted_recent_demand(
    sales_history: pd.DataFrame,
    store_id: str,
    model_id: str,
    week: pd.Timestamp,
    weekly_sales: Optional[Dict[Tuple[str, str], pd.Series]] = None,
) -> float:
    """Estimate demand from six completed weeks, weighting recent weeks more."""
    if weekly_sales is None:
        matching = sales_history[
            (sales_history["store_id"] == store_id)
            & (sales_history["model_id"] == model_id)
            & (pd.to_datetime(sales_history["week_start_date"]) < week)
        ]
        weekly = matching.groupby("week_start_date")["units_sold"].sum().sort_index().tail(6)
    else:
        weekly = weekly_sales.get((store_id, model_id), pd.Series(dtype=float))
        weekly = weekly.loc[weekly.index < week].tail(6)
    if weekly.empty:
        return 0.0
    weights = np.arange(1, len(weekly) + 1, dtype=float)
    return float(np.average(weekly.to_numpy(dtype=float), weights=weights))


def _reasoning(
    store_name: str,
    model_name: str,
    allocated_units: int,
    expected_demand: float,
    capital: int,
    expected_revenue: float,
    roi: float,
    cap_applied: bool,
) -> str:
    """Render a business-readable rupee explanation for one recommendation."""
    cap_note = " [capped at 15% of budget]" if cap_applied else ""
    return (
        f"{store_name}: {allocated_units} units of {model_name} - expected demand "
        f"{expected_demand:.1f}/week, capital tied up Rs {capital:,.0f}, expected "
        f"gross profit Rs {expected_revenue:,.0f}, ROI {roi:.2f}x{cap_note}"
    )


def allocate_weekly_stock(
    sales_history: pd.DataFrame,
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
    current_stock: Union[pd.DataFrame, Dict[str, int], Dict[Tuple[str, str], int]],
    budget: int = 4_00_00_000,
    week_start_date: Optional[pd.Timestamp] = None,
    max_share_per_store_model: float = 0.15,
) -> Tuple[pd.DataFrame, int, int, pd.DataFrame]:
    """Allocate stock by expected revenue per rupee, subject to concentration.

    Pure ROI maximization can put a month's budget into a handful of attractive
    but slow-moving flagships. The 15% store-model cap limits that failure mode,
    while the greedy pass preserves the simple, explainable business rule:
    serve the highest-return demand first, then move down the ranked list.
    ROI here means expected gross profit divided by capital tied up, not revenue divided by capital.

    ``current_stock`` may be a model-level warehouse DataFrame/dict or a
    store-model DataFrame/dict. The return value is allocations, budget used,
    remaining budget, and an explicit unmet-demand DataFrame. A row is included
    in the unmet list whenever meaningful demand was blocked by stock or budget.
    """
    if week_start_date is None:
        raise ValueError("week_start_date is required so the forecast is reproducible")
    if not 0 < max_share_per_store_model <= 1:
        raise ValueError("max_share_per_store_model must be between 0 and 1")
    missing_sales = REQUIRED_SALES_COLUMNS - set(sales_history.columns)
    if missing_sales:
        raise ValueError(f"sales_history is missing columns: {sorted(missing_sales)}")

    week = pd.Timestamp(week_start_date)
    budget = festive_precommitment_budget(week, budget)
    catalog_by_id = catalog.set_index("model_id").to_dict("index")
    launch_dates = pd.to_datetime(catalog["launch_date"])
    launch_by_id = dict(zip(catalog["model_id"], launch_dates))
    weekly_sales_frame = sales_history.groupby(
        ["store_id", "model_id", "week_start_date"], as_index=False
    )["units_sold"].sum()
    weekly_sales = {
        key: group.set_index("week_start_date")["units_sold"].sort_index()
        for key, group in weekly_sales_frame.groupby(["store_id", "model_id"])
    }
    model_stock, pair_stock = _normalise_current_stock(current_stock)
    cap_per_pair = budget * max_share_per_store_model
    candidates: List[dict] = []

    for store in stores.itertuples(index=False):
        for model in catalog.itertuples(index=False):
            model_info = catalog_by_id[model.model_id]
            successor_launch = (
                launch_by_id.get(model.successor_model_id)
                if pd.notna(model.successor_model_id)
                else None
            )
            stage, stage_factor = _lifecycle_stage(
                week, launch_by_id[model.model_id], model.expected_lifecycle_weeks, successor_launch
            )
            trend = _weighted_recent_demand(
                sales_history, store.store_id, model.model_id, week, weekly_sales
            )
            if trend == 0 and stage != "pre-launch":
                # New launches need a small prior until six weeks of history exist.
                category_prior = {"budget": 2.5, "mid": 1.5, "flagship": 0.5}[model.price_tier]
                fit_prior = {"flagship-heavy": 1.3, "mid-range": 1.0, "budget-heavy": 0.8}[store.sales_mix_profile]
                trend = category_prior * (store.footfall_index / 100) * fit_prior
            expected_demand = trend * stage_factor * festive_multiplier(week)
            requested_units = max(0, ceil(expected_demand))
            if requested_units == 0:
                continue
            available = pair_stock.get((store.store_id, model.model_id), model_stock.get(model.model_id, 0))
            candidates.append(
                {
                    "store_id": store.store_id,
                    "store_name": store.store_name,
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "price_tier": model.price_tier,
                    "selling_price": int(model.selling_price),
                    "margin_pct": float(model.margin_pct),
                    "acquisition_cost": int(model.acquisition_cost),
                    "lifecycle_stage": stage,
                    "expected_demand": expected_demand,
                    "requested_units": requested_units,
                    "available_units": available,
                    "expected_revenue": expected_demand * model.selling_price * model.margin_pct,
                }
            )

    for candidate in candidates:
        requested_capital = candidate["requested_units"] * candidate["acquisition_cost"]
        candidate["priority_score"] = candidate["expected_revenue"] / requested_capital if requested_capital else 0
    candidates.sort(key=lambda item: (item["priority_score"], item["expected_revenue"]), reverse=True)

    allocations: List[dict] = []
    unmet: List[dict] = []
    budget_used = 0
    for candidate in candidates:
        pair_key = (candidate["store_id"], candidate["model_id"])
        available = pair_stock.get(pair_key, model_stock.get(candidate["model_id"], 0))
        affordable_units = int((budget - budget_used) // candidate["acquisition_cost"])
        cap_units = int(cap_per_pair // candidate["acquisition_cost"])
        allocated_units = min(candidate["requested_units"], available, affordable_units, cap_units)
        capital = allocated_units * candidate["acquisition_cost"]
        budget_used += capital
        cap_applied = allocated_units == cap_units and candidate["requested_units"] > cap_units
        expected_revenue = allocated_units * candidate["selling_price"] * candidate["margin_pct"]
        allocated_roi = expected_revenue / capital if capital else 0
        allocations.append(
            {
                **candidate,
                "allocated_units": allocated_units,
                "capital_tied_up": capital,
                "allocated_expected_revenue": expected_revenue,
                "cap_applied": cap_applied,
                "reasoning": _reasoning(
                    candidate["store_name"], candidate["model_name"], allocated_units,
                    candidate["expected_demand"], capital, expected_revenue,
                    allocated_roi, cap_applied,
                ),
            }
        )
        if available == model_stock.get(candidate["model_id"], available) and candidate["model_id"] in model_stock:
            model_stock[candidate["model_id"]] -= allocated_units
        if pair_key in pair_stock:
            pair_stock[pair_key] -= allocated_units
        shortfall_units = candidate["requested_units"] - allocated_units
        if shortfall_units > 0:
            if budget - budget_used < candidate["acquisition_cost"] and available >= shortfall_units:
                reason = "budget exhausted before full demand could be covered"
            elif available < candidate["requested_units"]:
                reason = "warehouse stock shortfall"
            else:
                reason = "15% concentration cap limited this pair"
            unmet.append(
                {
                    "store_id": candidate["store_id"],
                    "store_name": candidate["store_name"],
                    "model_id": candidate["model_id"],
                    "model_name": candidate["model_name"],
                    "requested_units": candidate["requested_units"],
                    "allocated_units": allocated_units,
                    "shortfall_units": shortfall_units,
                    "rupee_shortfall": shortfall_units * candidate["acquisition_cost"],
                    "reason": reason,
                }
            )

    allocation_df = pd.DataFrame(allocations)
    unmet_df = pd.DataFrame(unmet)
    return allocation_df, budget_used, budget - budget_used, unmet_df


if __name__ == "__main__":
    from catalog import generate_catalog
    from sales_generator import generate_sales_history
    from stores import generate_stores

    stores = generate_stores()
    catalog = generate_catalog()
    history = generate_sales_history(stores, catalog)
    sample_week = pd.Timestamp("2025-10-20")
    stress_mask = (
        history["store_id"].eq("BLR-02")
        & history["model_id"].eq("F05")
        & history["week_start_date"].between(
            sample_week - pd.Timedelta(weeks=6), sample_week - pd.Timedelta(days=1)
        )
    )
    history.loc[stress_mask, "units_sold"] *= 100
    current_stock = {model_id: 10_000 for model_id in catalog["model_id"]}
    allocations, used, remaining, unmet = allocate_weekly_stock(
        history, catalog, stores, current_stock, week_start_date=sample_week
    )
    print("Top 10 allocations by capital tied up:")
    print(allocations.nlargest(10, "capital_tied_up")[["capital_tied_up", "reasoning"]].to_string(index=False))
    print(f"\nBudget used: Rs {used:,.0f} / Rs 40,000,000")
    print(f"Remaining budget: Rs {remaining:,.0f}")
    print(f"Pairs hitting the 15% concentration cap: {allocations['cap_applied'].sum()}")
    print("\nUnmet-demand list:")
    print(unmet.to_string(index=False) if not unmet.empty else "None")