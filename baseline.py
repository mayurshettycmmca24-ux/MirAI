"""Naive last-month-sales allocation baseline for comparison."""

from math import floor
from typing import Dict, Tuple, Union

import pandas as pd


def naive_baseline_allocation(
    sales_history: pd.DataFrame,
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
    current_stock: Union[Dict[str, int], pd.DataFrame],
    week_start_date: pd.Timestamp,
    budget: int = 4_00_00_000,
) -> pd.DataFrame:
    """Allocate each model by each store's share of the preceding four weeks.

    This is intentionally naive: it follows observed sales volume only, ignores
    lifecycle and margin, and therefore provides a fair benchmark for whether
    the risk-aware allocator earns its added complexity. It still respects the
    same chain budget so the comparison is not distorted by a larger spend.
    """
    week = pd.Timestamp(week_start_date)
    history = sales_history.copy()
    history["week_start_date"] = pd.to_datetime(history["week_start_date"])
    prior = history[
        (history["week_start_date"] >= week - pd.Timedelta(weeks=4))
        & (history["week_start_date"] < week)
    ]
    store_sales = prior.groupby("store_id")["units_sold"].sum()
    shares = store_sales.reindex(stores["store_id"], fill_value=0).astype(float)
    shares = shares / shares.sum() if shares.sum() else pd.Series(1 / len(stores), index=stores["store_id"])

    if isinstance(current_stock, pd.DataFrame):
        stock = current_stock.groupby("model_id")["units_available"].sum().to_dict()
    else:
        stock = {key: int(value) for key, value in current_stock.items()}
    catalog_by_id = catalog.set_index("model_id").to_dict("index")
    cap_per_pair = budget * 0.15
    rows = []
    for model_id, model_stock in stock.items():
        if model_id not in catalog_by_id or model_stock <= 0:
            continue
        model = catalog_by_id[model_id]
        for store_id, share in shares.items():
            requested = floor(model_stock * share)
            cap_units = floor(cap_per_pair / model["acquisition_cost"])
            allocated = min(requested, cap_units)
            rows.append({
                "week_start_date": week,
                "store_id": store_id,
                "model_id": model_id,
                "allocated_units": allocated,
                "acquisition_cost": int(model["acquisition_cost"]),
                "selling_price": int(model["selling_price"]),
                "capital_tied_up": allocated * int(model["acquisition_cost"]),
                "reasoning": (
                    f"{store_id}: {allocated} units of {model['model_name']} - "
                    f"allocated from {share:.1%} share of preceding four-week sales; "
                    f"capital tied up Rs {allocated * int(model['acquisition_cost']):,.0f}"
                ),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    remaining = budget
    selected = []
    for row in result.sort_values("capital_tied_up", ascending=False).itertuples(index=False):
        spend = min(row.capital_tied_up, remaining)
        units = int(spend // row.acquisition_cost)
        selected.append({**row._asdict(), "allocated_units": units, "capital_tied_up": units * row.acquisition_cost})
        remaining -= units * row.acquisition_cost
    return pd.DataFrame(selected)
