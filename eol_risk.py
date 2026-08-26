"""End-of-life inventory risk assessment for MobiMart."""

from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


FIT_SCORES = {
    "flagship-heavy": {"budget": 0.45, "mid": 0.85, "flagship": 1.55},
    "mid-range": {"budget": 0.75, "mid": 1.35, "flagship": 0.70},
    "budget-heavy": {"budget": 1.55, "mid": 0.80, "flagship": 0.30},
}


def _stock_by_pair(
    current_stock: Union[pd.DataFrame, Dict[Tuple[str, str], int]]
) -> Dict[Tuple[str, str], int]:
    """Normalize store-model inventory so risk is assessed where stock sits."""
    if isinstance(current_stock, pd.DataFrame):
        if not {"store_id", "model_id"}.issubset(current_stock.columns):
            raise ValueError("current_stock must include store_id and model_id")
        quantity_column = next(
            (column for column in ("units_available", "allocated_units", "current_stock") if column in current_stock),
            None,
        )
        if quantity_column is None:
            raise ValueError("current_stock needs units_available, allocated_units, or current_stock")
        return {
            (str(row.store_id), str(row.model_id)): max(0, int(getattr(row, quantity_column)))
            for row in current_stock.itertuples(index=False)
            if int(getattr(row, quantity_column)) > 0
        }
    if isinstance(current_stock, dict):
        if not all(isinstance(key, tuple) and len(key) == 2 for key in current_stock):
            raise ValueError("dictionary current_stock must use (store_id, model_id) keys")
        return {
            (str(key[0]), str(key[1])): max(0, int(units))
            for key, units in current_stock.items()
            if int(units) > 0
        }
    raise TypeError("current_stock must be a store-model DataFrame or dictionary")


def _recent_sales(sales_history: pd.DataFrame, store_id: str, model_id: str, as_of_date: pd.Timestamp) -> Tuple[float, float]:
    """Return recent two-week and prior-six-week average unit sales."""
    matching = sales_history[
        (sales_history["store_id"] == store_id)
        & (sales_history["model_id"] == model_id)
        & (pd.to_datetime(sales_history["week_start_date"]) < as_of_date)
    ].groupby("week_start_date")["units_sold"].sum().sort_index()
    return float(matching.tail(2).mean() or 0), float(matching.tail(8).head(6).mean() or 0)


def _risk_stage(
    as_of_date: pd.Timestamp,
    launch_date: pd.Timestamp,
    lifecycle_weeks: int,
    successor_launch_date: Optional[pd.Timestamp],
    recent_two_week_sales: float,
    prior_six_week_sales: float,
) -> Tuple[List[str], float]:
    """Apply lifecycle, successor, and sell-through triggers with severity."""
    age_weeks = (as_of_date - launch_date).days / 7
    peak_week = min(10, max(8, round(lifecycle_weeks * 0.30)))
    triggers: List[str] = []
    severity = 0.0
    if age_weeks > peak_week and recent_two_week_sales < prior_six_week_sales:
        triggers.append("past peak with declining store trend")
        severity = max(severity, min(1.0, (age_weeks - peak_week) / max(1, lifecycle_weeks - peak_week)))
    if successor_launch_date is not None:
        weeks_to_successor = (successor_launch_date - as_of_date).days / 7
        if 0 <= weeks_to_successor <= 4:
            triggers.append(f"successor launches in {max(0, ceil(weeks_to_successor))} weeks")
            severity = max(severity, 1 - (weeks_to_successor / 4))
    if prior_six_week_sales > 0 and recent_two_week_sales < 0.4 * prior_six_week_sales:
        triggers.append("last two weeks below 40% of prior six-week average")
        severity = max(severity, 0.8)
    if age_weeks > lifecycle_weeks:
        triggers.append("past expected lifecycle")
        severity = max(severity, 0.9)
    return triggers, min(1.0, severity)


def _hold_cost(
    capital_at_risk: float,
    weeks_until_forced_action: float,
    probability_unsold_at_deadline: float,
    future_markdown_pct: float,
) -> Tuple[float, float, float]:
    """Price the full hold horizon, including deadline-weighted markdown risk."""
    opportunity_cost = capital_at_risk * 0.18 / 52 * weeks_until_forced_action
    expected_markdown_cost = (
        capital_at_risk * future_markdown_pct * probability_unsold_at_deadline
    )
    return opportunity_cost, expected_markdown_cost, opportunity_cost + expected_markdown_cost


def assess_eol_risk(
    current_stock: Union[pd.DataFrame, Dict[Tuple[str, str], int]],
    catalog: pd.DataFrame,
    sales_history: pd.DataFrame,
    stores: pd.DataFrame,
    as_of_date: pd.Timestamp,
    markdown_pct_range: Tuple[float, float] = (0.15, 0.30),
    transfer_cost_per_unit_range: Tuple[int, int] = (300, 800),
) -> pd.DataFrame:
    """Compare markdown, transfer, and hold costs for every risky stock position.

    A lifecycle trigger catches inventory whose launch excitement has passed,
    a successor trigger anticipates the sharp demand shift caused by a newer
    family member, and the sell-through trigger catches local execution or
    preference changes before the catalog lifecycle alone would. Transfer is
    allowed only to a store with a stronger price-tier fit and no risk flag for
    that model, preventing the system from moving dead stock between stores.
    """
    if not 0 < markdown_pct_range[0] <= markdown_pct_range[1] <= 1:
        raise ValueError("markdown_pct_range must be within (0, 1]")
    if not 0 < transfer_cost_per_unit_range[0] <= transfer_cost_per_unit_range[1]:
        raise ValueError("transfer_cost_per_unit_range must be positive and ordered")
    stock_by_pair = _stock_by_pair(current_stock)
    week = pd.Timestamp(as_of_date)
    catalog_by_id = catalog.set_index("model_id").to_dict("index")
    launch_by_id = dict(zip(catalog["model_id"], pd.to_datetime(catalog["launch_date"])))
    store_by_id = stores.set_index("store_id").to_dict("index")
    risk_flags: Dict[Tuple[str, str], Tuple[List[str], float, float, float]] = {}

    for (store_id, model_id), units in stock_by_pair.items():
        model = catalog_by_id[model_id]
        successor_launch = (
            launch_by_id.get(model["successor_model_id"])
            if pd.notna(model["successor_model_id"])
            else None
        )
        successor_status = (
            str(catalog_by_id[model["successor_model_id"]]["launch_date_status"])
            if pd.notna(model["successor_model_id"])
            else "confirmed"
        )
        recent_two, prior_six = _recent_sales(sales_history, store_id, model_id, week)
        triggers, severity = _risk_stage(
            week, launch_by_id[model_id], model["expected_lifecycle_weeks"],
            successor_launch, recent_two, prior_six,
        )
        if triggers:
            risk_flags[(store_id, model_id)] = (triggers, severity, recent_two, prior_six)

    rng = np.random.default_rng(2025)
    results: List[dict] = []
    for (store_id, model_id), (triggers, severity, recent_two, prior_six) in risk_flags.items():
        units = stock_by_pair[(store_id, model_id)]
        model = catalog_by_id[model_id]
        source_store = store_by_id[store_id]
        acquisition_cost = int(model["acquisition_cost"])
        capital_at_risk = units * acquisition_cost
        markdown_pct = markdown_pct_range[0] + severity * (markdown_pct_range[1] - markdown_pct_range[0])
        markdown_cost = capital_at_risk * markdown_pct
        successor_launch = (
            launch_by_id.get(model["successor_model_id"])
            if pd.notna(model["successor_model_id"])
            else None
        )
        if successor_launch is not None:
            weeks_to_successor = (successor_launch - week).days / 7
        else:
            weeks_to_successor = float("inf")
        if 0 <= weeks_to_successor <= 4:
            weeks_until_forced_action = max(1.0, weeks_to_successor)
        else:
            weeks_until_forced_action = min(
                6.0, max(1.0, ceil(units / max(recent_two, 0.1)))
            )
        probability_unsold = min(
            0.98, 0.20 + 0.80 * (1 - (weeks_until_forced_action - 1) / 5)
        )
        if successor_status == "rumoured":
            probability_unsold *= 0.75
        hold_opportunity_cost, hold_risk_cost, hold_total_cost = _hold_cost(
            capital_at_risk, weeks_until_forced_action, probability_unsold,
            max(markdown_pct, markdown_pct_range[1]) + 0.05,
        )

        targets = []
        for target_id, target_store in store_by_id.items():
            target_fit = FIT_SCORES[target_store["sales_mix_profile"]][model["price_tier"]]
            source_fit = FIT_SCORES[source_store["sales_mix_profile"]][model["price_tier"]]
            if target_id != store_id and target_fit > source_fit and (target_id, model_id) not in risk_flags:
                targets.append((target_fit, target_id))
        transfer_target = None
        transfer_cost = np.nan
        transfer_note = "No better-fit store is available without creating another at-risk position."
        if targets:
            _, transfer_target = max(targets)
            transfer_per_unit = int(rng.integers(transfer_cost_per_unit_range[0], transfer_cost_per_unit_range[1] + 1))
            holding_cost = capital_at_risk * 0.18 * 4 / 365
            transfer_cost = units * transfer_per_unit + holding_cost
            transfer_note = f"Transfer target {store_by_id[transfer_target]['store_name']} at Rs {transfer_per_unit}/unit plus 4-day holding cost."

        costs = {"MARKDOWN": markdown_cost, "TRANSFER": transfer_cost, "HOLD": hold_total_cost}
        available_costs = {action: cost for action, cost in costs.items() if pd.notna(cost)}
        recommendation = min(available_costs, key=available_costs.get)
        recommendation_cost = available_costs[recommendation]
        successor_text = "no successor within four weeks"
        if model["successor_model_id"] in launch_by_id:
            successor_weeks = (launch_by_id[model["successor_model_id"]] - week).days / 7
            successor_text = f"successor in {successor_weeks:.0f} weeks" if 0 <= successor_weeks <= 4 else successor_text
        reasoning = (
            f"{source_store['store_name']}, {model['model_name']}, {units} units at risk "
            f"({'; '.join(triggers)}; {successor_text}):\n"
            f"- Markdown ({markdown_pct:.0%}): Rs {markdown_cost:,.0f} margin lost\n"
            f"- Transfer: {f'Rs {transfer_cost:,.0f}' if pd.notna(transfer_cost) else 'unavailable'}; {transfer_note}\n"
            f"- Hold: Rs {hold_opportunity_cost:,.0f} opportunity cost + Rs {hold_risk_cost:,.0f} markdown risk "
            f"= Rs {hold_total_cost:,.0f}\n"
            f"RECOMMENDED: {recommendation} at Rs {recommendation_cost:,.0f}"
        )
        results.append(
            {
                "store_id": store_id,
                "store_name": source_store["store_name"],
                "model_id": model_id,
                "model_name": model["model_name"],
                "price_tier": model["price_tier"],
                "successor_launch_status": successor_status,
                "units_at_risk": units,
                "risk_triggers": "; ".join(triggers),
                "recent_two_week_sales": recent_two,
                "prior_six_week_avg_sales": prior_six,
                "capital_at_risk": capital_at_risk,
                "markdown_pct": markdown_pct,
                "markdown_cost": markdown_cost,
                "weeks_until_forced_action": weeks_until_forced_action,
                "probability_unsold_at_deadline": probability_unsold,
                "transfer_target": transfer_target,
                "transfer_cost": transfer_cost,
                "hold_opportunity_cost": hold_opportunity_cost,
                "hold_markdown_risk_cost": hold_risk_cost,
                "hold_total_cost": hold_total_cost,
                "recommended_action": recommendation,
                "recommended_cost": recommendation_cost,
                "reasoning": reasoning,
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    from allocation_engine import allocate_weekly_stock
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
        & history["week_start_date"].between(sample_week - pd.Timedelta(weeks=6), sample_week - pd.Timedelta(days=1))
    )
    history.loc[stress_mask, "units_sold"] *= 100
    allocations, _, _, _ = allocate_weekly_stock(
        history, catalog, stores, {model_id: 10_000 for model_id in catalog["model_id"]}, week_start_date=sample_week
    )
    risks = assess_eol_risk(allocations, catalog, history, stores, as_of_date="2025-10-06")
    from pathlib import Path
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    risks.to_csv(output_dir / "eol_risk_output.csv", index=False)
    risks.to_json(output_dir / "eol_risk_output.json", orient="records")
    print("At-risk store-model pairs:", len(risks))
    print("Total capital at risk: Rs", f"{risks['capital_at_risk'].sum():,.0f}")
    breakdown = risks["recommended_action"].value_counts().reindex(
        ["MARKDOWN", "TRANSFER", "HOLD"], fill_value=0
    )
    print("Recommendation breakdown:\n", breakdown.to_string())
    markdown_rows = risks.loc[risks["recommended_action"].eq("MARKDOWN")]
    successor_markdown = markdown_rows.loc[
        markdown_rows["risk_triggers"].str.contains("successor")
    ]
    other_markdowns = markdown_rows.loc[
        ~markdown_rows.index.isin(successor_markdown.index)
    ]
    markdown_examples = pd.concat([successor_markdown.head(1), other_markdowns.head(2)])
    print("\nMarkdown-winning examples:")
    for row in markdown_examples.itertuples(index=False):
        print(f"\n{row.store_id} / {row.model_id} / {row.recommended_action}")
        print(row.reasoning)