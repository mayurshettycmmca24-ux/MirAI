"""Evaluation harness comparing the ROI engine with a naive baseline."""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from allocation_engine import allocate_weekly_stock
from baseline import naive_baseline_allocation
from catalog import generate_catalog
from sales_generator import generate_sales_history
from stores import generate_stores


def _actual_units_after(sales_history: pd.DataFrame, store_id: str, model_id: str, week: pd.Timestamp, horizon: int = 1) -> float:
    """Read actual future units used to score a prior recommendation."""
    dates = pd.to_datetime(sales_history["week_start_date"])
    future = sales_history[
        (sales_history["store_id"] == store_id)
        & (sales_history["model_id"] == model_id)
        & (dates > week)
        & (dates <= week + pd.Timedelta(weeks=horizon))
    ]
    return float(future["units_sold"].sum())


def score_recommendation_log(
    recommendation_log: pd.DataFrame,
    sales_history: pd.DataFrame,
    catalog: pd.DataFrame,
    dead_stock_horizon_weeks: int = 4,
) -> pd.DataFrame:
    """Attach realized outcomes and calculate business-facing scorecard metrics."""
    if recommendation_log.empty:
        return pd.DataFrame()
    costs = catalog.set_index("model_id")["acquisition_cost"].to_dict()
    economics = catalog.set_index("model_id")[["selling_price", "margin_pct"]].to_dict("index")
    actual_by_pair_week = sales_history.groupby(
        ["store_id", "model_id", "week_start_date"]
    )["units_sold"].sum()
    rows = []
    for row in recommendation_log.itertuples(index=False):
        week = pd.Timestamp(row.week_start_date)
        pair_history = actual_by_pair_week.get((row.store_id, row.model_id), pd.Series(dtype=float))
        if not pair_history.empty:
            dates = pd.to_datetime(pair_history.index.get_level_values("week_start_date"))
            actual_next = pair_history.loc[
                (dates > week) & (dates <= week + pd.Timedelta(weeks=1))
            ].sum()
            actual_horizon = pair_history.loc[
                (dates > week)
                & (dates <= week + pd.Timedelta(weeks=dead_stock_horizon_weeks))
            ].sum()
        else:
            actual_next = actual_horizon = 0
        allocated = int(row.allocated_units)
        acquisition_cost = int(costs[row.model_id])
        flow_values = row._asdict()
        rows.append({
            **row._asdict(),
            "actual_next_week_units": actual_next,
            "actual_horizon_units": actual_horizon,
            "stockout": flow_values.get("flow_stockout", int(actual_next > allocated)),
            "weeks_of_cover": flow_values.get("flow_weeks_of_cover", allocated / max(actual_next, 0.1)),
            "dead_stock": flow_values.get("flow_dead_stock", int(actual_horizon == 0 and allocated > 0)),
            "markdown_loss": flow_values.get("flow_markdown_loss", max(allocated - actual_horizon, 0) * acquisition_cost * 0.25),
            "realized_cogs": flow_values.get("flow_realized_cogs", min(allocated, actual_next) * acquisition_cost),
            "inventory_value": flow_values.get("flow_inventory_value", allocated * acquisition_cost),
            "predicted_gross_profit": allocated * economics[row.model_id]["selling_price"] * economics[row.model_id]["margin_pct"],
            "realized_gross_profit": min(allocated, actual_next) * economics[row.model_id]["selling_price"] * economics[row.model_id]["margin_pct"],
        })
    scored = pd.DataFrame(rows)
    summary = []
    for strategy, group in scored.groupby("strategy"):
        average_inventory = group["inventory_value"].mean()
        summary.append({
            "strategy": strategy,
            "stockout_rate": group["stockout"].mean(),
            "average_weeks_of_cover": group["weeks_of_cover"].mean(),
            "dead_stock_pct": group["dead_stock"].mean(),
            "markdown_loss": group["markdown_loss"].sum(),
            "capital_turns": group["realized_cogs"].sum() / average_inventory if average_inventory else 0,
            "recommendation_rows": len(group),
        })
    summary_df = pd.DataFrame(summary)
    return scored, summary_df


def _simulate_strategy(
    strategy: str,
    sales_history: pd.DataFrame,
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
    weeks: Iterable[pd.Timestamp],
    initial_stock: Dict[str, int],
    budget: int,
    markdown_horizon_weeks: int = 4,
) -> pd.DataFrame:
    """Run one strategy with warehouse stock and FIFO store inventory lots."""
    weeks = [pd.Timestamp(week) for week in weeks]
    costs = catalog.set_index("model_id")["acquisition_cost"].astype(int).to_dict()
    demand = sales_history.copy()
    demand["week_start_date"] = pd.to_datetime(demand["week_start_date"])
    demand_by_pair_week = demand.groupby(
        ["store_id", "model_id", "week_start_date"]
    )["units_sold"].sum()
    warehouse = {model_id: int(units) for model_id, units in initial_stock.items()}
    lots = defaultdict(list)
    markdown_by_key = defaultdict(float)
    all_lots = {}
    rows = []

    for week in weeks:
        for (store_id, model_id), pair_lots in list(lots.items()):
            remaining_demand = int(demand_by_pair_week.get((store_id, model_id, week), 0))
            for lot in pair_lots:
                sold = min(lot["units"], remaining_demand)
                lot["units"] -= sold
                lot["sold_total"] += sold
                remaining_demand -= sold
                if remaining_demand == 0:
                    break
            lots[(store_id, model_id)] = [lot for lot in pair_lots if lot["units"] > 0]

        if strategy == "real_engine":
            allocation, _, _, _ = allocate_weekly_stock(
                sales_history, catalog, stores, warehouse.copy(), budget=budget, week_start_date=week
            )
        else:
            allocation = naive_baseline_allocation(
                sales_history, catalog, stores, warehouse.copy(), budget=budget, week_start_date=week
            )
        if allocation.empty:
            continue

        for row_number, row in enumerate(allocation.itertuples(index=False)):
            allocated = int(row.allocated_units)
            model_id = row.model_id
            warehouse[model_id] = max(0, warehouse.get(model_id, 0) - allocated)
            lot_key = (week, row.store_id, model_id, row_number)
            lot = {
                "key": lot_key,
                "units": allocated,
                "original_units": allocated,
                "sold_total": 0,
                "markdown_date": week + pd.Timedelta(weeks=markdown_horizon_weeks),
            }
            lots[(row.store_id, model_id)].append(lot)
            all_lots[lot_key] = lot
            actual_next = int(demand_by_pair_week.get(
                (row.store_id, model_id, week + pd.Timedelta(weeks=1)), 0
            ))
            available_before_demand = sum(
                item["units"] for item in lots[(row.store_id, model_id)]
            )
            inventory_value = sum(
                lot["units"] * costs[pair_model]
                for (_, pair_model), pair_lots in lots.items()
                for lot in pair_lots
            )
            rows.append({
                "store_id": row.store_id,
                "model_id": model_id,
                "allocated_units": allocated,
                "week_start_date": week,
                "strategy": strategy,
                "flow_stockout": int(actual_next > available_before_demand),
                "flow_weeks_of_cover": allocated / max(actual_next, 0.1),
                "flow_dead_stock": 0,
                "flow_realized_cogs": 0,
                "flow_inventory_value": inventory_value,
                "flow_markdown_loss": 0.0,
                "flow_lot_key": lot_key,
            })

        for pair, pair_lots in list(lots.items()):
            for lot in pair_lots:
                if week >= lot["markdown_date"] and lot["units"] > 0:
                    markdown_by_key[lot["key"]] += lot["units"] * costs[pair[1]] * 0.25
                    lot["units"] = 0
            lots[pair] = [lot for lot in pair_lots if lot["units"] > 0]

    for pair, pair_lots in lots.items():
        for lot in pair_lots:
            markdown_by_key[lot["key"]] += lot["units"] * costs[pair[1]] * 0.25

    for row in rows:
        lot = all_lots[row.pop("flow_lot_key")]
        row["flow_dead_stock"] = int(lot["sold_total"] == 0 and lot["original_units"] > 0)
        row["flow_realized_cogs"] = lot["sold_total"] * costs[row["model_id"]]
        row["flow_markdown_loss"] = markdown_by_key.get(lot["key"], 0.0)
    return pd.DataFrame(rows)


def build_recommendation_log(
    sales_history: pd.DataFrame,
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
    weeks: Iterable[pd.Timestamp],
    current_stock: Dict[str, int],
    budget: int = 4_00_00_000,
) -> pd.DataFrame:
    """Run both allocators on identical weekly inputs and retain an auditable log."""
    simulations = [
        _simulate_strategy(strategy, sales_history, catalog, stores, weeks, current_stock, budget)
        for strategy in ("real_engine", "naive_baseline")
    ]
    simulations = [simulation for simulation in simulations if not simulation.empty]
    if not simulations:
        return pd.DataFrame(columns=["week_start_date", "strategy", "store_id", "model_id", "allocated_units"])
    return pd.concat(simulations, ignore_index=True)


def evaluate_over_history(
    sales_history: pd.DataFrame,
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
    current_stock: Dict[str, int],
    budget: int = 4_00_00_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one full year of weekly decisions against the same observed history."""
    dates = pd.to_datetime(sales_history["week_start_date"]).sort_values().unique()
    weeks = dates[:-1]
    log = build_recommendation_log(sales_history, catalog, stores, weeks, current_stock, budget)
    scored_log, scorecard = score_recommendation_log(log, sales_history, catalog)
    return log, scored_log, scorecard


def print_scorecard(scorecard: pd.DataFrame) -> None:
    """Print the side-by-side metrics and an honest winner/loser comparison."""
    display = scorecard.set_index("strategy")[[
        "stockout_rate", "average_weeks_of_cover", "dead_stock_pct", "markdown_loss", "capital_turns"
    ]].copy()
    print(display.to_string(float_format=lambda value: f"{value:.3f}"))
    real = scorecard.loc[scorecard.strategy.eq("real_engine")].iloc[0]
    baseline = scorecard.loc[scorecard.strategy.eq("naive_baseline")].iloc[0]
    print("\nMetric comparison (higher is better for capital turns; lower is better otherwise):")
    for metric in display.columns:
        if metric == "capital_turns":
            winner = "real_engine" if real[metric] > baseline[metric] else "naive_baseline"
        else:
            winner = "real_engine" if real[metric] < baseline[metric] else "naive_baseline"
        relation = "wins" if real[metric] != baseline[metric] else "ties"
        print(f"{metric}: {winner} {relation} ({real[metric]:.3f} vs {baseline[metric]:.3f})")


if __name__ == "__main__":
    stores = generate_stores()
    catalog = generate_catalog()
    history = generate_sales_history(stores, catalog)
    stock = {model_id: 1_000 for model_id in catalog["model_id"]}
    log, scored_log, scorecard = evaluate_over_history(history, catalog, stores, stock)
    from pathlib import Path
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    log.to_csv(output_dir / "recommendation_log.csv", index=False)
    scored_log.to_csv(output_dir / "recommendation_scored_log.csv", index=False)
    trailing_weeks = pd.to_datetime(scored_log["week_start_date"]).nlargest(4).unique()
    trailing = scored_log[scored_log["week_start_date"].isin(trailing_weeks)]
    trailing.groupby("strategy").agg(
        predicted_gross_profit=("predicted_gross_profit", "sum"),
        realized_gross_profit=("realized_gross_profit", "sum"),
        markdown_loss=("markdown_loss", "sum"),
        stockouts=("stockout", "sum"),
    ).reset_index().to_csv(output_dir / "trailing_4_week_performance.csv", index=False)
    latest_week = pd.Timestamp(log["week_start_date"].max())
    latest_inventory = log[
        (log["strategy"] == "real_engine")
        & (pd.to_datetime(log["week_start_date"]) == latest_week)
    ].merge(catalog[["model_id", "acquisition_cost"]], on="model_id", how="left")
    latest_inventory["capital_tied_up"] = (
        latest_inventory["allocated_units"] * latest_inventory["acquisition_cost"]
    )
    latest_inventory.to_csv(output_dir / "current_inventory_position.csv", index=False)
    from eol_dashboard import build_dashboard, load_risk_output
    performance = pd.read_csv(output_dir / "trailing_4_week_performance.csv")
    build_dashboard(
        load_risk_output(output_dir / "eol_risk_output.json"),
        inventory_df=latest_inventory,
        performance_df=performance,
        out_path=output_dir / "eol_dashboard.html",
    )
    print("Recommendation log rows:", len(log))
    print_scorecard(scorecard)
