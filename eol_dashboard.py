"""Generate the canonical HTML dashboard from real Stage 4 risk output."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = {
    "store_id",
    "model_id",
    "units_at_risk",
    "capital_at_risk",
    "markdown_cost",
    "transfer_cost",
    "hold_total_cost",
    "weeks_until_forced_action",
    "recommended_action",
    "risk_triggers",
}


def build_dashboard(
    risk_df: pd.DataFrame,
    out_path: str = "output/eol_dashboard.html",
    inventory_df: Optional[pd.DataFrame] = None,
    performance_df: Optional[pd.DataFrame] = None,
    scorecard_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Write the risk, all-store capital, and trailing-performance views."""
    missing = REQUIRED_COLUMNS - set(risk_df.columns)
    if missing:
        raise ValueError(f"risk_df is missing columns: {sorted(missing)}")

    records = json.loads(risk_df.to_json(orient="records"))
    counts = risk_df["recommended_action"].value_counts().reindex(
        ["MARKDOWN", "TRANSFER", "HOLD"], fill_value=0
    )
    total = len(risk_df)
    capital = risk_df["capital_at_risk"].sum()
    payload = json.dumps(records, ensure_ascii=True)
    if inventory_df is not None and not inventory_df.empty:
        inventory = inventory_df.groupby("store_id", as_index=False).agg(
            units=("allocated_units", "sum"), capital=("capital_tied_up", "sum")
        ).sort_values("capital", ascending=False)
        inventory_rows = "".join(
            f"<tr><td>{row.store_id}</td><td class='num'>{row.units:,.0f}</td><td class='num'>Rs {row.capital:,.0f}</td></tr>"
            for row in inventory.itertuples(index=False)
        )
    else:
        inventory_rows = "<tr><td colspan='3'>No current inventory position supplied.</td></tr>"
    if performance_df is not None and not performance_df.empty:
        performance_rows = "".join(
            f"<tr><td>{row.strategy}</td><td class='num'>Rs {row.predicted_gross_profit:,.0f}</td><td class='num'>Rs {row.realized_gross_profit:,.0f}</td><td class='num'>Rs {row.markdown_loss:,.0f}</td><td class='num'>{row.stockouts:,.0f}</td></tr>"
            for row in performance_df.itertuples(index=False)
        )
    else:
        performance_rows = "<tr><td colspan='5'>No trailing recommendation performance supplied.</td></tr>"
    if scorecard_df is not None and not scorecard_df.empty:
        scorecard_rows = "".join(
            f"<tr><td>{row.strategy}</td><td class='num'>{row.stockout_rate:.3f}</td><td class='num'>{row.average_weeks_of_cover:.3f}</td><td class='num'>{row.dead_stock_pct:.3f}</td><td class='num'>Rs {row.markdown_loss:,.0f}</td><td class='num'>{row.capital_turns:.3f}</td></tr>"
            for row in scorecard_df.itertuples(index=False)
        )
    else:
        scorecard_rows = "<tr><td colspan='6'>No full scorecard supplied.</td></tr>"
    output = Path(out_path)
    output.write_text(
        f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EOL Risk Dashboard - MobiMart</title>
<style>
:root{{--ink:#eae4d6;--paper:#14181a;--raised:#1b2124;--line:#333c3f;--amber:#d9822b;--teal:#3f8f83;--rust:#b8482f;--muted:#9aa3a6;--mono:Consolas,monospace;--display:'Arial Narrow',Arial,sans-serif}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--display);padding:32px 16px 64px}}.wrap{{max-width:1180px;margin:auto}}
header{{display:flex;justify-content:space-between;align-items:end;gap:20px;flex-wrap:wrap;border-bottom:2px solid var(--amber);padding-bottom:18px;margin-bottom:24px}}.stamp{{display:inline-block;background:var(--amber);color:var(--paper);font:600 11px var(--mono);letter-spacing:.15em;padding:4px 8px;margin-bottom:10px}}h1{{font-size:34px;text-transform:uppercase;margin:0 0 4px}}.sub,.meta{{color:var(--muted);font:13px Arial,sans-serif}}.meta{{text-align:right;font-family:var(--mono);font-size:12px;line-height:1.7}}
.summary{{display:grid;grid-template-columns:1fr 1.6fr;gap:16px;margin-bottom:16px}}.card{{background:var(--raised);border:1px solid var(--line);border-left:3px solid var(--amber);padding:18px 20px}}.label{{display:block;color:var(--muted);font:11px var(--mono);letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px}}.value{{color:var(--amber);font-size:38px;font-weight:600}}.note{{display:block;color:var(--muted);font:12px Arial,sans-serif;margin-top:5px}}.bar{{display:flex;height:15px;border:1px solid var(--line);margin:10px 0}}.bar i{{display:block;height:100%}}.markdown{{background:var(--amber)}}.transfer{{background:var(--teal)}}.hold{{background:var(--rust)}}.legend{{display:flex;gap:18px;flex-wrap:wrap;font:12px var(--mono);color:#cfd5d2}}
.controls{{display:flex;gap:12px;align-items:end;flex-wrap:wrap;background:var(--raised);border:1px solid var(--line);padding:14px 16px;margin-bottom:16px}}label{{display:block;color:var(--muted);font:10px var(--mono);letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}}select,input{{background:var(--paper);border:1px solid var(--line);color:var(--ink);padding:8px;font:13px Arial,sans-serif;min-width:145px}}.search{{flex:1;min-width:220px}}.search input{{width:100%}}.shown{{margin-left:auto;padding-bottom:8px;color:var(--muted);font:12px var(--mono)}}.shown b{{color:var(--amber)}}
table{{width:100%;border-collapse:collapse;font:13px Arial,sans-serif}}th{{text-align:left;color:var(--muted);border-bottom:1px solid var(--amber);padding:8px;font:10px var(--mono);text-transform:uppercase;letter-spacing:.08em}}td{{padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:middle}}tr:hover{{background:rgba(217,130,43,.06)}}.num{{text-align:right}}.trigger{{color:#b7bfbd;font-size:12px;max-width:260px}}.pill{{border:1px solid;padding:4px 8px;font:10px var(--mono);letter-spacing:.05em}}.pill.markdown{{color:var(--amber);border-color:var(--amber)}}.pill.transfer{{color:var(--teal);border-color:var(--teal)}}.pill.hold{{color:var(--rust);border-color:var(--rust)}}.gauge{{position:relative;width:70px;height:14px;border:1px solid var(--line);background:var(--paper)}}.gauge-fill{{height:100%;background:var(--teal)}}.urgent .gauge-fill{{background:var(--rust)}}.gauge span{{position:absolute;right:3px;top:-1px;font:9px var(--mono);mix-blend-mode:difference}}.empty{{padding:36px;text-align:center;color:var(--muted);font:13px Arial,sans-serif}}footer{{border-top:1px solid var(--line);margin-top:26px;padding-top:13px;color:#6c7476;font:11px Arial,sans-serif}}@media(max-width:720px){{body{{padding:20px 10px}}.summary{{grid-template-columns:1fr}}.meta{{text-align:left}}.controls>*{{width:100%}}.shown{{margin-left:0}}table{{display:block;overflow-x:auto;white-space:nowrap}}}}
</style>
</head>
<body><main class="wrap">
<header><div><div class="stamp">EOL-RISK</div><h1>End-of-Life Stock Dashboard</h1><p class="sub">MobiMart Inventory Allocation System - real Stage 4 output</p></div><div class="meta">POSITIONS {total}<br>CAPITAL AT RISK Rs {capital:,.0f}</div></header>
<section class="summary"><div class="card"><span class="label">Capital at risk</span><span class="value">Rs {capital:,.0f}</span><span class="note">across {total} real at-risk store-model positions</span></div><div class="card"><span class="label">Recommended disposition</span><div class="bar"><i class="markdown" style="width:{counts['MARKDOWN'] / total * 100 if total else 0:.2f}%"></i><i class="transfer" style="width:{counts['TRANSFER'] / total * 100 if total else 0:.2f}%"></i><i class="hold" style="width:{counts['HOLD'] / total * 100 if total else 0:.2f}%"></i></div><div class="legend">MARKDOWN {counts['MARKDOWN']} &nbsp; TRANSFER {counts['TRANSFER']} &nbsp; HOLD {counts['HOLD']}</div></div></section>
<section class="controls"><div><label for="store">Store</label><select id="store"><option value="">All stores</option></select></div><div><label for="model">Model</label><select id="model"><option value="">All models</option></select></div><div><label for="action">Action</label><select id="action"><option value="">All actions</option><option>MARKDOWN</option><option>TRANSFER</option><option>HOLD</option></select></div><div class="search"><label for="query">Search</label><input id="query" placeholder="store, model, or trigger..."></div><div class="shown"><b id="visible">{total}</b> shown</div></section>
<table id="ledger"><thead><tr><th>Store</th><th>Model</th><th class="num">Units</th><th>Trigger</th><th class="num">Markdown Rs</th><th class="num">Transfer Rs</th><th class="num">Hold Rs</th><th>Runway</th><th>Action</th></tr></thead><tbody></tbody></table><div class="empty" hidden>No positions match the current filters.</div>
<section class="data-section"><h2>Current capital by store</h2><p class="sub">All stores, based on the latest real allocation state.</p><table><thead><tr><th>Store</th><th class="num">Units</th><th class="num">Capital tied up</th></tr></thead><tbody>{inventory_rows}</tbody></table></section>
<section class="data-section"><h2>Trailing 4-week recommendation performance</h2><p class="sub">Predicted and realized gross profit, plus markdown loss and stockouts.</p><table><thead><tr><th>Strategy</th><th class="num">Predicted gross profit</th><th class="num">Realized gross profit</th><th class="num">Markdown loss</th><th class="num">Stockouts</th></tr></thead><tbody>{performance_rows}</tbody></table></section>
<section class="data-section"><h2>Full corrected scorecard</h2><p class="sub">51-week carried-inventory evaluation.</p><table><thead><tr><th>Strategy</th><th class="num">Stockout rate</th><th class="num">Weeks of cover</th><th class="num">Dead stock</th><th class="num">Markdown loss</th><th class="num">Capital turns</th></tr></thead><tbody>{scorecard_rows}</tbody></table></section>
<footer>Costs are the real Stage 4 dispositions. Hold is priced across the full forced-action horizon.</footer>
</main><script>
const RECORDS={payload};
const $=id=>document.getElementById(id), money=v=>v==null?'unavailable':'Rs '+Number(v).toLocaleString('en-IN',{{maximumFractionDigits:0}});
function options(id,key){{[...new Set(RECORDS.map(r=>r[key]))].sort().forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;$(id).append(o)}})}}
function render(){{const s=$('store').value,m=$('model').value,a=$('action').value,q=$('query').value.toLowerCase().trim();const rows=RECORDS.filter(r=>(!s||r.store_id===s)&&(!m||r.model_id===m)&&(!a||r.recommended_action===a)&&(!q||(`${{r.store_id}} ${{r.model_id}} ${{r.risk_triggers}}`).toLowerCase().includes(q)));$('ledger').querySelector('tbody').innerHTML=rows.map(r=>{{const w=Number(r.weeks_until_forced_action||0),fill=Math.max(0,100-Math.min(12,w)/12*100);return `<tr title="${{r.reasoning||''}}"><td class="mono">${{r.store_id}}</td><td>${{r.model_id}}</td><td class="num">${{r.units_at_risk}}</td><td class="trigger">${{r.risk_triggers||'-'}}</td><td class="num">${{money(r.markdown_cost)}}</td><td class="num">${{money(r.transfer_cost)}}</td><td class="num">${{money(r.hold_total_cost)}}</td><td><div class="gauge ${{w<=2?'urgent':''}}"><div class="gauge-fill" style="width:${{fill}}%"></div><span>${{w}}w</span></div></td><td><span class="pill ${{r.recommended_action.toLowerCase()}}">${{r.recommended_action}}</span></td></tr>`}}).join('');$('visible').textContent=rows.length;$('ledger').style.display=rows.length?'':'none';document.querySelector('.empty').hidden=rows.length>0}}
options('store','store_id');options('model','model_id');['store','model','action'].forEach(id=>$(id).addEventListener('change',render));$('query').addEventListener('input',render);render();
</script></body></html>''',
        encoding="utf-8",
    )
    return output


def load_risk_output(path: str = "output/eol_risk_output.json") -> pd.DataFrame:
    """Load generated risk output, preferring JSON and falling back to CSV."""
    source = Path(path)
    if source.exists():
        return pd.read_json(source)
    csv_source = source.with_suffix(".csv")
    if csv_source.exists():
        return pd.read_csv(csv_source)
    raise FileNotFoundError(f"No risk output found at {source} or {csv_source}")


if __name__ == "__main__":
    risk_df = load_risk_output()
    dashboard_path = build_dashboard(risk_df)
    print(f"Dashboard written to {dashboard_path}")
    print("Recommended actions:")
    print(risk_df["recommended_action"].value_counts().reindex(["MARKDOWN", "TRANSFER", "HOLD"], fill_value=0).to_string())
    print(f"Capital at risk: Rs {risk_df['capital_at_risk'].sum():,.0f}")
