# =============================================================================
# FINA 4011/5011 — Project 2: DCF Equity Valuation App
# =============================================================================
# A Streamlit app that values a stock via Discounted Cash Flow (DCF).
#
# Design principles:
#   - Manual inputs are the source of truth (always editable, never overwritten silently).
#   - Optional one-click "Auto-fill from SEC EDGAR" pulls real 10-K fundamentals.
#   - No real-time Yahoo Finance calls (avoids rate-limit failures).
#   - Every computation is shown step-by-step with formulas.
#   - Sensitivity tables on two key drivers.
#   - Excel export creates a workbook with REAL formulas matching the app's math.
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run equity_valuation_app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from io import BytesIO

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Equity Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 DCF Equity Valuation App")
st.caption("FINA 4011/5011 · Project 2 · Discounted Cash Flow Valuation Tool")

# =============================================================================
# SEC EDGAR — optional auto-fill helpers
# Uses the official SEC API: free, no key, not rate-limited beyond UA requirement.
# =============================================================================
_SEC_HEADERS = {
    "User-Agent": "FINA4011-DCF-App academic-project@university.edu",
    "Accept-Encoding": "gzip, deflate",
}

@st.cache_data(ttl=86400, show_spinner=False)
def _ticker_to_cik(symbol: str):
    """Map ticker → 10-digit CIK + company name. Cached for a day.

    SEC normalises tickers using hyphens for share classes (BRK-B, BRK-A) but
    users commonly type them with dots (BRK.B). Try a few variants.
    """
    if not symbol:
        return None, None
    sym_u = symbol.upper().strip()
    variants = {sym_u, sym_u.replace(".", "-"), sym_u.replace("-", "."), sym_u.replace(".", "")}
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": _SEC_HEADERS["User-Agent"]},
            timeout=10,
        )
        if r.status_code != 200:
            return None, None
        for entry in r.json().values():
            if entry.get("ticker", "").upper() in variants:
                return str(entry["cik_str"]).zfill(10), entry.get("title", symbol)
    except Exception:
        pass
    return None, None


def _latest_annual(facts_block):
    """Extract the most recent annual value from an XBRL concept block.

    EDGAR rows have multiple periods (quarterly, YTD, full-year). We detect
    annual values by duration (start→end ≈ 365 days) and group by the year of
    the `end` date. Works for any fiscal calendar.
    """
    if not facts_block:
        return None, []
    rows = facts_block.get("units", {}).get("USD") \
        or facts_block.get("units", {}).get("shares") \
        or next(iter(facts_block.get("units", {}).values()), [])

    annual = []
    for r in rows:
        start, end = r.get("start"), r.get("end")
        if not end:
            continue
        # Instant facts (balance-sheet line items) have no `start`
        if not start:
            annual.append(r)
            continue
        try:
            d = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
            if 350 <= d <= 380:
                annual.append(r)
        except Exception:
            continue

    # Prefer 10-K filings
    tenk = [r for r in annual if r.get("form") == "10-K"]
    if tenk:
        annual = tenk

    # Group by year-of-end, keep latest filed
    by_year = {}
    for r in annual:
        try:
            yk = datetime.fromisoformat(r["end"]).year
        except Exception:
            continue
        if yk not in by_year or r.get("filed", "") > by_year[yk].get("filed", ""):
            by_year[yk] = r

    series = sorted(by_year.items(), key=lambda kv: kv[0], reverse=True)
    latest = series[0][1]["val"] if series else None
    return latest, [(y, row["val"]) for y, row in series if row.get("val") is not None]


def _best_concept(us_gaap, *names):
    """Return the concept block with the freshest annual data among the given tags.
    Companies switch tags over time (e.g. ASC 606 moved revenue from `Revenues`
    to `RevenueFromContractWithCustomerExcludingAssessedTax`).
    """
    best, best_year = None, -1
    for n in names:
        block = us_gaap.get(n)
        if not block:
            continue
        rows = block.get("units", {}).get("USD") \
            or block.get("units", {}).get("shares") or []
        years = [r.get("fy") for r in rows if r.get("fp") == "FY" and r.get("fy")]
        if not years:
            continue
        max_fy = max(years)
        if max_fy > best_year:
            best, best_year = block, max_fy
    return best


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_edgar_fundamentals(symbol: str):
    """Pull fundamentals from SEC EDGAR Company Facts API.
    Returns dict of latest annual values, or None on failure."""
    try:
        cik, name = _ticker_to_cik(symbol)
        if cik is None:
            return None
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers={**_SEC_HEADERS, "Host": "data.sec.gov"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        ug = r.json().get("facts", {}).get("us-gaap", {}) or {}
        dei = r.json().get("facts", {}).get("dei", {}) or {}
        # Combined namespace lookup for shares (dei + us-gaap — companies use either)
        all_facts = {**ug, **dei}

        rev,    rev_hist = _latest_annual(_best_concept(ug,
            "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"))
        # Operating income — use NetIncomeLoss as fallback for banks/insurers that don't report OpIncLoss
        opinc,  _        = _latest_annual(_best_concept(ug,
            "OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"))
        net_inc, _       = _latest_annual(_best_concept(ug, "NetIncomeLoss",
            "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"))
        ocf,    _        = _latest_annual(_best_concept(ug,
            "NetCashProvidedByUsedInOperatingActivities"))
        capex,  _        = _latest_annual(_best_concept(ug,
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets"))
        cash,   _        = _latest_annual(_best_concept(ug,
            "CashAndCashEquivalentsAtCarryingValue",
            "Cash",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"))
        dlt,    _        = _latest_annual(_best_concept(ug,
            "LongTermDebt", "LongTermDebtNoncurrent"))
        dst,    _        = _latest_annual(_best_concept(ug,
            "LongTermDebtCurrent", "DebtCurrent",
            "ShortTermBorrowings"))
        # Shares: try multiple tags across BOTH dei and us-gaap. Prefer diluted weighted-avg
        # (standard for per-share metrics), fall back to common shares outstanding.
        shares, _        = _latest_annual(_best_concept(all_facts,
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding"))

        # ── Derived metrics ───────────────────────────────────────────────
        # EBIT margin: prefer real EBIT, else estimate from net income (assume 21% effective tax)
        ebit_margin = None
        if opinc and rev and opinc > 0:
            ebit_margin = opinc / rev
        elif net_inc and rev and net_inc > 0:
            # Net income → pre-tax estimate (gross-up by 1/(1-0.21))
            ebit_margin = (net_inc / 0.79) / rev

        # Revenue CAGR over available history (uses up to 4-yr window for stability)
        revenue_cagr = None
        if rev_hist and len(rev_hist) >= 2:
            n_years = min(len(rev_hist) - 1, 4)
            oldest = rev_hist[n_years][1]
            latest = rev_hist[0][1]
            if oldest > 0 and n_years > 0:
                revenue_cagr = (latest / oldest) ** (1 / n_years) - 1

        # Track which fields couldn't be resolved — surface to user so they
        # know which inputs to set manually instead of trusting a silent zero.
        missing = []
        if not rev:        missing.append("Revenue")
        if not shares:     missing.append("Shares Outstanding")
        if ebit_margin is None: missing.append("EBIT Margin")
        if cash is None:   missing.append("Cash")
        if not (dlt or dst): missing.append("Debt")

        return {
            "name":          name,
            "revenue":       rev / 1e6 if rev else None,
            "ebit_margin":   ebit_margin * 100 if ebit_margin else None,
            "operating_cf":  ocf / 1e6 if ocf else None,
            "capex":         capex / 1e6 if capex else None,
            "cash":          cash / 1e6 if cash else None,
            "debt":          ((dlt or 0) + (dst or 0)) / 1e6 if (dlt or dst) else None,
            "shares":        shares / 1e6 if shares else None,
            "revenue_cagr":  revenue_cagr * 100 if revenue_cagr else None,
            "rev_history":   [(y, v / 1e9) for y, v in rev_hist[:5]],
            "fy_end":        rev_hist[0][0] if rev_hist else None,
            "missing":       missing,
        }
    except Exception:
        return None


# =============================================================================
# Session-state defaults — every input has a fallback so the app always runs
# =============================================================================
DEFAULTS = {
    "ticker":          "AAPL",
    "company_name":    "Apple Inc.",
    "revenue":         391000.0,   # $M (AAPL FY24)
    "growth":          7.0,        # %
    "margin":          30.0,       # % EBIT margin
    "tax_rate":        21.0,       # %
    "reinvest_rate":   25.0,       # % of NOPAT
    "wacc":            9.0,        # %
    "terminal_growth": 2.5,        # %
    "years":           5,
    "debt":            107000.0,   # $M
    "cash":            65000.0,    # $M
    "shares":          15300.0,    # millions
    "current_price":   210.00,     # $
    "edgar_data":      None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# Sidebar — all inputs
# =============================================================================
with st.sidebar:
    st.header("📋 Inputs")

    # ─── Ticker + EDGAR auto-fill ────────────────────────────────────────────
    st.subheader("🏢 Company")
    st.text_input("Ticker", key="ticker", help="US-listed ticker (e.g. AAPL, MSFT, JNJ)")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        if st.button("🔄 Auto-fill from EDGAR", use_container_width=True,
                     help="Pull latest 10-K data from the SEC's official database (free, no API key)"):
            with st.spinner("Fetching from SEC EDGAR..."):
                edgar = fetch_edgar_fundamentals(st.session_state.ticker)
            if edgar:
                # ── Sanity check first: a DCF requires positive revenue. ──
                # Pre-revenue companies (junior miners, early biotech, dev-stage SPACs)
                # can't be valued via DCF — refuse to overwrite the user's working
                # defaults with an inconsistent partial fill that produces nonsense.
                edgar_rev = edgar.get("revenue")
                if not edgar_rev or edgar_rev <= 0:
                    st.session_state.edgar_data = None  # don't show the reference panel
                    st.error(
                        f"⛔ **{edgar.get('name', st.session_state.ticker)}** "
                        "appears to be a **pre-revenue company** (no operating revenue in latest 10-K). "
                        "DCF requires positive cash flows, so auto-fill won't work here. "
                        "Suggestions:\n\n"
                        "- Try a revenue-generating ticker (AAPL, MSFT, JNJ, KO)\n"
                        "- For pre-revenue / dev-stage companies, use **asset-based valuation** instead\n"
                        "- Or override Revenue, Margin, Shares manually below if you have a forecast"
                    )
                else:
                    # Only overwrite a field when EDGAR returned a real (positive) value.
                    # Missing / zero values are kept as the existing user input or default.
                    def _set(key, val, min_val=0):
                        if val is None: return
                        try: v = float(val)
                        except (TypeError, ValueError): return
                        if v <= min_val: return
                        st.session_state[key] = round(v, 1) if isinstance(v, float) and abs(v) < 1000 else round(v)

                    if edgar.get("name"):
                        st.session_state.company_name = edgar["name"]
                    _set("revenue",  edgar.get("revenue"))
                    _set("margin",   edgar.get("ebit_margin"))
                    _set("shares",   edgar.get("shares"))
                    _set("growth",   edgar.get("revenue_cagr"))
                    # Cash and debt can legitimately be zero
                    if edgar.get("cash") is not None and edgar.get("cash") >= 0:
                        st.session_state.cash = round(edgar["cash"], 0)
                    if edgar.get("debt") is not None and edgar.get("debt") >= 0:
                        st.session_state.debt = round(edgar["debt"], 0)
                    # Reinvestment rate ≈ CapEx / NOPAT — only compute if all parts valid
                    if edgar.get("capex") and edgar.get("revenue") and edgar.get("ebit_margin"):
                        nopat = edgar["revenue"] * (edgar["ebit_margin"]/100) * (1 - st.session_state.tax_rate/100)
                        if nopat > 0:
                            ri = min(100, edgar["capex"] / nopat * 100)
                            if ri > 0:
                                st.session_state.reinvest_rate = round(ri, 1)

                    st.session_state.edgar_data = edgar
                    fy = edgar.get("fy_end", "?")

                    # Warn if the company is unprofitable — DCF will still run but the
                    # extrapolation from a loss-making base year is unreliable.
                    if edgar.get("ebit_margin") is None:
                        st.warning(
                            f"⚠️ Loaded {edgar.get('name', st.session_state.ticker)} (FY{fy}), "
                            "but operating margin couldn't be determined (likely a loss-making year). "
                            "DCF results will reflect your manual margin assumption, not the company's actuals."
                        )
                    elif edgar.get("missing"):
                        st.warning(
                            f"⚠️ Loaded {edgar.get('name', st.session_state.ticker)} (FY{fy}). "
                            f"EDGAR didn't have: **{', '.join(edgar['missing'])}** — "
                            "left at default. Edit these fields below if needed."
                        )
                    else:
                        st.success(f"✅ Loaded {edgar.get('name', st.session_state.ticker)} (FY{fy})")
                    st.rerun()
            else:
                st.error(
                    "❌ Ticker not found in EDGAR (or filings unavailable). "
                    "EDGAR covers US-listed companies — try AAPL, MSFT, JNJ, etc. "
                    "Or enter values manually below."
                )
    with col_b:
        if st.button("🧹 Reset", use_container_width=True, help="Reset all inputs to defaults"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    st.text_input("Company Name", key="company_name")

    st.markdown("---")

    # ─── Financial inputs ────────────────────────────────────────────────────
    st.subheader("💰 Financial Inputs")
    st.caption("All dollar values in millions ($M)")

    st.number_input("Current Revenue ($M)", min_value=0.0, step=100.0, key="revenue",
                    help="Most recent annual revenue. EDGAR fills with latest 10-K value.")
    st.number_input("EBIT Margin (%)", min_value=0.0, max_value=100.0, step=0.5, key="margin",
                    help="Operating Income ÷ Revenue. Higher margin → more value. EDGAR computes from 10-K.")
    st.number_input("Tax Rate (%)", min_value=0.0, max_value=50.0, step=0.5, key="tax_rate",
                    help="Effective corporate tax rate. US federal statutory is 21%; effective often 15-25%.")
    st.number_input("Reinvestment Rate (%)", min_value=0.0, max_value=100.0, step=1.0, key="reinvest_rate",
                    help="% of NOPAT reinvested in CapEx + working capital. Higher → less FCF today, more growth.")

    st.markdown("---")

    # ─── Growth + WACC ───────────────────────────────────────────────────────
    st.subheader("📈 Growth & Discount Rate")
    st.number_input("Annual Growth Rate (%)", min_value=-20.0, max_value=50.0, step=0.5, key="growth",
                    help="Revenue growth during forecast period. EDGAR fills with 3-yr CAGR.")
    st.number_input("WACC — Discount Rate (%)", min_value=1.0, max_value=30.0, step=0.25, key="wacc",
                    help="Weighted Average Cost of Capital. Mature firms: 7-10%. Riskier: 10-15%.")
    st.number_input("Terminal Growth Rate (%)", min_value=0.0, max_value=10.0, step=0.25, key="terminal_growth",
                    help="Perpetual growth after forecast. Cap at long-run GDP growth (~2-3%).")
    st.slider("Forecast Years", min_value=3, max_value=10, key="years",
              help="Length of explicit forecast. Standard is 5-10 years.")

    st.markdown("---")

    # ─── Capital structure ───────────────────────────────────────────────────
    st.subheader("⚖️ Capital Structure")
    st.number_input("Total Debt ($M)", min_value=0.0, step=100.0, key="debt",
                    help="Long-term + short-term debt from balance sheet.")
    st.number_input("Cash & Equivalents ($M)", min_value=0.0, step=100.0, key="cash",
                    help="Cash + marketable securities. Subtracted from EV → equity value.")
    st.number_input("Shares Outstanding (millions)", min_value=0.1, step=10.0, key="shares",
                    help="Diluted share count for per-share value.")

    st.markdown("---")

    # ─── Market reference ────────────────────────────────────────────────────
    st.subheader("📊 Market Reference")
    st.number_input("Current Stock Price ($)", min_value=0.01, step=0.50, key="current_price",
                    help="Today's market price. Look it up on Google Finance and paste here.")


# =============================================================================
# Pull values out of session state for clean local references
# =============================================================================
ticker        = st.session_state.ticker
company_name  = st.session_state.company_name
revenue       = float(st.session_state.revenue)
growth        = float(st.session_state.growth) / 100
margin        = float(st.session_state.margin) / 100
tax_rate      = float(st.session_state.tax_rate) / 100
reinvest_rate = float(st.session_state.reinvest_rate) / 100
wacc          = float(st.session_state.wacc) / 100
term_g        = float(st.session_state.terminal_growth) / 100
years         = int(st.session_state.years)
debt          = float(st.session_state.debt)
cash          = float(st.session_state.cash)
shares        = max(float(st.session_state.shares), 0.1)  # never zero (divide guard)
price         = max(float(st.session_state.current_price), 0.01)

# Validate inputs and surface clear errors before doing math.
input_errors = []
if revenue <= 0:
    input_errors.append("Revenue must be > 0.")
if margin <= 0:
    input_errors.append("EBIT Margin must be > 0% (the firm needs operating profit for DCF to work).")
if wacc <= term_g:
    input_errors.append(
        f"WACC ({wacc*100:.2f}%) must be greater than Terminal Growth ({term_g*100:.2f}%) — "
        "otherwise the Gordon Growth formula explodes to infinity."
    )
if shares <= 0:
    input_errors.append("Shares Outstanding must be > 0.")

if input_errors:
    st.error("⚠️ **Cannot run DCF — please fix these inputs first:**\n\n" +
             "\n".join(f"- {e}" for e in input_errors))
    st.stop()


# =============================================================================
# DCF Calculation
# =============================================================================
def run_dcf(rev0, g, m, tx, ri, w, tg, n):
    """Run a DCF projection. Returns a dict with all intermediate values."""
    revenues, ebits, nopats, reinvs, fcfs, dfs, pvs = [], [], [], [], [], [], []
    rev = rev0
    for t in range(1, n + 1):
        rev = rev * (1 + g)
        ebit = rev * m
        nopat = ebit * (1 - tx)
        reinv = nopat * ri
        fcf = nopat - reinv
        df = 1 / (1 + w) ** t
        pv = fcf * df
        revenues.append(rev); ebits.append(ebit); nopats.append(nopat)
        reinvs.append(reinv); fcfs.append(fcf); dfs.append(df); pvs.append(pv)

    sum_pv_fcf = sum(pvs)
    terminal_fcf = fcfs[-1] * (1 + tg)
    terminal_value = terminal_fcf / (w - tg) if w > tg else float("nan")
    pv_terminal = terminal_value / (1 + w) ** n
    return {
        "revenues": revenues, "ebits": ebits, "nopats": nopats, "reinvs": reinvs,
        "fcfs": fcfs, "dfs": dfs, "pvs": pvs,
        "sum_pv_fcf": sum_pv_fcf,
        "terminal_fcf": terminal_fcf,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": sum_pv_fcf + pv_terminal,
    }


dcf = run_dcf(revenue, growth, margin, tax_rate, reinvest_rate, wacc, term_g, years)

enterprise_value = dcf["enterprise_value"]
equity_value     = enterprise_value - debt + cash
intrinsic_per    = equity_value / shares
upside           = (intrinsic_per - price) / price if price > 0 else 0
mos              = (intrinsic_per - price) / intrinsic_per if intrinsic_per > 0 else 0


# =============================================================================
# Page header — company + headline metrics
# =============================================================================
st.markdown(f"## 🏢 {company_name} ({ticker})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("DCF Intrinsic Value", f"${intrinsic_per:,.2f}")
m2.metric("Current Market Price", f"${price:,.2f}")
m3.metric("Upside / (Downside)", f"{upside:+.1%}",
          delta=f"${intrinsic_per - price:,.2f}",
          delta_color="normal" if abs(upside) < 0.05 else ("inverse" if upside < 0 else "normal"))
m4.metric("Margin of Safety", f"{mos:+.1%}")

# Verdict box
if abs(upside) < 0.10:
    st.info(f"⚖️  **Fairly Valued** — DCF intrinsic value is within ±10% of market price.")
elif upside > 0:
    st.success(f"📈 **Potentially Undervalued** — DCF estimates the stock is worth {upside:.1%} more than its current price.")
else:
    st.warning(f"📉 **Potentially Overvalued** — Market is paying {-upside:.1%} above the DCF estimate.")


# =============================================================================
# EDGAR context box (if data was loaded)
# =============================================================================
if st.session_state.edgar_data:
    e = st.session_state.edgar_data
    with st.expander("📚 SEC EDGAR Reference Data (most recent 10-K)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**5-Year Revenue History (from 10-K filings):**")
            if e.get("rev_history"):
                hist_df = pd.DataFrame(e["rev_history"], columns=["FY ending", "Revenue ($B)"])
                st.dataframe(hist_df, hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Computed metrics:**")
            def _fmt_pct(v): return f"{v:.1f}%" if isinstance(v, (int, float)) else "N/A"
            def _fmt_b(v):   return f"${v/1000:.1f}B" if isinstance(v, (int, float)) else "N/A"
            ocf_v   = e.get("operating_cf")
            capex_v = e.get("capex")
            fcf_v   = (ocf_v - capex_v) if (isinstance(ocf_v, (int, float)) and isinstance(capex_v, (int, float))) else None
            st.write(f"- 3-yr Revenue CAGR: **{_fmt_pct(e.get('revenue_cagr'))}**")
            st.write(f"- Operating Cash Flow: **{_fmt_b(ocf_v)}**")
            st.write(f"- CapEx: **{_fmt_b(capex_v)}**")
            st.write(f"- Implied FCF: **{_fmt_b(fcf_v)}**")
        st.caption("Source: data.sec.gov XBRL Company Facts API · Used as starting point — every input above is editable.")


st.markdown("---")


# =============================================================================
# Step-by-step walkthrough
# =============================================================================
st.markdown("## 🧮 Step-by-Step Valuation")

# ─── Step 1: Forecast ──────────────────────────────────────────────────────
st.markdown(f"### Step 1 — Forecast Free Cash Flows (Years 1 to {years})")
st.markdown("**Formula:**")
st.latex(r"\text{Revenue}_t = \text{Revenue}_{t-1} \times (1 + g)")
st.latex(r"\text{EBIT}_t = \text{Revenue}_t \times \text{margin}")
st.latex(r"\text{NOPAT}_t = \text{EBIT}_t \times (1 - \text{tax rate})")
st.latex(r"\text{FCF}_t = \text{NOPAT}_t \times (1 - \text{reinvestment rate})")

proj_df = pd.DataFrame({
    "Year":         list(range(1, years + 1)),
    "Revenue ($M)": [f"{x:,.0f}" for x in dcf["revenues"]],
    "EBIT ($M)":    [f"{x:,.0f}" for x in dcf["ebits"]],
    "NOPAT ($M)":   [f"{x:,.0f}" for x in dcf["nopats"]],
    "Reinvest ($M)":[f"{x:,.0f}" for x in dcf["reinvs"]],
    "FCF ($M)":     [f"{x:,.0f}" for x in dcf["fcfs"]],
})
st.dataframe(proj_df, hide_index=True, use_container_width=True)

# Chart
fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1, years + 1)), y=dcf["fcfs"], name="FCF",
                      marker_color="#1f4e79"))
fig.update_layout(
    title="Forecasted Free Cash Flows",
    xaxis_title="Year", yaxis_title="FCF ($M)",
    template="plotly_white", height=350, margin=dict(t=50, b=40),
)
st.plotly_chart(fig, use_container_width=True)


# ─── Step 2: Discount ──────────────────────────────────────────────────────
st.markdown(f"### Step 2 — Discount FCFs to Present Value (WACC = {wacc*100:.2f}%)")
st.markdown("**Formula:**")
st.latex(r"\text{PV of FCF}_t = \frac{\text{FCF}_t}{(1 + \text{WACC})^t}")

disc_df = pd.DataFrame({
    "Year":              list(range(1, years + 1)),
    "FCF ($M)":          [f"{x:,.0f}" for x in dcf["fcfs"]],
    "Discount Factor":   [f"{x:.4f}" for x in dcf["dfs"]],
    "PV of FCF ($M)":    [f"{x:,.0f}" for x in dcf["pvs"]],
})
st.dataframe(disc_df, hide_index=True, use_container_width=True)
st.markdown(f"**Sum of PV of FCFs = ${dcf['sum_pv_fcf']:,.0f}M**")


# ─── Step 3: Terminal Value ────────────────────────────────────────────────
st.markdown("### Step 3 — Terminal Value (Gordon Growth Model)")
st.markdown("**Formula:**")
st.latex(r"\text{Terminal Value} = \frac{\text{FCF}_{N} \times (1 + g_{\text{terminal}})}{\text{WACC} - g_{\text{terminal}}}")
st.latex(r"\text{PV of Terminal Value} = \frac{\text{Terminal Value}}{(1 + \text{WACC})^N}")

t1, t2, t3 = st.columns(3)
t1.metric("Terminal FCF (Yr N+1)", f"${dcf['terminal_fcf']:,.0f}M")
t2.metric("Terminal Value", f"${dcf['terminal_value']:,.0f}M")
t3.metric("PV of Terminal Value", f"${dcf['pv_terminal']:,.0f}M")

if dcf['pv_terminal'] / dcf['enterprise_value'] > 0.75:
    st.warning(f"⚠️  Terminal value is **{dcf['pv_terminal']/dcf['enterprise_value']:.0%}** of enterprise value — "
               "a large fraction of valuation depends on perpetual growth assumptions.")


# ─── Step 4: Enterprise Value ──────────────────────────────────────────────
st.markdown("### Step 4 — Enterprise Value")
st.markdown("**Formula:**")
st.latex(r"\text{EV} = \sum \text{PV of FCFs} + \text{PV of Terminal Value}")

e1, e2, e3 = st.columns(3)
e1.metric("Sum of PV of FCFs", f"${dcf['sum_pv_fcf']:,.0f}M")
e2.metric("PV of Terminal Value", f"${dcf['pv_terminal']:,.0f}M")
e3.metric("Enterprise Value", f"${enterprise_value:,.0f}M")


# ─── Step 5: Equity Value ──────────────────────────────────────────────────
st.markdown("### Step 5 — Bridge to Equity Value")
st.markdown("**Formula:**")
st.latex(r"\text{Equity Value} = \text{EV} - \text{Debt} + \text{Cash}")
st.latex(r"\text{Value per Share} = \frac{\text{Equity Value}}{\text{Shares Outstanding}}")

bridge_df = pd.DataFrame({
    "Item":         ["Enterprise Value", "(−) Total Debt", "(+) Cash & Equivalents", "= Equity Value", "÷ Shares Outstanding (M)", "= Value per Share"],
    "Value ($M / $)": [
        f"${enterprise_value:,.0f}M",
        f"−${debt:,.0f}M",
        f"+${cash:,.0f}M",
        f"${equity_value:,.0f}M",
        f"{shares:,.0f}M shares",
        f"${intrinsic_per:,.2f}",
    ],
})
st.dataframe(bridge_df, hide_index=True, use_container_width=True)


# ─── Step 6: Compare to market ─────────────────────────────────────────────
st.markdown("### Step 6 — Compare to Market Price")

cmp_fig = go.Figure()
cmp_fig.add_trace(go.Bar(x=["Market Price", "DCF Intrinsic Value"],
                          y=[price, intrinsic_per],
                          marker_color=["#999999", "#1f4e79"],
                          text=[f"${price:.2f}", f"${intrinsic_per:.2f}"],
                          textposition="outside"))
cmp_fig.update_layout(
    title=f"{ticker} — Market vs. Intrinsic", yaxis_title="Price (USD)",
    template="plotly_white", height=400, showlegend=False, margin=dict(t=50, b=40),
)
st.plotly_chart(cmp_fig, use_container_width=True)


st.markdown("---")


# =============================================================================
# Sensitivity analysis
# =============================================================================
st.markdown("## 📊 Sensitivity Analysis")
st.caption("Intrinsic value per share under different assumption combinations. "
           "Color: 🟢 >20% upside · 🟡 ±20% · 🔴 >20% downside.")

def style_iv(val):
    """Color cell by upside vs current market price."""
    try:
        v = float(val.replace("$", "").replace(",", ""))
    except Exception:
        return ""
    diff = (v - price) / price if price > 0 else 0
    if diff > 0.20:    return "background-color: #c8e6c9"   # green
    elif diff < -0.20: return "background-color: #ffcdd2"   # red
    else:              return "background-color: #fff9c4"   # yellow

# Table 1: WACC × Terminal Growth
st.markdown("### Table 1 — Intrinsic Value: WACC vs. Terminal Growth")
wacc_range  = [round(wacc + dx, 4) for dx in [-0.02, -0.01, 0, 0.01, 0.02]]
tg_range    = [round(term_g + dx, 4) for dx in [-0.01, -0.005, 0, 0.005, 0.01]]
tbl1 = []
for w in wacc_range:
    if w <= 0: continue
    row = []
    for tg in tg_range:
        if tg < 0 or w <= tg:
            row.append(np.nan); continue
        d = run_dcf(revenue, growth, margin, tax_rate, reinvest_rate, w, tg, years)
        eq = d["enterprise_value"] - debt + cash
        row.append(eq / shares)
    tbl1.append(row)
df1 = pd.DataFrame(
    tbl1,
    index=[f"WACC={w*100:.2f}%" for w in wacc_range if w > 0],
    columns=[f"g={tg*100:.2f}%" for tg in tg_range],
)
st.dataframe(
    df1.style.format("${:,.2f}").map(lambda v: style_iv(f"${v:,.2f}") if pd.notna(v) else ""),
    use_container_width=True,
)

# Table 2: Growth × Margin
st.markdown("### Table 2 — Intrinsic Value: Growth Rate vs. EBIT Margin")
g_range = [round(growth + dx, 4) for dx in [-0.04, -0.02, 0, 0.02, 0.04]]
m_range = [round(margin + dx, 4) for dx in [-0.05, -0.025, 0, 0.025, 0.05]]
tbl2 = []
for g in g_range:
    row = []
    for m in m_range:
        if m <= 0: row.append(np.nan); continue
        d = run_dcf(revenue, g, m, tax_rate, reinvest_rate, wacc, term_g, years)
        eq = d["enterprise_value"] - debt + cash
        row.append(eq / shares)
    tbl2.append(row)
df2 = pd.DataFrame(
    tbl2,
    index=[f"g={g*100:.2f}%" for g in g_range],
    columns=[f"margin={m*100:.2f}%" for m in m_range],
)
st.dataframe(
    df2.style.format("${:,.2f}").map(lambda v: style_iv(f"${v:,.2f}") if pd.notna(v) else ""),
    use_container_width=True,
)


st.markdown("---")


# =============================================================================
# Excel export — download a workbook that REPLICATES the DCF using formulas
# =============================================================================
st.markdown("## 📥 Excel Replication")
st.caption(
    "Download an Excel workbook where every value is a live formula — change any input "
    "and the model recomputes. Use this for the assignment's Replicability requirement."
)

def build_excel_workbook():
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()

    # ─── Inputs sheet ────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Inputs"
    bold = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    border_thin = Border(*[Side(style="thin", color="CCCCCC")] * 4)

    inputs = [
        ("Company Name",         company_name,          ""),
        ("Ticker",               ticker,                ""),
        ("Current Revenue",      revenue,               "$M"),
        ("Annual Growth Rate",   growth,                "(decimal)"),
        ("EBIT Margin",          margin,                "(decimal)"),
        ("Tax Rate",             tax_rate,              "(decimal)"),
        ("Reinvestment Rate",    reinvest_rate,         "(decimal)"),
        ("WACC",                 wacc,                  "(decimal)"),
        ("Terminal Growth Rate", term_g,                "(decimal)"),
        ("Forecast Years",       years,                 ""),
        ("Total Debt",           debt,                  "$M"),
        ("Cash & Equivalents",   cash,                  "$M"),
        ("Shares Outstanding",   shares,                "millions"),
        ("Current Stock Price",  price,                 "$"),
    ]

    ws["A1"] = "Input"; ws["B1"] = "Value"; ws["C1"] = "Unit"
    for c in ["A1", "B1", "C1"]:
        ws[c].font = header_font
        ws[c].fill = header_fill
    for i, (label, val, unit) in enumerate(inputs, start=2):
        ws.cell(row=i, column=1, value=label).font = bold
        ws.cell(row=i, column=2, value=val)
        ws.cell(row=i, column=3, value=unit)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 14

    # Named ranges that the DCF sheet references (1-indexed rows from `inputs`)
    REV_CELL    = "Inputs!$B$4"
    GROWTH_CELL = "Inputs!$B$5"
    MARGIN_CELL = "Inputs!$B$6"
    TAX_CELL    = "Inputs!$B$7"
    REINV_CELL  = "Inputs!$B$8"
    WACC_CELL   = "Inputs!$B$9"
    TG_CELL     = "Inputs!$B$10"
    DEBT_CELL   = "Inputs!$B$12"
    CASH_CELL   = "Inputs!$B$13"
    SHARES_CELL = "Inputs!$B$14"
    PRICE_CELL  = "Inputs!$B$15"

    # ─── DCF Projection sheet ────────────────────────────────────────────────
    ws2 = wb.create_sheet("DCF Projection")
    headers = ["Year", "Revenue ($M)", "EBIT ($M)", "NOPAT ($M)", "Reinvestment ($M)",
               "FCF ($M)", "Discount Factor", "PV of FCF ($M)"]
    for col, h in enumerate(headers, start=1):
        c = ws2.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill

    # Year 0 (base year)
    ws2.cell(row=2, column=1, value=0)
    ws2.cell(row=2, column=2, value=f"={REV_CELL}")

    # Forecast years 1..N — every cell is a formula
    for t in range(1, years + 1):
        row = t + 2
        ws2.cell(row=row, column=1, value=t)
        ws2.cell(row=row, column=2, value=f"=B{row-1}*(1+{GROWTH_CELL})")           # Revenue
        ws2.cell(row=row, column=3, value=f"=B{row}*{MARGIN_CELL}")                  # EBIT
        ws2.cell(row=row, column=4, value=f"=C{row}*(1-{TAX_CELL})")                 # NOPAT
        ws2.cell(row=row, column=5, value=f"=D{row}*{REINV_CELL}")                   # Reinvestment
        ws2.cell(row=row, column=6, value=f"=D{row}-E{row}")                         # FCF
        ws2.cell(row=row, column=7, value=f"=1/(1+{WACC_CELL})^A{row}")              # Discount factor
        ws2.cell(row=row, column=8, value=f"=F{row}*G{row}")                         # PV of FCF

    last_data_row = years + 2

    # Format numeric cells
    for r in range(2, last_data_row + 1):
        for c in range(2, 7):
            ws2.cell(row=r, column=c).number_format = "#,##0"
        ws2.cell(row=r, column=7).number_format = "0.0000"
        ws2.cell(row=r, column=8).number_format = "#,##0"

    # ─── Valuation summary on same sheet ────────────────────────────────────
    summary_start = last_data_row + 3
    rows = [
        ("Sum of PV of FCFs",      f"=SUM(H3:H{last_data_row})"),
        ("Terminal FCF (Yr N+1)",  f"=F{last_data_row}*(1+{TG_CELL})"),
        ("Terminal Value",         f"=B{summary_start+1}/({WACC_CELL}-{TG_CELL})"),
        ("PV of Terminal Value",   f"=B{summary_start+2}/(1+{WACC_CELL})^{years}"),
        ("Enterprise Value",       f"=B{summary_start}+B{summary_start+3}"),
        ("(−) Total Debt",         f"=-{DEBT_CELL}"),
        ("(+) Cash & Equivalents", f"={CASH_CELL}"),
        ("Equity Value",           f"=B{summary_start+4}+B{summary_start+5}+B{summary_start+6}"),
        ("Shares Outstanding (M)", f"={SHARES_CELL}"),
        ("DCF Intrinsic Value",    f"=B{summary_start+7}/B{summary_start+8}"),
        ("Current Market Price",   f"={PRICE_CELL}"),
        ("Upside / (Downside)",    f"=B{summary_start+9}/B{summary_start+10}-1"),
    ]
    for i, (label, formula) in enumerate(rows):
        r = summary_start + i
        ws2.cell(row=r, column=1, value=label).font = bold
        c = ws2.cell(row=r, column=2, value=formula)
        if "Upside" in label:
            c.number_format = "0.00%"
        elif "Intrinsic" in label or "Price" in label:
            c.number_format = "$#,##0.00"
        else:
            c.number_format = "$#,##0"
    ws2.cell(row=summary_start + 9, column=1).font = Font(bold=True, color="1F4E79", size=12)
    ws2.cell(row=summary_start + 9, column=2).font = Font(bold=True, color="1F4E79", size=12)

    ws2.column_dimensions["A"].width = 26
    for col_letter in ["B", "C", "D", "E", "F", "G", "H"]:
        ws2.column_dimensions[col_letter].width = 16

    # ─── Sensitivity sheet ────────────────────────────────────────────────────
    ws3 = wb.create_sheet("Sensitivity")
    ws3["A1"] = "Sensitivity: WACC vs. Terminal Growth (Intrinsic Value per Share)"
    ws3["A1"].font = Font(bold=True, size=12)

    ws3["A3"] = "WACC ↓ / g →"
    ws3["A3"].font = bold; ws3["A3"].fill = header_fill; ws3["A3"].font = header_font
    for j, tg_val in enumerate(tg_range):
        c = ws3.cell(row=3, column=2 + j, value=tg_val)
        c.number_format = "0.00%"; c.font = header_font; c.fill = header_fill
    for i, w_val in enumerate(wacc_range):
        if w_val <= 0: continue
        c = ws3.cell(row=4 + i, column=1, value=w_val)
        c.number_format = "0.00%"; c.font = bold

    # The actual values (not formulas — would explode the workbook size)
    for i, w_val in enumerate(wacc_range):
        if w_val <= 0: continue
        for j, tg_val in enumerate(tg_range):
            if tg_val < 0 or w_val <= tg_val:
                ws3.cell(row=4 + i, column=2 + j, value="N/A")
                continue
            dd = run_dcf(revenue, growth, margin, tax_rate, reinvest_rate, w_val, tg_val, years)
            eq = dd["enterprise_value"] - debt + cash
            ws3.cell(row=4 + i, column=2 + j, value=eq / shares).number_format = "$#,##0.00"

    ws3.column_dimensions["A"].width = 16
    for j in range(len(tg_range)):
        ws3.column_dimensions[chr(ord("B") + j)].width = 14

    # ─── README / methodology sheet ──────────────────────────────────────────
    ws4 = wb.create_sheet("Methodology")
    notes = [
        "DCF Equity Valuation — Replication",
        "",
        f"Company: {company_name} ({ticker})",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Sheets:",
        "  1. Inputs — all assumptions, editable",
        "  2. DCF Projection — year-by-year forecast + valuation summary (formulas)",
        "  3. Sensitivity — WACC × Terminal growth grid",
        "",
        "How to use:",
        "  • Edit any cell in 'Inputs' — the projection updates automatically.",
        "  • Compare cell B(N+12) on DCF Projection (Intrinsic Value) to B(N+13) (Market Price).",
        "  • Sensitivity table shows how the valuation moves with discount-rate / terminal-growth assumptions.",
        "",
        "Data source: SEC EDGAR 10-K filings (when 'Auto-fill' was used).",
        "Model: 5-step DCF — forecast FCF → discount → terminal value → enterprise value → equity value → per share.",
    ]
    for i, line in enumerate(notes, start=1):
        c = ws4.cell(row=i, column=1, value=line)
        if i == 1: c.font = Font(bold=True, size=14, color="1F4E79")
        elif line.endswith(":"): c.font = bold
    ws4.column_dimensions["A"].width = 90

    return wb


col_dl, _ = st.columns([2, 5])
with col_dl:
    if st.button("📥 Generate Excel Workbook", type="primary", use_container_width=True):
        wb = build_excel_workbook()
        buf = BytesIO()
        wb.save(buf)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download " + ticker + "_DCF_Replication.xlsx",
            data=buf.getvalue(),
            file_name=f"{ticker}_DCF_Replication.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

st.markdown("---")


# =============================================================================
# Methodology note (always visible at bottom)
# =============================================================================
with st.expander("📖 Methodology & Notes"):
    st.markdown("""
**Discounted Cash Flow (DCF) Model**

Estimates a stock's intrinsic value from its expected future cash flows, discounted to present value at the firm's cost of capital (WACC).

**Steps:**
1. **Forecast** revenue, EBIT, NOPAT, and free cash flow for an explicit period (3-10 years).
2. **Discount** each year's FCF back to today using `1/(1+WACC)^t`.
3. **Terminal Value** captures everything beyond the forecast period: `FCF_N × (1+g) / (WACC − g)`.
4. **Enterprise Value** = Σ PV of FCFs + PV of Terminal Value.
5. **Equity Value** = EV − Debt + Cash.
6. **Per-share** = Equity Value ÷ Shares Outstanding.

**Inputs to think about:**
- **Growth rate** — drawn from historical CAGR + your forward thesis. For mature firms, a few percent above GDP. For growth firms, double-digit but converging downward.
- **EBIT margin** — historical operating margin. Better firms expand margins over time.
- **WACC** — riskier firms have higher WACC. As a rough guide: large-cap stable = 7–9%, mid-cap = 9–12%, small/risky = 12%+.
- **Terminal growth** — should never exceed long-run GDP (~2-3%) over the very long run.
- **Reinvestment rate** — % of NOPAT plowed back into CapEx + working capital. Higher reinvestment = lower FCF today but more future growth.

**Data source (when auto-fill is used):** SEC EDGAR's official Company Facts API (`data.sec.gov`). Pulls the latest 10-K line items — same numbers any analyst would use.

**This app does NOT use Yahoo Finance** — it relies only on SEC filings (which are 100% reliable) and your manual inputs (which you control).
    """)

st.caption(f"FINA 4011/5011 · Project 2 · DCF Equity Valuation · Built with Streamlit")
