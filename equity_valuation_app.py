# =============================================================================
# DCF Equity Valuation — Streamlit App
# =============================================================================
# Discounted Cash Flow valuation tool with optional SEC EDGAR auto-fill,
# CAPM-based WACC build-up, multi-stage growth, margin expansion path,
# and sensitivity tables.
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

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Equity Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 DCF Equity Valuation")
st.caption("Discounted Cash Flow valuation tool with SEC filings auto-fill")

# =============================================================================
# SEC EDGAR — optional auto-fill helpers
# Uses the official SEC API: free, no key, not rate-limited beyond UA requirement.
# =============================================================================
_SEC_HEADERS = {
    "User-Agent": "DCF-Valuation-App contact@dcf-app.local",
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
    "revenue":         391000.0,   # $M (AAPL FY24 starting point)
    "growth":          8.0,        # % — modest growth above GDP
    "margin":          30.0,       # % EBIT margin
    "tax_rate":        21.0,       # %
    "reinvest_rate":   15.0,       # % of NOPAT — mature firms reinvest less
    "wacc":            8.5,        # % — large-cap blue chip
    "terminal_growth": 2.5,        # %
    "years":           5,
    "debt":            107000.0,   # $M
    "cash":            65000.0,    # $M
    "shares":          15300.0,    # millions
    "current_price":   210.00,     # $
    "edgar_data":      None,
    "loaded_ticker":   "AAPL",
    "edgar_msg":       None,
    # ── Advanced modeling toggles ────────────────────────────────────────
    "wacc_mode":       "Direct",   # "Direct" | "CAPM build-up"
    "rf":              4.00,       # Risk-free rate (10-yr Treasury), %
    "erp":             5.50,       # Equity risk premium, %
    "beta":            1.00,       # Levered beta
    "rd_pretax":       5.00,       # Pre-tax cost of debt, %
    "weight_debt":     20.0,       # Target debt weight, % (rest = equity)
    "growth_mode":     "Constant", # "Constant" | "3-stage (high → fade → terminal)"
    "high_growth":     15.0,
    "high_years":      3,
    "fade_years":      4,
    "margin_mode":     "Constant", # "Constant" | "Linear expansion"
    "margin_terminal": 30.0,
    # Terminal value method
    "tv_method":       "Gordon Growth", # "Gordon Growth" | "Exit Multiple"
    "exit_multiple":   12.0,       # EV / EBITDA at year N
    # Discounting convention
    "mid_year":        False,      # mid-year convention (more accurate)
    # Verdict thresholds
    "fair_band":       10.0,       # % band around price that counts as "fairly valued"
    # Working capital change as a separate driver
    "nwc_pct":         2.0,        # change in NWC as % of revenue (deducted from FCF)
}
# Process pending reset BEFORE any widget instantiates — Streamlit doesn't
# allow modifying a widget's session_state after the widget has been rendered
# in the same script run. Reset button just sets a flag; the actual restore
# happens here at the top of the next run.
if st.session_state.pop("_pending_reset", False):
    for k in list(st.session_state.keys()):
        if k in DEFAULTS:
            del st.session_state[k]

# Initialize defaults for any keys not yet in session_state
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_edgar_autofill():
    """Fetch EDGAR data for the current ticker and apply it to all inputs.

    Used by both the explicit "Auto-fill" button AND the ticker on_change callback,
    so changing the ticker (Tab/Enter) immediately refreshes the model.
    Sets st.session_state.edgar_msg so the sidebar can render the result inline.
    """
    sym = (st.session_state.ticker or "").strip().upper()
    if not sym:
        return
    edgar = fetch_edgar_fundamentals(sym)
    if not edgar:
        st.session_state.edgar_msg = ("error",
            f"❌ Ticker **{sym}** not found in EDGAR. Use a US-listed ticker, "
            "or enter values manually below.")
        return
    edgar_rev = edgar.get("revenue")
    if not edgar_rev or edgar_rev <= 0:
        st.session_state.edgar_data = None
        st.session_state.edgar_msg = ("error",
            f"⛔ **{edgar.get('name', sym)}** appears to be a **pre-revenue company** "
            "(no operating revenue in latest 10-K). DCF requires positive cash flows. "
            "Try a revenue-generating ticker (AAPL, MSFT, JNJ), or use asset-based valuation, "
            "or enter values manually below.")
        return

    def _set(key, val, min_val=0):
        if val is None: return
        try: v = float(val)
        except (TypeError, ValueError): return
        if v <= min_val: return
        st.session_state[key] = round(v, 1) if isinstance(v, float) and abs(v) < 1000 else round(v)

    if edgar.get("name"):
        st.session_state.company_name = edgar["name"]
    _set("revenue", edgar.get("revenue"))
    _set("margin",  edgar.get("ebit_margin"))
    _set("shares",  edgar.get("shares"))
    _set("growth",  edgar.get("revenue_cagr"))
    if edgar.get("cash") is not None and edgar.get("cash") >= 0:
        st.session_state.cash = round(edgar["cash"], 0)
    if edgar.get("debt") is not None and edgar.get("debt") >= 0:
        st.session_state.debt = round(edgar["debt"], 0)
    if edgar.get("capex") and edgar.get("revenue") and edgar.get("ebit_margin"):
        nopat = edgar["revenue"] * (edgar["ebit_margin"]/100) * (1 - st.session_state.tax_rate/100)
        if nopat > 0:
            ri = min(100, edgar["capex"] / nopat * 100)
            if ri > 0:
                st.session_state.reinvest_rate = round(ri, 1)
    st.session_state.edgar_data = edgar
    st.session_state.loaded_ticker = sym
    fy = edgar.get("fy_end", "?")
    if edgar.get("ebit_margin") is None:
        st.session_state.edgar_msg = ("warning",
            f"⚠️ Loaded {edgar.get('name', sym)} (FY{fy}), but operating margin "
            "couldn't be determined (likely a loss year). DCF will reflect your manual "
            "margin assumption, not the company's actuals.")
    elif edgar.get("missing"):
        st.session_state.edgar_msg = ("warning",
            f"⚠️ Loaded {edgar.get('name', sym)} (FY{fy}). EDGAR didn't have: "
            f"**{', '.join(edgar['missing'])}** — left at default. Edit fields below if needed.")
    else:
        st.session_state.edgar_msg = ("success",
            f"✅ Loaded {edgar.get('name', sym)} (FY{fy})")


def on_ticker_change():
    """Streamlit callback fired when the ticker text_input commits.
    Only auto-fetch if the ticker actually changed since last load — avoids
    a wasted EDGAR call when the user just tabs through the field.
    """
    cur = (st.session_state.get("ticker") or "").strip().upper()
    loaded = (st.session_state.get("loaded_ticker") or "").strip().upper()
    if cur and cur != loaded:
        apply_edgar_autofill()


# =============================================================================
# Sidebar — all inputs
# =============================================================================
with st.sidebar:
    st.header("📋 Inputs")

    # ─── Ticker + EDGAR auto-fill ────────────────────────────────────────────
    st.subheader("🏢 Company")
    st.text_input(
        "Ticker", key="ticker",
        on_change=on_ticker_change,
        help="US-listed ticker (e.g. AAPL, MSFT, JNJ). Press Enter or Tab to auto-load EDGAR data.",
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        if st.button("🔄 Re-fetch from EDGAR", use_container_width=True,
                     help="Re-pull latest 10-K data from SEC EDGAR for the current ticker"):
            apply_edgar_autofill()
            st.rerun()
    with col_b:
        if st.button("🧹 Reset", use_container_width=True, help="Reset all inputs to defaults"):
            # Defer the actual reset to the top of the next run — see notes there
            st.session_state["_pending_reset"] = True
            st.session_state["edgar_msg"] = None
            st.rerun()

    # ── Render the last EDGAR-fetch message inline (set by the callback / button) ──
    msg = st.session_state.get("edgar_msg")
    if msg:
        kind, text = msg
        {"success": st.success, "warning": st.warning, "error": st.error}[kind](text)

    # ── Mismatch banner: ticker changed but data is stale (network failed?) ──
    cur = (st.session_state.ticker or "").strip().upper()
    loaded = (st.session_state.get("loaded_ticker") or "").strip().upper()
    if cur and loaded and cur != loaded:
        st.info(
            f"📌 Inputs below currently reflect **{loaded}**. "
            f"Press Enter or click 'Re-fetch from EDGAR' to load **{cur}**."
        )

    st.text_input("Company Name", key="company_name")

    st.markdown("---")

    # ─── Financial inputs ────────────────────────────────────────────────────
    st.subheader("💰 Financial Inputs")
    st.caption("All dollar values in millions ($M)")

    # Dollar amounts stay as number_input — sliders are too imprecise for $100B+
    st.number_input("Current Revenue ($M)", min_value=0.0, step=100.0, key="revenue",
                    help="Most recent annual revenue. EDGAR fills with latest 10-K value.")

    # ── Margin: simple slider, with optional linear-expansion mode ──
    st.radio("Margin assumption", options=["Constant", "Linear expansion"],
             key="margin_mode", horizontal=True,
             help="Constant = flat EBIT margin every year. Linear expansion = margin moves linearly from Year 1 to Year N (good for growth firms with operating leverage).")
    st.slider("EBIT Margin (Year 1) (%)", min_value=0.0, max_value=100.0, step=0.5, key="margin",
              format="%.1f%%",
              help="Operating Income ÷ Revenue. Higher margin → more value. EDGAR auto-fills from 10-K.")
    if st.session_state.margin_mode == "Linear expansion":
        st.slider("Terminal EBIT Margin (Year N) (%)", min_value=0.0, max_value=100.0, step=0.5,
                  key="margin_terminal", format="%.1f%%",
                  help="Margin in the final forecast year. Linear interpolation between Year 1 and Year N.")

    st.slider("Tax Rate (%)", min_value=0.0, max_value=50.0, step=0.5, key="tax_rate",
              format="%.1f%%",
              help="Effective corporate tax rate. US federal statutory is 21%; effective often 15-25%.")
    st.slider("Reinvestment Rate (%)", min_value=0.0, max_value=100.0, step=1.0, key="reinvest_rate",
              format="%.0f%%",
              help="% of NOPAT reinvested in CapEx + working capital. Higher → less FCF today, more future growth.")

    st.slider("Δ Working Capital (% of Δ Revenue)", min_value=0.0, max_value=30.0, step=0.5,
              key="nwc_pct", format="%.1f%%",
              help="Net working capital tied up per dollar of revenue growth. Subtracted from FCF. Typical 1-5%.")

    st.markdown("---")

    # ─── Growth ──────────────────────────────────────────────────────────────
    st.subheader("📈 Growth")
    st.radio("Growth path", options=["Constant", "3-stage (high → fade → terminal)"],
             key="growth_mode", horizontal=False,
             help="Constant = same growth rate every year. 3-stage = high-growth period, then linear fade, then stable terminal.")

    if st.session_state.growth_mode == "Constant":
        st.slider("Annual Growth Rate (%)", min_value=-20.0, max_value=50.0, step=0.5, key="growth",
                  format="%.1f%%",
                  help="Revenue growth during the forecast period. EDGAR fills with 3-yr CAGR.")
    else:
        st.slider("High-growth Rate (%)", min_value=-10.0, max_value=80.0, step=0.5,
                  key="high_growth", format="%.1f%%",
                  help="Growth rate during the high-growth phase. Often 10-25% for growth companies.")
        st.slider("High-growth Years", min_value=1, max_value=10,
                  key="high_years",
                  help="Number of years at the high-growth rate before fading begins.")
        st.slider("Fade-down Years", min_value=1, max_value=10,
                  key="fade_years",
                  help="Number of years over which growth linearly fades from high-growth rate to terminal rate.")
        # In 3-stage mode, the simple `growth` slider is unused — show it disabled for transparency
        st.caption("(The simple 'Annual Growth Rate' slider is ignored in 3-stage mode.)")

    st.slider("Terminal Growth Rate (%)", min_value=0.0, max_value=10.0, step=0.25,
              key="terminal_growth", format="%.2f%%",
              help="Perpetual growth after the forecast period. Cap at long-run GDP growth (~2-3%).")

    st.slider("Forecast Years", min_value=3, max_value=15, key="years",
              help="Length of the explicit forecast. Standard is 5-10 years.")

    # ── Terminal Value method ──
    st.radio("Terminal Value method", options=["Gordon Growth", "Exit Multiple"],
             key="tv_method", horizontal=True,
             help="Gordon Growth = perpetuity formula using terminal growth rate. Exit Multiple = EV/EBITDA × terminal-year EBITDA (assumes the firm is sold at year N).")
    if st.session_state.tv_method == "Exit Multiple":
        st.slider("Exit EV/EBITDA Multiple", min_value=2.0, max_value=40.0, step=0.5,
                  key="exit_multiple", format="%.1fx",
                  help="EV / EBITDA multiple at exit. Mature: 8-12x. Growth: 15-25x. Premium tech: 25x+.")

    st.markdown("---")

    # ─── WACC: direct or CAPM build-up ───────────────────────────────────────
    st.subheader("💸 Discount Rate (WACC)")
    st.radio("WACC mode", options=["Direct", "CAPM build-up"],
             key="wacc_mode", horizontal=True,
             help="Direct = enter WACC directly. CAPM build-up = compute WACC from risk-free rate, beta, equity risk premium, cost of debt, and capital structure (textbook formula).")

    if st.session_state.wacc_mode == "Direct":
        st.slider("WACC (%)", min_value=1.0, max_value=30.0, step=0.25, key="wacc",
                  format="%.2f%%",
                  help="Weighted Average Cost of Capital. Mature firms: 7-10%. Riskier: 10-15%.")
    else:
        st.slider("Risk-free Rate Rf (%)", min_value=0.0, max_value=10.0, step=0.05,
                  key="rf", format="%.2f%%",
                  help="10-year US Treasury yield. As of 2025, ~4.0-4.5%.")
        st.slider("Equity Risk Premium ERP (%)", min_value=2.0, max_value=12.0, step=0.1,
                  key="erp", format="%.2f%%",
                  help="Excess return required over the risk-free rate for holding stocks. US historical ~5-6%.")
        st.slider("Beta (β)", min_value=0.0, max_value=3.0, step=0.05, key="beta",
                  help="Stock's sensitivity to market moves. 1.0 = market average. Tech > 1, utilities < 1.")
        st.slider("Pre-tax Cost of Debt Rd (%)", min_value=0.0, max_value=15.0, step=0.1,
                  key="rd_pretax", format="%.2f%%",
                  help="Yield on the company's debt. Investment-grade typically 4-6%, high-yield 6-10%.")
        st.slider("Target Debt Weight (D/V) (%)", min_value=0.0, max_value=80.0, step=1.0,
                  key="weight_debt", format="%.0f%%",
                  help="Debt as % of total capital. Equity weight = 100% − Debt weight.")

    st.markdown("---")

    # ─── Advanced settings ───────────────────────────────────────────────────
    with st.expander("🔧 Advanced settings"):
        st.checkbox("Mid-year discounting convention", key="mid_year",
                    help="Standard practice in banking: assumes cash flows arrive mid-year, "
                         "discounted with exponent (t − 0.5) instead of t. Yields ~half-year-of-WACC higher valuation.")
        st.slider("Verdict threshold — 'Fairly Valued' band (±%)", min_value=2.0, max_value=25.0,
                  step=1.0, key="fair_band", format="%.0f%%",
                  help="If intrinsic value is within ±this band of market price, verdict says 'Fairly Valued'. "
                       "Outside the band: 'Undervalued' or 'Overvalued'.")

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
term_g        = float(st.session_state.terminal_growth) / 100
years         = int(st.session_state.years)
debt          = float(st.session_state.debt)
cash          = float(st.session_state.cash)
shares        = max(float(st.session_state.shares), 0.1)  # never zero (divide guard)
price         = max(float(st.session_state.current_price), 0.01)

# Resolve WACC (direct entry OR CAPM build-up) BEFORE validation
cost_equity = after_tax_rd = None
if st.session_state.wacc_mode == "CAPM build-up":
    wacc, cost_equity, after_tax_rd = None, None, None  # filled below
else:
    wacc = float(st.session_state.wacc) / 100


# =============================================================================
# DCF Calculation — accepts per-year growth/margin paths for multi-stage models
# =============================================================================
def build_growth_path(simple_g, n, mode, hg_rate, hg_years, fade_years, term_g):
    """Generate a per-year growth-rate list.
    Modes:
      "Constant": flat g for all forecast years.
      "3-stage": high-growth period at hg_rate, then linear fade over fade_years
                 down to term_g, then term_g for any remaining years.
    """
    if mode != "3-stage (high → fade → terminal)":
        return [simple_g] * n
    path = []
    hg_yrs    = max(0, int(hg_years))
    fade_yrs  = max(1, int(fade_years))
    for t in range(1, n + 1):
        if t <= hg_yrs:
            path.append(hg_rate)
        elif t <= hg_yrs + fade_yrs:
            # Linear interpolation from hg_rate (start of fade) to term_g (end)
            step = (t - hg_yrs) / fade_yrs
            path.append(hg_rate + (term_g - hg_rate) * step)
        else:
            path.append(term_g)
    return path


def build_margin_path(simple_m, n, mode, terminal_m):
    """Generate a per-year EBIT margin list.
    Modes:
      "Constant": flat margin for all years.
      "Linear expansion": linear interpolation from simple_m (Year 1) to terminal_m (Year N).
    """
    if mode != "Linear expansion" or n < 2:
        return [simple_m] * n
    return [simple_m + (terminal_m - simple_m) * (t - 1) / (n - 1) for t in range(1, n + 1)]


def compute_capm_wacc(rf, beta, erp, rd_pretax, w_debt, tx):
    """Compute WACC from CAPM components.
    cost_equity   = rf + beta × ERP
    after_tax_rd  = rd_pretax × (1 − tax)
    WACC          = w_equity × cost_equity + w_debt × after_tax_rd
    Returns (wacc, cost_equity, after_tax_rd) — all decimals.
    """
    cost_equity  = rf + beta * erp
    after_tax_rd = rd_pretax * (1 - tx)
    w_equity     = 1 - w_debt
    wacc         = w_equity * cost_equity + w_debt * after_tax_rd
    return wacc, cost_equity, after_tax_rd


def run_dcf(rev0, growth_path, margin_path, tx, ri, w, tg, n,
            *, mid_year=False, tv_method="Gordon Growth", exit_multiple=10.0,
            nwc_pct=0.0, prev_revenue=None):
    """Run a DCF projection with per-year growth and margin paths.

    growth_path[t] applies to Year t+1 (compounded onto prior year's revenue).
    margin_path[t] is the EBIT margin for Year t+1.
    nwc_pct is change in net working capital as % of revenue change (deducted from FCF).
    mid_year=True uses (t-0.5) exponent for discounting (assumes cash arrives mid-year).
    tv_method='Exit Multiple' computes terminal value as exit_multiple × EBITDA_N
        (we approximate EBITDA ≈ EBIT × 1.15 since D&A isn't separately modelled here).
    """
    revenues, ebits, nopats, reinvs, nwcs, fcfs, dfs, pvs = [], [], [], [], [], [], [], []
    rev = rev0
    prev_rev = prev_revenue if prev_revenue is not None else rev0
    for t in range(n):
        g_t = growth_path[t]
        m_t = margin_path[t]
        new_rev = rev * (1 + g_t)
        d_rev   = new_rev - prev_rev
        rev     = new_rev
        ebit    = rev * m_t
        nopat   = ebit * (1 - tx)
        reinv   = nopat * ri
        nwc     = max(0, d_rev * nwc_pct)  # only deduct if revenue grew
        fcf     = nopat - reinv - nwc
        # Discounting: end-year (t) or mid-year (t - 0.5)
        exp = (t + 0.5) if mid_year else (t + 1)
        df = 1 / (1 + w) ** exp
        pv = fcf * df
        revenues.append(rev); ebits.append(ebit); nopats.append(nopat)
        reinvs.append(reinv); nwcs.append(nwc); fcfs.append(fcf); dfs.append(df); pvs.append(pv)
        prev_rev = rev

    sum_pv_fcf = sum(pvs)
    last_fcf  = fcfs[-1] if fcfs else 0
    last_ebit = ebits[-1] if ebits else 0
    if tv_method == "Exit Multiple":
        # EBITDA ≈ EBIT × 1.15 as a rough proxy (no D&A line modelled here)
        terminal_value = exit_multiple * (last_ebit * 1.15)
        terminal_fcf   = last_fcf * (1 + tg)  # only used for display
    else:
        terminal_fcf   = last_fcf * (1 + tg)
        terminal_value = terminal_fcf / (w - tg) if w > tg else float("nan")
    # Discount terminal value back: end-year uses (1+w)^n, mid-year uses (1+w)^(n-0.5)
    tv_exp = (n - 0.5) if mid_year else n
    pv_terminal = terminal_value / (1 + w) ** tv_exp if n > 0 else 0
    return {
        "revenues": revenues, "ebits": ebits, "nopats": nopats, "reinvs": reinvs,
        "nwcs": nwcs, "fcfs": fcfs, "dfs": dfs, "pvs": pvs,
        "growth_path": growth_path, "margin_path": margin_path,
        "sum_pv_fcf": sum_pv_fcf,
        "terminal_fcf": terminal_fcf,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": sum_pv_fcf + pv_terminal,
        "tv_method": tv_method,
    }


# Build growth + margin paths from the user's choice of mode
growth_path = build_growth_path(
    growth, years, st.session_state.growth_mode,
    st.session_state.high_growth / 100,
    st.session_state.high_years,
    st.session_state.fade_years,
    term_g,
)
margin_path = build_margin_path(
    margin, years, st.session_state.margin_mode,
    st.session_state.margin_terminal / 100,
)

# Compute WACC — either from CAPM components or use direct entry
if st.session_state.wacc_mode == "CAPM build-up":
    wacc, cost_equity, after_tax_rd = compute_capm_wacc(
        st.session_state.rf / 100,
        st.session_state.beta,
        st.session_state.erp / 100,
        st.session_state.rd_pretax / 100,
        st.session_state.weight_debt / 100,
        tax_rate,
    )

# Now validate (WACC has been resolved either way)
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

dcf = run_dcf(
    revenue, growth_path, margin_path, tax_rate, reinvest_rate, wacc, term_g, years,
    mid_year=st.session_state.mid_year,
    tv_method=st.session_state.tv_method,
    exit_multiple=st.session_state.exit_multiple,
    nwc_pct=st.session_state.nwc_pct / 100,
)

enterprise_value = dcf["enterprise_value"]
equity_value     = enterprise_value - debt + cash
intrinsic_per    = equity_value / shares
upside           = (intrinsic_per - price) / price if price > 0 else 0
mos              = (intrinsic_per - price) / intrinsic_per if intrinsic_per > 0 else 0


# =============================================================================
# Page header — company + headline metrics
# =============================================================================
# If the loaded company name doesn't match the current ticker (user typed a new
# ticker but autofill hasn't refreshed yet), show ticker only — never display a
# stale company name that doesn't match the ticker.
_loaded = (st.session_state.get("loaded_ticker") or "").strip().upper()
_cur_t = (ticker or "").strip().upper()
if _loaded == _cur_t and company_name:
    st.markdown(f"## 🏢 {company_name} ({ticker})")
else:
    st.markdown(f"## 🏢 {ticker}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("DCF Intrinsic Value", f"${intrinsic_per:,.2f}")
m2.metric("Current Market Price", f"${price:,.2f}")
m3.metric("Upside / (Downside)", f"{upside:+.1%}",
          delta=f"${intrinsic_per - price:,.2f}",
          delta_color="normal" if abs(upside) < 0.05 else ("inverse" if upside < 0 else "normal"))
m4.metric("Margin of Safety", f"{mos:+.1%}")

# Verdict box (threshold customisable in Advanced settings)
fair_band = st.session_state.fair_band / 100
if abs(upside) < fair_band:
    st.info(f"⚖️  **Fairly Valued** — DCF intrinsic value is within ±{fair_band*100:.0f}% of market price.")
elif upside > 0:
    st.success(f"📈 **Potentially Undervalued** — DCF estimates the stock is worth {upside:.1%} more than its current price.")
else:
    st.warning(f"📉 **Potentially Overvalued** — Market is paying {-upside:.1%} above the DCF estimate.")

# Hint when DCF is meaningfully below market — common for premium-priced stocks
if upside < -0.30:
    st.caption(
        "💡 *DCF often produces lower values than market price for high-multiple stocks. "
        "The market may be pricing in faster growth, margin expansion, or longer-than-modeled growth. "
        "Try increasing **growth rate**, increasing **EBIT margin**, lowering **WACC**, or extending the **forecast horizon** "
        "to model a more bullish view. Use the sensitivity tables to see what assumptions justify the market price.*"
    )


# =============================================================================
# CAPM WACC build-up panel (only when CAPM mode is active)
# =============================================================================
if st.session_state.wacc_mode == "CAPM build-up":
    with st.expander(f"💸 WACC build-up (CAPM) → **{wacc*100:.2f}%**", expanded=False):
        st.latex(r"\text{Cost of Equity} = R_f + \beta \times \text{ERP}")
        st.latex(r"\text{WACC} = \frac{E}{V} \times \text{Cost of Equity} + \frac{D}{V} \times R_d \times (1 - \text{tax})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk-free Rate (Rf)", f"{st.session_state.rf:.2f}%")
        c1.metric("Beta (β)", f"{st.session_state.beta:.2f}")
        c2.metric("Equity Risk Premium", f"{st.session_state.erp:.2f}%")
        c2.metric("Cost of Equity (Re)", f"{cost_equity*100:.2f}%")
        c3.metric("Pre-tax Cost of Debt", f"{st.session_state.rd_pretax:.2f}%")
        c3.metric("After-tax Cost of Debt", f"{after_tax_rd*100:.2f}%")
        st.markdown(
            f"**Capital structure:** {(100-st.session_state.weight_debt):.0f}% Equity / "
            f"{st.session_state.weight_debt:.0f}% Debt  →  "
            f"**WACC = {wacc*100:.2f}%**"
        )


# =============================================================================
# Multi-stage growth / margin path display (only when non-constant)
# =============================================================================
if st.session_state.growth_mode != "Constant" or st.session_state.margin_mode != "Constant":
    with st.expander("📐 Growth & Margin Path (per year)", expanded=False):
        path_df = pd.DataFrame({
            "Year":             [f"Year {t+1}" for t in range(years)],
            "Growth Rate":      [f"{g*100:.2f}%" for g in growth_path],
            "EBIT Margin":      [f"{m*100:.2f}%" for m in margin_path],
        })
        st.dataframe(path_df, hide_index=True, use_container_width=True)
        if st.session_state.growth_mode != "Constant":
            st.caption(
                f"3-stage growth: {st.session_state.high_years} years at "
                f"{st.session_state.high_growth:.1f}%, then {st.session_state.fade_years}-year "
                f"linear fade to {st.session_state.terminal_growth:.2f}% terminal."
            )
        if st.session_state.margin_mode != "Constant":
            st.caption(
                f"Linear margin expansion: {st.session_state.margin:.1f}% → "
                f"{st.session_state.margin_terminal:.1f}% over {years} years."
            )


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
    "Year":           list(range(1, years + 1)),
    "Growth":         [f"{g*100:.2f}%" for g in dcf["growth_path"]],
    "Margin":         [f"{m*100:.2f}%" for m in dcf["margin_path"]],
    "Revenue ($M)":   [f"{x:,.0f}" for x in dcf["revenues"]],
    "EBIT ($M)":      [f"{x:,.0f}" for x in dcf["ebits"]],
    "NOPAT ($M)":     [f"{x:,.0f}" for x in dcf["nopats"]],
    "Reinvest ($M)":  [f"{x:,.0f}" for x in dcf["reinvs"]],
    "Δ NWC ($M)":     [f"{x:,.0f}" for x in dcf["nwcs"]],
    "FCF ($M)":       [f"{x:,.0f}" for x in dcf["fcfs"]],
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
           "Color scale: 🔴 lowest value → 🟡 mid → 🟢 highest value within each table.")

# Smooth red→yellow→green colormap that works on both light and dark themes.
# Uses HSL: red (0°) → yellow (60°) → green (120°), with low saturation to keep
# text readable. Returns CSS background strings for pandas Styler.applymap.
def _gradient_style(val, vmin, vmax):
    if pd.isna(val) or vmax == vmin:
        return ""
    ratio = max(0, min(1, (val - vmin) / (vmax - vmin)))
    hue = int(120 * ratio)              # 0 = red, 60 = yellow, 120 = green
    return f"background-color: hsl({hue}, 65%, 78%); color: #111;"

def style_table_gradient(df):
    """Apply a per-table red→green gradient based on numeric cell values.
    Min value in the table → red, max → green, linear interpolation between.
    """
    flat = df.stack().dropna()
    if flat.empty:
        return df.style.format("${:,.2f}")
    vmin, vmax = float(flat.min()), float(flat.max())
    return (df.style
            .format("${:,.2f}")
            .map(lambda v: _gradient_style(v, vmin, vmax) if pd.notna(v) else ""))

# Table 1: WACC × Terminal Growth
# Run DCF with given overrides; reuses the live growth/margin paths and all advanced settings
def _sens_dcf(rev=None, g=None, m=None, w=None, tg=None):
    rev_use = revenue if rev is None else rev
    g_use   = growth if g is None else g
    m_use   = margin if m is None else m
    w_use   = wacc if w is None else w
    tg_use  = term_g if tg is None else tg
    # Rebuild paths with the overridden simple growth/margin
    gp = build_growth_path(g_use, years, st.session_state.growth_mode,
                           st.session_state.high_growth/100,
                           st.session_state.high_years, st.session_state.fade_years, tg_use)
    mp = build_margin_path(m_use, years, st.session_state.margin_mode,
                           st.session_state.margin_terminal/100)
    return run_dcf(rev_use, gp, mp, tax_rate, reinvest_rate, w_use, tg_use, years,
                   mid_year=st.session_state.mid_year,
                   tv_method=st.session_state.tv_method,
                   exit_multiple=st.session_state.exit_multiple,
                   nwc_pct=st.session_state.nwc_pct/100)

st.markdown("### Table 1 — Intrinsic Value: WACC vs. Terminal Growth")
wacc_range = [round(wacc + dx, 4) for dx in [-0.02, -0.01, 0, 0.01, 0.02]]
tg_range   = [round(term_g + dx, 4) for dx in [-0.01, -0.005, 0, 0.005, 0.01]]
tbl1 = []
for w in wacc_range:
    if w <= 0: continue
    row = []
    for tg in tg_range:
        if tg < 0 or w <= tg:
            row.append(np.nan); continue
        d = _sens_dcf(w=w, tg=tg)
        eq = d["enterprise_value"] - debt + cash
        row.append(eq / shares)
    tbl1.append(row)
df1 = pd.DataFrame(
    tbl1,
    index=[f"WACC={w*100:.2f}%" for w in wacc_range if w > 0],
    columns=[f"g={tg*100:.2f}%" for tg in tg_range],
)
st.dataframe(style_table_gradient(df1), use_container_width=True)

# Table 2: Growth × Margin
st.markdown("### Table 2 — Intrinsic Value: Growth Rate vs. EBIT Margin")
g_range = [round(growth + dx, 4) for dx in [-0.04, -0.02, 0, 0.02, 0.04]]
m_range = [round(margin + dx, 4) for dx in [-0.05, -0.025, 0, 0.025, 0.05]]
tbl2 = []
for g in g_range:
    row = []
    for m in m_range:
        if m <= 0: row.append(np.nan); continue
        d = _sens_dcf(g=g, m=m)
        eq = d["enterprise_value"] - debt + cash
        row.append(eq / shares)
    tbl2.append(row)
df2 = pd.DataFrame(
    tbl2,
    index=[f"g={g*100:.2f}%" for g in g_range],
    columns=[f"margin={m*100:.2f}%" for m in m_range],
)
st.dataframe(style_table_gradient(df2), use_container_width=True)


st.markdown("---")




# =============================================================================
# Methodology note (always visible at bottom)
# =============================================================================
with st.expander("📖 Methodology"):
    st.markdown("""
**Discounted Cash Flow Model**

Estimates a stock's intrinsic value from expected future cash flows, discounted to present value at the firm's cost of capital (WACC).

**Steps:**
1. **Forecast** revenue, EBIT, NOPAT, and free cash flow for an explicit period (3-15 years).
2. **Discount** each year's FCF to today using `1/(1+WACC)^t`.
3. **Terminal Value** captures everything beyond the forecast: `FCF_N × (1+g) / (WACC − g)`.
4. **Enterprise Value** = Σ PV of FCFs + PV of Terminal Value.
5. **Equity Value** = EV − Debt + Cash.
6. **Per-share** = Equity Value ÷ Diluted Shares Outstanding.

**WACC build-up (CAPM):** WACC = (E/V) × [Rf + β × ERP] + (D/V) × Rd × (1 − tax).

**Multi-stage growth (3-stage):** Years 1 to N₁ at the high-growth rate; Years N₁+1 to N₁+N₂ linearly fade to the terminal rate; Years beyond compound at the terminal rate (also used for the perpetuity).

**Margin expansion:** Optionally model margins improving (or compressing) linearly between Year 1 and Year N. Useful for growth firms with operating leverage.

**Input guidance:**
- **Growth rate** — informed by historical CAGR + forward thesis. Mature firms: a few % above GDP. Growth firms: double-digit, fading down.
- **EBIT margin** — historical operating margin. Better firms expand margins over time.
- **WACC** — riskier firms have higher WACC. Rough guide: large-cap stable 7–9%, mid-cap 9–12%, small/risky 12%+.
- **Terminal growth** — should not exceed long-run GDP (~2-3%) over the very long run.
- **Reinvestment rate** — % of NOPAT going back into CapEx + working capital. Higher reinvestment = less FCF today, more future growth.

**Data source:** SEC EDGAR Company Facts API (`data.sec.gov`) — official filings, used by professional analysts.
    """)
