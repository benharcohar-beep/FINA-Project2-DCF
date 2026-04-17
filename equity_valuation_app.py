# =============================================================================
# FINA 4011/5011 — Project 2: DCF Equity Valuation App
# =============================================================================
# Requirements (install via pip or conda):
#   pip install streamlit yfinance pandas numpy plotly openpyxl
#
# Run:
#   streamlit run equity_valuation_app.py
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings("ignore")

# curl_cffi impersonates a real browser's TLS fingerprint, which reliably
# bypasses Yahoo Finance's rate-limiting of Streamlit Cloud's shared IPs.
try:
    from curl_cffi import requests as curl_requests
    _YF_SESSION = curl_requests.Session(impersonate="chrome")
except Exception:
    _YF_SESSION = None

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Equity Valuation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 0.85rem; color: #555; }
    .formula-box {
        background: #eaf3fb;
        border-left: 4px solid #1f4e79;
        padding: 10px 14px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 8px 0;
    }
    .info-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin: 8px 0;
    }
    .section-divider { margin: 2rem 0 1rem 0; }
    h2 { color: #1f4e79; }
    h3 { color: #2e75b6; }
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("📈 DCF Equity Valuation App")
st.caption("FINA 4011/5011 · Discounted Cash Flow Model · Data via Yahoo Finance")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Stock Selection")
    ticker_input = st.text_input(
        "Enter Ticker Symbol",
        value="AAPL",
        max_chars=10,
        help="E.g. AAPL, MSFT, GOOGL, TSLA",
    ).upper().strip()

    st.markdown("---")
    st.header("⚙️ Model Assumptions")

    with st.expander("📊 Revenue Growth Rates", expanded=True):
        st.markdown(
            "<div class='info-box'>Growth assumptions are the biggest driver of value. "
            "Compare to historical growth and analyst forecasts shown in the app.</div>",
            unsafe_allow_html=True,
        )
        rev_growth_1_5 = (
            st.slider("Near-Term Growth – Yr 1–5 (%)", -20.0, 60.0, 12.0, 0.5,
                      help="Expected annual revenue growth for the first 5 years.")
            / 100
        )
        rev_growth_6_10 = (
            st.slider("Mid-Term Growth – Yr 6–10 (%)", -10.0, 40.0, 7.0, 0.5,
                      help="Growth typically slows as a company matures.")
            / 100
        )
        terminal_growth = (
            st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 2.5, 0.1,
                      help="Perpetual growth rate beyond the forecast. Usually near GDP (2–3%).")
            / 100
        )

    with st.expander("💼 Profitability & Reinvestment", expanded=True):
        st.markdown(
            "<div class='info-box'>Margins determine how much of each revenue dollar "
            "converts to cash. Review historical margins for benchmarking.</div>",
            unsafe_allow_html=True,
        )
        ebit_margin = (
            st.slider("EBIT Margin (%)", -10.0, 60.0, 25.0, 0.5,
                      help="Operating income ÷ Revenue. Check historical and peer comparisons.")
            / 100
        )
        tax_rate = (
            st.slider("Effective Tax Rate (%)", 0.0, 40.0, 21.0, 0.5,
                      help="Effective tax rate on operating income. US federal rate is 21%.")
            / 100
        )
        capex_pct = (
            st.slider("CapEx / Revenue (%)", 0.0, 30.0, 5.0, 0.5,
                      help="Capital expenditures as a share of revenue. Higher for asset-heavy firms.")
            / 100
        )
        nwc_pct = (
            st.slider("ΔNWC / Revenue (%)", -5.0, 10.0, 1.5, 0.5,
                      help="Incremental net working capital needed per unit of revenue growth.")
            / 100
        )

    with st.expander("📉 Discount Rate (WACC)", expanded=True):
        st.markdown(
            "<div class='info-box'>WACC is the minimum return required by all capital providers. "
            "A higher WACC means future cash flows are worth less today.</div>",
            unsafe_allow_html=True,
        )
        wacc_mode = st.radio("WACC Input Mode", ["Set WACC Directly", "Build from Components"])

        if wacc_mode == "Set WACC Directly":
            wacc = (
                st.slider("WACC (%)", 4.0, 25.0, 10.0, 0.25,
                          help="Typical ranges: 6–8% (low risk), 8–12% (moderate), 12%+ (high risk).")
                / 100
            )
        else:
            ke = st.slider("Cost of Equity – Ke (%)", 5.0, 25.0, 12.0, 0.25) / 100
            kd = st.slider("Pre-Tax Cost of Debt – Kd (%)", 2.0, 12.0, 4.0, 0.25) / 100
            we = st.slider("Equity Weight (%)", 20.0, 100.0, 80.0, 1.0) / 100
            wd = 1.0 - we
            wacc = we * ke + wd * kd * (1 - tax_rate)
            st.success(f"Calculated WACC: **{wacc*100:.2f}%**")

    projection_years = st.selectbox(
        "Forecast Period (Years)", [5, 7, 10], index=2,
        help="Length of explicit forecast. Terminal value captures everything beyond."
    )

# ─── Data Fetching ─────────────────────────────────────────────────────────────
# Note: @st.cache_data requires return values to be picklable. The yfinance
# Ticker object (especially with a curl_cffi session attached) is NOT picklable,
# so we only return the serializable data (dict + DataFrames), not the Ticker.
@st.cache_data(ttl=600, show_spinner="Fetching data from Yahoo Finance...")
def fetch_stock_data(sym, max_retries: int = 4):
    """Fetch Yahoo Finance data with browser-impersonation session + retry/backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            if _YF_SESSION is not None:
                stk = yf.Ticker(sym, session=_YF_SESSION)
            else:
                stk = yf.Ticker(sym)

            info = stk.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not info or price is None:
                last_err = "Empty response from Yahoo Finance (likely rate-limited)."
                time.sleep(2.0 * (attempt + 1))
                continue

            return {
                "info": dict(info),
                "financials": stk.financials,
                "cashflow": stk.cashflow,
                "balance_sheet": stk.balance_sheet,
                "price_hist": stk.history(period="1y"),
                "error": None,
            }
        except Exception as exc:
            last_err = str(exc)
            time.sleep(2.0 * (attempt + 1))

    # Raise so the error is NOT cached — each call retries fresh
    raise RuntimeError(last_err or "Unknown error fetching data.")

try:
    _data = fetch_stock_data(ticker_input)
    info            = _data["info"]
    financials      = _data["financials"]
    cashflow        = _data["cashflow"]
    balance_sheet   = _data["balance_sheet"]
    price_hist      = _data["price_hist"]
    stock_obj       = None  # no longer needed downstream
    fetch_error     = None
except Exception as _e:
    info, financials, cashflow, balance_sheet, price_hist, stock_obj = (None,) * 6
    fetch_error = str(_e)

if fetch_error or info is None:
    st.error(
        f"❌ Could not load live data for **'{ticker_input}'**.\n\n"
        f"**Reason:** {fetch_error or 'Unknown error.'}\n\n"
        "This is usually a temporary Yahoo Finance rate-limit on Streamlit Cloud's shared IPs. "
        "Try again in ~30 seconds, switch tickers (e.g. MSFT, GOOGL, JNJ), or run the app locally."
    )

    with st.expander("🛠 Manual-input mode — run the DCF without Yahoo Finance"):
        st.markdown(
            "Enter the inputs below to run the DCF model manually. "
            "You can pull these from any 10-K / analyst report."
        )
        c1, c2 = st.columns(2)
        with c1:
            m_price = st.number_input("Current Share Price ($)", min_value=0.0, value=150.0, step=1.0, key="m_price")
            m_rev   = st.number_input("Latest Annual Revenue ($M)", min_value=0.0, value=100000.0, step=1000.0, key="m_rev")
            m_shares = st.number_input("Shares Outstanding (millions)", min_value=0.0, value=1000.0, step=10.0, key="m_shares")
        with c2:
            m_debt  = st.number_input("Total Debt ($M)", min_value=0.0, value=5000.0, step=100.0, key="m_debt")
            m_cash  = st.number_input("Cash & Equivalents ($M)", min_value=0.0, value=10000.0, step=100.0, key="m_cash")
            m_da    = st.number_input("D&A ($M)", min_value=0.0, value=3000.0, step=100.0, key="m_da")

        if st.button("▶ Run DCF with manual inputs", type="primary"):
            # Stub an `info` dict so the rest of the script can proceed
            info = {
                "currentPrice": m_price,
                "regularMarketPrice": m_price,
                "totalRevenue": m_rev * 1e6,
                "sharesOutstanding": m_shares * 1e6,
                "totalDebt": m_debt * 1e6,
                "totalCash": m_cash * 1e6,
                "longName": ticker_input,
                "sector": "Manual Input",
                "industry": "Manual Input",
                "beta": 1.0,
            }
            financials, cashflow, balance_sheet, price_hist, stock_obj = None, None, None, None, None
            st.session_state["_manual_da"] = m_da * 1e6
        else:
            st.stop()
    if info is None:
        st.stop()

# ─── Helper ───────────────────────────────────────────────────────────────────
def sg(key, default=0):
    v = info.get(key, default)
    return v if (v is not None and v != "") else default

# ─── Extract Key Financials ───────────────────────────────────────────────────
current_price   = sg("currentPrice") or sg("regularMarketPrice")
market_cap      = sg("marketCap")
shares_out      = sg("sharesOutstanding")
total_debt      = sg("totalDebt")
cash_eq         = sg("totalCash")
beta            = sg("beta", 1.0)
company_name    = sg("longName", ticker_input)
sector          = sg("sector", "N/A")
industry        = sg("industry", "N/A")
pe_ttm          = sg("trailingPE")
forward_pe      = sg("forwardPE")
pb_ratio        = sg("priceToBook")
analyst_target  = sg("targetMeanPrice")
revenue_ttm     = sg("totalRevenue")
description     = sg("longBusinessSummary", "No description available.")

# Base revenue (most recent annual)
try:
    if financials is not None and not financials.empty and "Total Revenue" in financials.index:
        base_revenue = float(financials.loc["Total Revenue"].iloc[0])
    else:
        base_revenue = float(revenue_ttm) if revenue_ttm else 1e9
except Exception:
    base_revenue = float(revenue_ttm) if revenue_ttm else 1e9

# D&A (used in FCF build-up)
da_value = 0.0
try:
    if cashflow is not None and not cashflow.empty:
        for k in ["Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion"]:
            if k in cashflow.index:
                da_value = float(cashflow.loc[k].iloc[0])
                break
except Exception:
    pass
if da_value == 0:
    # Use manual-input D&A if user supplied it, otherwise fallback to 3% of revenue
    da_value = st.session_state.get("_manual_da") or (base_revenue * 0.03)

# ─── Section 1: Company Snapshot ──────────────────────────────────────────────
st.header(f"🏢 {company_name}  ({ticker_input})")
st.caption(f"**Sector:** {sector}  ·  **Industry:** {industry}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", f"${current_price:,.2f}" if current_price else "N/A")
c2.metric("Market Cap", f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
c3.metric("P/E (TTM)", f"{pe_ttm:.1f}x" if pe_ttm else "N/A")
c4.metric("Fwd P/E", f"{forward_pe:.1f}x" if forward_pe else "N/A")
c5.metric("P/B", f"{pb_ratio:.1f}x" if pb_ratio else "N/A")
c6.metric("Beta", f"{beta:.2f}" if beta else "N/A")

if analyst_target:
    analyst_upside = (analyst_target - current_price) / current_price * 100 if current_price else 0
    st.info(
        f"📌 **Analyst Consensus Target:** ${analyst_target:,.2f}  "
        f"({'▲' if analyst_upside >= 0 else '▼'} {abs(analyst_upside):.1f}% from current price)"
    )

with st.expander("📋 Business Description"):
    st.write(description)

# 1-year price chart
if price_hist is not None and not price_hist.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_hist.index, y=price_hist["Close"],
        mode="lines", line=dict(color="#1f4e79", width=2), name="Close",
        fill="tozeroy", fillcolor="rgba(31,78,121,0.08)",
    ))
    fig_price.update_layout(
        title=f"{ticker_input} — 1-Year Price History",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=280, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#222", size=12),
        title_font=dict(color="#1f4e79", size=15),
        xaxis=dict(gridcolor="#e5e5e5", linecolor="#999", tickfont=dict(color="#222")),
        yaxis=dict(gridcolor="#e5e5e5", linecolor="#999", tickfont=dict(color="#222")),
        margin=dict(t=40, b=30, l=0, r=0),
    )
    st.plotly_chart(fig_price, use_container_width=True)

st.markdown("---")

# ─── Section 2: Historical Financials ─────────────────────────────────────────
st.header("📊 Historical Financials")

with st.expander("💡 Why review historical data before setting assumptions?", expanded=False):
    st.markdown("""
    Historical performance is the **empirical anchor** for your assumptions.

    - **Revenue growth** — compare your near-term assumption to the actual compounded growth
    - **EBIT margin** — mean-reversion is common; wild departures from history need justification
    - **CapEx intensity** — identifies whether the business is capital-light or capital-heavy

    > Rule of thumb: assumptions that diverge significantly from history require explicit,
    > documented reasons (new product, industry shift, margin expansion program, etc.)
    """)

hist_rows = {}
try:
    if financials is not None and not financials.empty:
        rev_row    = financials.loc["Total Revenue"]          if "Total Revenue"    in financials.index else None
        ebit_row   = (financials.loc["EBIT"]                  if "EBIT"             in financials.index
                      else financials.loc["Operating Income"] if "Operating Income" in financials.index else None)

        if rev_row is not None:
            cols = [str(c)[:7] for c in rev_row.index]
            hist_rows["Revenue ($B)"] = {c: v / 1e9 for c, v in zip(cols, rev_row.values)}
        if ebit_row is not None:
            cols = [str(c)[:7] for c in ebit_row.index]
            hist_rows["EBIT ($B)"] = {c: v / 1e9 for c, v in zip(cols, ebit_row.values)}
            if rev_row is not None:
                hist_rows["EBIT Margin"] = {
                    c: (e / r) if r != 0 else 0
                    for c, e, r in zip(cols, ebit_row.values, rev_row.values)
                }

        if hist_rows:
            hist_df = pd.DataFrame(hist_rows).T
            st.dataframe(
                hist_df.style.format(
                    lambda v: f"{v*100:.1f}%" if v < 2 and v > -0.5 else f"${v:,.2f}B"
                ),
                use_container_width=True,
            )

        # Show historical avg growth vs user assumption
        if rev_row is not None and len(rev_row) >= 2:
            rev_sorted = rev_row.sort_index()
            cagr = (rev_sorted.iloc[-1] / rev_sorted.iloc[0]) ** (1 / (len(rev_sorted) - 1)) - 1
            color = "normal" if abs(rev_growth_1_5 - cagr) < 0.05 else "off"
            st.info(
                f"📈 **Historical Revenue CAGR ({len(rev_sorted)-1}yr):** {cagr*100:.1f}%  ·  "
                f"**Your Near-Term Assumption:** {rev_growth_1_5*100:.1f}%  "
                + ("✅ Consistent" if abs(rev_growth_1_5 - cagr) < 0.05 else "⚠️ Diverges from history — ensure you have a thesis")
            )
except Exception as e:
    st.warning(f"Could not parse historical financials: {e}")

st.markdown("---")

# ─── Section 3: DCF Walkthrough ────────────────────────────────────────────────
st.header("🔢 DCF Model — Step-by-Step")

with st.expander("📖 How a DCF Model Works", expanded=False):
    st.markdown("""
    A **Discounted Cash Flow (DCF)** model estimates what a company is worth today
    based on the cash it is expected to generate in the future.

    The logic has three steps:

    **Step 1 — Forecast Free Cash Flows (FCF)**
    > FCF = EBIT × (1 − Tax Rate) + D&A − CapEx − ΔNWC

    Free Cash Flow is the cash left over after the company has paid taxes, maintained
    its assets (CapEx), and funded growth (ΔNWC). It belongs to **all** capital providers.

    **Step 2 — Estimate Terminal Value**
    > TV = FCF_last × (1 + g) / (WACC − g)

    The Terminal Value captures all cash flows beyond the forecast window, using the
    perpetuity growth model (Gordon Growth Model).

    **Step 3 — Discount Back to Today**
    > PV = CF_t / (1 + WACC)^t

    Each future cash flow and the terminal value are discounted to present value. Sum
    everything, subtract net debt, divide by shares outstanding → intrinsic value per share.
    """)

# ── Build projections ─────────────────────────────────────────────────────────
rows = []
rev = base_revenue
for yr in range(1, projection_years + 1):
    g = rev_growth_1_5 if yr <= 5 else rev_growth_6_10
    rev = rev * (1 + g)
    ebit    = rev * ebit_margin
    nopat   = ebit * (1 - tax_rate)
    capex   = rev * capex_pct
    d_nwc   = rev * nwc_pct
    fcf     = nopat + da_value - capex - d_nwc
    df      = 1 / (1 + wacc) ** yr
    pv_fcf  = fcf * df
    rows.append({
        "Year":              f"Year {yr}",
        "Revenue ($M)":      rev / 1e6,
        "Rev Growth":        g,
        "EBIT ($M)":         ebit / 1e6,
        "EBIT Margin":       ebit_margin,
        "NOPAT ($M)":        nopat / 1e6,
        "D&A ($M)":          da_value / 1e6,
        "CapEx ($M)":        capex / 1e6,
        "ΔNWC ($M)":         d_nwc / 1e6,
        "FCF ($M)":          fcf / 1e6,
        "Discount Factor":   df,
        "PV of FCF ($M)":    pv_fcf / 1e6,
    })

proj_df = pd.DataFrame(rows).set_index("Year")

# ── Step 1 table ──────────────────────────────────────────────────────────────
st.subheader("Step 1 — Project Free Cash Flows")

display_cols = ["Revenue ($M)", "Rev Growth", "EBIT ($M)", "EBIT Margin",
                "NOPAT ($M)", "D&A ($M)", "CapEx ($M)", "ΔNWC ($M)",
                "FCF ($M)", "Discount Factor", "PV of FCF ($M)"]

fmt = {
    "Revenue ($M)":    "${:,.1f}",
    "Rev Growth":      "{:.1%}",
    "EBIT ($M)":       "${:,.1f}",
    "EBIT Margin":     "{:.1%}",
    "NOPAT ($M)":      "${:,.1f}",
    "D&A ($M)":        "${:,.1f}",
    "CapEx ($M)":      "${:,.1f}",
    "ΔNWC ($M)":       "${:,.1f}",
    "FCF ($M)":        "${:,.1f}",
    "Discount Factor": "{:.4f}",
    "PV of FCF ($M)":  "${:,.1f}",
}

st.dataframe(proj_df[display_cols].style.format(fmt), use_container_width=True)

with st.expander("🔍 FCF Formula Breakdown"):
    st.markdown(
        f"""
<div class='formula-box'>
FCF = NOPAT + D&amp;A − CapEx − ΔNWC<br>
NOPAT = EBIT × (1 − Tax Rate) = Revenue × EBIT Margin × (1 − {tax_rate:.0%})<br>
D&amp;A = ${da_value/1e6:,.1f}M (from most recent cash flow statement)<br>
CapEx = {capex_pct:.1%} × Revenue<br>
ΔNWC = {nwc_pct:.1%} × Revenue
</div>
        """,
        unsafe_allow_html=True,
    )

# FCF chart
fig_fcf = go.Figure()
fig_fcf.add_bar(
    x=proj_df.index, y=proj_df["FCF ($M)"],
    name="FCF", marker_color="#1f4e79",
)
fig_fcf.add_bar(
    x=proj_df.index, y=proj_df["PV of FCF ($M)"],
    name="PV of FCF", marker_color="#4472c4",
)
fig_fcf.update_layout(
    title="Projected FCF vs. Present Value of FCF",
    xaxis_title="Year", yaxis_title="$ Millions",
    barmode="group", height=320,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="#222", size=12),
    title_font=dict(color="#1f4e79", size=15),
    xaxis=dict(gridcolor="#e5e5e5", linecolor="#999", tickfont=dict(color="#222")),
    yaxis=dict(gridcolor="#e5e5e5", linecolor="#999", tickfont=dict(color="#222")),
    legend=dict(orientation="h", y=1.1, font=dict(color="#222")),
    margin=dict(t=50, b=30),
)
st.plotly_chart(fig_fcf, use_container_width=True)

# ── Step 2: Terminal Value ────────────────────────────────────────────────────
st.subheader("Step 2 — Terminal Value")

last_fcf_abs    = proj_df["FCF ($M)"].iloc[-1] * 1e6
tv_fcf          = last_fcf_abs * (1 + terminal_growth)
terminal_value  = tv_fcf / (wacc - terminal_growth) if wacc > terminal_growth else float("nan")
pv_terminal     = terminal_value / (1 + wacc) ** projection_years if not np.isnan(terminal_value) else float("nan")

if np.isnan(terminal_value):
    st.error("⚠️ Terminal Growth Rate must be **less than** WACC. Please adjust your assumptions.")
    st.stop()

with st.expander("🔍 Terminal Value Formula"):
    st.markdown(
        f"""
<div class='formula-box'>
TV = FCF_(Year {projection_years}) × (1 + g) / (WACC − g)<br>
   = ${last_fcf_abs/1e6:,.1f}M × (1 + {terminal_growth:.2%}) / ({wacc:.2%} − {terminal_growth:.2%})<br>
   = <b>${terminal_value/1e9:,.2f}B</b><br><br>
PV of TV = TV / (1 + WACC)^{projection_years}<br>
         = ${terminal_value/1e9:,.2f}B / (1 + {wacc:.2%})^{projection_years}<br>
         = <b>${pv_terminal/1e9:,.2f}B</b>
</div>
        """,
        unsafe_allow_html=True,
    )

tv_pct = pv_terminal / (proj_df["PV of FCF ($M)"].sum() * 1e6 + pv_terminal) * 100
col_tv1, col_tv2, col_tv3 = st.columns(3)
col_tv1.metric("Terminal FCF (Yr N+1)", f"${tv_fcf/1e6:,.1f}M")
col_tv2.metric("Terminal Value (Undiscounted)", f"${terminal_value/1e9:,.2f}B")
col_tv3.metric("PV of Terminal Value", f"${pv_terminal/1e9:,.2f}B")

st.warning(
    f"📌 Terminal Value accounts for **{tv_pct:.0f}%** of Enterprise Value. "
    "This sensitivity to long-run assumptions is why scenario/sensitivity analysis is critical."
)

# ── Step 3: Enterprise → Equity Value ─────────────────────────────────────────
st.subheader("Step 3 — Enterprise Value to Equity Value per Share")

sum_pv_fcf        = proj_df["PV of FCF ($M)"].sum() * 1e6
enterprise_value  = sum_pv_fcf + pv_terminal
equity_value      = enterprise_value - total_debt + cash_eq
iv_per_share      = equity_value / shares_out if shares_out else 0.0

with st.expander("🔍 EV → Equity Value Bridge"):
    st.markdown(
        f"""
<div class='formula-box'>
Enterprise Value   = Σ PV(FCF) + PV(Terminal Value)<br>
                   = ${sum_pv_fcf/1e9:,.2f}B + ${pv_terminal/1e9:,.2f}B<br>
                   = <b>${enterprise_value/1e9:,.2f}B</b><br><br>
Equity Value       = EV − Total Debt + Cash<br>
                   = ${enterprise_value/1e9:,.2f}B − ${total_debt/1e9:,.2f}B + ${cash_eq/1e9:,.2f}B<br>
                   = <b>${equity_value/1e9:,.2f}B</b><br><br>
Intrinsic Value/Share = Equity Value ÷ Shares Outstanding<br>
                      = ${equity_value/1e9:,.2f}B ÷ {shares_out/1e9:,.2f}B<br>
                      = <b>${iv_per_share:,.2f}</b>
</div>
        """,
        unsafe_allow_html=True,
    )

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
col_e1.metric("Σ PV of FCFs", f"${sum_pv_fcf/1e9:,.2f}B")
col_e2.metric("+ PV of Terminal Value", f"${pv_terminal/1e9:,.2f}B")
col_e3.metric("Equity Value", f"${equity_value/1e9:,.2f}B")
col_e4.metric("Intrinsic Value / Share", f"${iv_per_share:,.2f}")

# EV composition donut
fig_pie = go.Figure(data=[go.Pie(
    labels=["PV of FCFs (Forecast)", "PV of Terminal Value"],
    values=[sum_pv_fcf / 1e9, pv_terminal / 1e9],
    hole=0.45,
    marker_colors=["#1f4e79", "#4472c4"],
    texttemplate="%{label}<br>%{percent:.1%}",
)])
fig_pie.update_layout(
    title="Enterprise Value Composition",
    height=300, showlegend=False,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="#222", size=12),
    title_font=dict(color="#1f4e79", size=15),
    margin=dict(t=40, b=10),
)
fig_pie.update_traces(textfont=dict(color="white", size=13))
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ─── Section 4: Valuation Summary ─────────────────────────────────────────────
st.header("🎯 Valuation Summary")

if current_price and iv_per_share:
    upside   = (iv_per_share - current_price) / current_price * 100
    mos      = (1 - current_price / iv_per_share) * 100 if iv_per_share > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DCF Intrinsic Value", f"${iv_per_share:,.2f}")
    c2.metric("Current Market Price", f"${current_price:,.2f}")
    c3.metric("Upside / (Downside)", f"{upside:+.1f}%")
    c4.metric("Margin of Safety", f"{mos:.1f}%")

    if upside > 20:
        st.success(
            f"✅ **Potentially Undervalued.** The DCF suggests {abs(upside):.1f}% upside "
            f"relative to the current price of ${current_price:,.2f}."
        )
    elif upside < -20:
        st.error(
            f"⚠️ **Potentially Overvalued.** The market price is {abs(upside):.1f}% "
            f"above the DCF intrinsic value of ${iv_per_share:,.2f}."
        )
    else:
        st.warning(
            f"🔄 **Fairly Valued.** The stock trades within ±20% of the DCF estimate."
        )

    # Bar chart — price vs. intrinsic value
    bar_colors = ["#1f4e79", "#2e75b6"]
    vals  = [current_price, iv_per_share]
    names = ["Current Market Price", "DCF Intrinsic Value"]
    if analyst_target:
        vals.append(analyst_target)
        names.append("Analyst Consensus Target")
        bar_colors.append("#ed7d31")

    fig_bar = go.Figure()
    fig_bar.add_bar(
        x=names, y=vals,
        marker_color=bar_colors,
        text=[f"${v:,.2f}" for v in vals],
        textposition="outside",
    )
    fig_bar.update_layout(
        title=f"{ticker_input} — Price vs. Intrinsic Value",
        yaxis_title="Price (USD)", height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#222", size=12),
        title_font=dict(color="#1f4e79", size=15),
        xaxis=dict(linecolor="#999", tickfont=dict(color="#222", size=13)),
        yaxis=dict(gridcolor="#e5e5e5", linecolor="#999", tickfont=dict(color="#222")),
        showlegend=False,
        margin=dict(t=50, b=30),
    )
    fig_bar.update_traces(textfont=dict(color="#222", size=13))
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ─── Section 5: Sensitivity Analysis ──────────────────────────────────────────
st.header("📐 Sensitivity Analysis")

with st.expander("💡 Why Sensitivity Analysis Matters", expanded=False):
    st.markdown("""
    A single DCF output is a **point estimate** built on uncertain assumptions.
    Sensitivity analysis reveals:

    - How much value changes as key assumptions vary
    - Which assumption matters most (usually WACC and terminal growth)
    - The range of plausible intrinsic values — a **valuation range**, not a single number

    **Reading the table:** green cells indicate scenarios where intrinsic value
    exceeds the current price by >20% (margin of safety). Red cells indicate the
    stock appears overvalued. Yellow = fairly valued.
    """)

def calc_iv(wacc_s, tg_s, g_s=None, margin_s=None):
    """Re-compute intrinsic value with alternative assumptions."""
    g_s      = g_s      if g_s is not None else rev_growth_1_5
    margin_s = margin_s if margin_s is not None else ebit_margin
    if wacc_s <= tg_s:
        return np.nan
    rev_s   = base_revenue
    pv_sum  = 0.0
    for yr in range(1, projection_years + 1):
        g_yr = g_s if yr <= 5 else rev_growth_6_10
        rev_s  *= (1 + g_yr)
        fcf_s   = rev_s * margin_s * (1 - tax_rate) + da_value - rev_s * capex_pct - rev_s * nwc_pct
        pv_sum += fcf_s / (1 + wacc_s) ** yr
    last_fcf_s = rev_s * margin_s * (1 - tax_rate) + da_value - rev_s * capex_pct - rev_s * nwc_pct
    tv_s     = last_fcf_s * (1 + tg_s) / (wacc_s - tg_s)
    pv_tv_s  = tv_s / (1 + wacc_s) ** projection_years
    eq_s     = (pv_sum + pv_tv_s) - total_debt + cash_eq
    return eq_s / shares_out if shares_out else 0.0

def color_cell(val):
    if pd.isna(val) or current_price is None or current_price == 0:
        return "background-color: #eeeeee"
    if val > current_price * 1.2:
        return "background-color: #c6efce; color: #276221"
    elif val < current_price * 0.8:
        return "background-color: #ffc7ce; color: #9c0006"
    return "background-color: #ffeb9c; color: #9c5700"

# ── Table 1: WACC vs Terminal Growth ─────────────────────────────────────────
st.subheader("Table 1 — Intrinsic Value: WACC vs. Terminal Growth Rate")

wacc_vals = sorted({round(wacc + d, 4) for d in [-0.02, -0.01, 0, 0.01, 0.02]})
wacc_vals = [max(0.05, w) for w in wacc_vals]
tg_vals   = sorted({round(terminal_growth + d, 4) for d in [-0.01, -0.005, 0, 0.005, 0.01]})
tg_vals   = [max(0.001, min(g, min(wacc_vals) - 0.001)) for g in tg_vals]

tbl1 = pd.DataFrame(
    {f"WACC={w:.2%}": {f"g={g:.2%}": calc_iv(w, g) for g in tg_vals} for w in wacc_vals}
)
tbl1.index.name = "Terminal Growth ↓  /  WACC →"

try:
    styled1 = tbl1.style.applymap(color_cell).format("${:,.2f}", na_rep="N/A")
except AttributeError:
    styled1 = tbl1.style.map(color_cell).format("${:,.2f}", na_rep="N/A")

st.dataframe(styled1, use_container_width=True)

# ── Table 2: Revenue Growth vs EBIT Margin ────────────────────────────────────
st.subheader("Table 2 — Intrinsic Value: Near-Term Growth vs. EBIT Margin")

g_vals = sorted({round(rev_growth_1_5 + d, 4) for d in [-0.05, -0.025, 0, 0.025, 0.05]})
g_vals = [max(-0.15, g) for g in g_vals]
m_vals = sorted({round(ebit_margin + d, 4) for d in [-0.05, -0.025, 0, 0.025, 0.05]})
m_vals = [max(0.0, m) for m in m_vals]

tbl2 = pd.DataFrame(
    {f"Margin={m:.1%}": {f"Growth={g:.1%}": calc_iv(wacc, terminal_growth, g_s=g, margin_s=m) for g in g_vals}
     for m in m_vals}
)
tbl2.index.name = "Revenue Growth ↓  /  EBIT Margin →"

try:
    styled2 = tbl2.style.applymap(color_cell).format("${:,.2f}", na_rep="N/A")
except AttributeError:
    styled2 = tbl2.style.map(color_cell).format("${:,.2f}", na_rep="N/A")

st.dataframe(styled2, use_container_width=True)

st.caption("🟢 Green: >20% upside (margin of safety)  ·  🟡 Yellow: fairly valued (±20%)  ·  🔴 Red: >20% downside")

st.markdown("---")

# ─── Section 6: Assumptions Summary ──────────────────────────────────────────
st.header("📋 Assumptions & Data Sources")

assumptions = pd.DataFrame({
    "Parameter": [
        "Base Revenue", "Near-Term Growth (Yr 1–5)", "Mid-Term Growth (Yr 6–10)",
        "Terminal Growth Rate", "EBIT Margin", "Effective Tax Rate",
        "CapEx / Revenue", "ΔNWC / Revenue", "D&A",
        "WACC", "Forecast Period",
        "Total Debt", "Cash & Equivalents", "Shares Outstanding",
    ],
    "Value": [
        f"${base_revenue/1e9:,.2f}B",
        f"{rev_growth_1_5:.1%}", f"{rev_growth_6_10:.1%}",
        f"{terminal_growth:.2%}", f"{ebit_margin:.1%}", f"{tax_rate:.1%}",
        f"{capex_pct:.1%}", f"{nwc_pct:.1%}", f"${da_value/1e6:,.1f}M",
        f"{wacc:.2%}", f"{projection_years} years",
        f"${total_debt/1e9:,.2f}B", f"${cash_eq/1e9:,.2f}B",
        f"{shares_out/1e9:,.2f}B shares",
    ],
    "Source": [
        "Yahoo Finance (most recent annual)",
        "User Input", "User Input", "User Input",
        "User Input", "User Input", "User Input", "User Input",
        "Yahoo Finance (cash flow statement)",
        "User Input", "User Input",
        "Yahoo Finance", "Yahoo Finance", "Yahoo Finance",
    ],
})

st.dataframe(assumptions, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:#999; font-size:0.82rem; padding:10px 0;'>
    FINA 4011/5011 · Project 2 · DCF Equity Valuation App ·
    Market data sourced from Yahoo Finance via <code>yfinance</code> ·
    For educational purposes only — not investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)
