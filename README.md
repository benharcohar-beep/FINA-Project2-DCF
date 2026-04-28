# DCF Equity Valuation App

**FINA 4011/5011 — Project 2**

A Streamlit application for Discounted Cash Flow (DCF) equity valuation. Users enter (or auto-fill) financial inputs, adjust valuation assumptions, and the app computes intrinsic value per share, compares it to current market price, and runs sensitivity analysis. Includes a one-click Excel export with live formulas for replication.

## Features

- **Manual-input DCF** — every assumption fully editable in the sidebar
- **Optional SEC EDGAR auto-fill** — pulls the latest 10-K fundamentals (revenue, FCF, debt, cash, shares) from the SEC's official API. No API key, no rate limits.
- **Step-by-step walkthrough** — formulas + intermediate values for each of 6 valuation steps
- **Sensitivity analysis** — two tables (WACC × terminal growth, growth × EBIT margin) with color-coded margin-of-safety
- **Excel export** — downloads an `.xlsx` workbook where every projection cell is a live formula referencing the inputs sheet, so you can replicate and modify the valuation in Excel
- **Plotly charts** — FCF projection bars, market vs. intrinsic value comparison

## Run Locally

```bash
pip install -r requirements.txt
streamlit run equity_valuation_app.py
```

## Live App

Deployed on Streamlit Community Cloud:
**https://fina-project2-dcf-valuation.streamlit.app/**

## Data Source

SEC EDGAR Company Facts API (`data.sec.gov`) — official 10-K filings, used only when the user clicks "Auto-fill from EDGAR." All inputs remain user-editable. No Yahoo Finance dependency.
