# DCF Equity Valuation App

A Streamlit application for Discounted Cash Flow (DCF) equity valuation. Users enter (or auto-fill from SEC filings) financial inputs, adjust valuation assumptions, and the app computes intrinsic value per share, compares it to current market price, and runs sensitivity analysis.

## Features

- **Manual-input DCF** — every assumption fully editable in the sidebar
- **Optional SEC EDGAR auto-fill** — pulls the latest 10-K fundamentals (revenue, FCF, debt, cash, shares) from the SEC's official API. No API key, no rate limits.
- **CAPM WACC build-up** — optional toggle to compute WACC from risk-free rate, beta, equity risk premium, cost of debt, and target capital structure
- **Multi-stage growth** — constant or 3-stage (high-growth → linear fade → terminal)
- **Margin expansion path** — optional linear interpolation between Year 1 and terminal margin
- **Terminal value methods** — Gordon Growth or Exit Multiple (EV/EBITDA)
- **Mid-year discounting** — banker-standard convention
- **Step-by-step walkthrough** — formulas + intermediate values for each valuation step
- **Sensitivity analysis** — two tables (WACC × terminal growth, growth × EBIT margin) with color gradient
- **Plotly charts** — FCF projection bars, market vs. intrinsic value comparison

## Run Locally

```bash
pip install -r requirements.txt
streamlit run equity_valuation_app.py
```

## Live App

**https://fina-project2-dcf-valuation.streamlit.app/**

## Data Source

SEC EDGAR Company Facts API (`data.sec.gov`) — official 10-K filings, used only when the user clicks "Auto-fill from EDGAR." All inputs remain user-editable.
