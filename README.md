# DCF Equity Valuation App

**FINA 4011/5011 — Project 2**

A Streamlit application that performs a Discounted Cash Flow (DCF) valuation on any publicly traded stock. Users can input valuation assumptions (growth, margin, WACC, etc.) and the app computes an intrinsic value per share, compares it to the current market price, and runs sensitivity analysis.

## Features

- **Live market data** via Yahoo Finance (`yfinance`)
- **10-year FCF projection** with user-controlled growth and margin assumptions
- **Terminal value** via Gordon Growth Model
- **Step-by-step walkthrough** of every calculation with formula explanations
- **Two sensitivity tables** (WACC × terminal growth; growth × margin)
- **Margin-of-safety** output and visual comparison vs. market price

## Run Locally

```bash
pip install -r requirements.txt
streamlit run equity_valuation_app.py
```

## Live App

Deployed on Streamlit Community Cloud.
