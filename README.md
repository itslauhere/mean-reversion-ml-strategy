# 📈 Mean Reversion Strategy with Machine Learning Signal

A backtesting engine for mean-reverting trading strategies, enhanced with a machine learning classifier to improve trade entry signals. Built using Python, pandas, and scikit-learn.

---

## 🚀 Overview

This project simulates a **mean reversion trading strategy** that buys when an asset's price deviates significantly below its average, assuming it will return ("revert") to the mean.

To improve signal quality, it also trains a **Random Forest Classifier** on technical indicators like Z-score, momentum, and volatility to predict next-day returns and filter trades.

---

## 📦 Features

- ✅ Z-score–based statistical entry/exit rules  
- ✅ Random Forest ML filter to reduce false trades  
- ✅ Portfolio-ready trade log export (CSV)  
- ✅ Strategy performance metrics:
  - Sharpe Ratio  
  - CAGR (Compound Annual Growth Rate)  
  - Maximum Drawdown  
  - Total P&L  
- ✅ Strategy optimizer for multiple assets and thresholds  
- ✅ Visualization of cumulative strategy vs market returns

---

## 🧠 Strategy Logic

- **Buy** when:
  - Z-score is below a negative entry threshold (e.g. -1.5)  
  - and the ML model predicts a price increase  
- **Sell** when:
  - Z-score crosses above an exit threshold (e.g. 0)

Trades are simulated, logged, and evaluated over historical data.

---

## 📂 Folder Structure

```
mean-reversion-ml-strategy/
├── data/                      # CSVs with historical prices (e.g., AAPL.csv, TSLA.csv)
├── mean_reversion.py          # Main backtester script
├── requirements.txt           # Project dependencies
├── strategy_summary_table.csv # Output with Sharpe/CAGR/P&L/Drawdown per asset
└── README.md                  # This file
```

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/itslauhere/mean-reversion-ml-strategy.git
cd mean-reversion-ml-strategy
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Data
Place historical `.csv` files in the `data/` folder. Files must contain:
- A `Date` column  
- A `Close` or `Adj Close` column

Example sources:
- [Yahoo Finance](https://finance.yahoo.com)
- [Alpha Vantage](https://www.alphavantage.co)
- [Wall Street Journal](https://www.wsj.com)

---

## ▶️ How to Run

```bash
python mean_reversion.py
```

The script will:
- Process each CSV in `data/`  
- Run strategy optimization  
- Save trade logs for each asset  
- Print and plot metrics  
- Output a summary CSV: `strategy_summary_table.csv`

---

## 📊 Example Output

| Asset | Sharpe Ratio | CAGR (%) | Total P&L (%) | Max Drawdown (%) |
|-------|--------------|----------|----------------|-------------------|
| AAPL  | 1.42         | 12.3     | 31.7           | -8.5              |
| TSLA  | 0.95         | 9.8      | 24.1           | -12.4             |

---

## 📌 Notes

- This is a simplified educational tool and **not financial advice**.
- You can expand it by:
  - Adding more ML features (RSI, MACD, volume)
  - Saving and loading models
  - Hyperparameter tuning with grid search
  - Portfolio-level backtesting

---

## 👩‍💻 Author

Built by [Laurentia Liennart](https://github.com/itslauhere) — Cognitive Science @ UCSD with Machine Learning & Neural Computation specialization.  
Experience in Southeast Asian markets as an Investment Analyst @ Kayana Kapital.

