# Mean-Reversion Strategy Backtester with ML Signal
# Tools: Python, Pandas, NumPy, Matplotlib, Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load sample historical price data from a general CSV file
def get_price_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]
    date_col = next((col for col in df.columns if 'date' in col), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)
    else:
        raise ValueError("CSV must contain a date-related column like 'Date'")
    close_candidates = ['close', 'adj close']
    close_col = next((col for col in close_candidates if col in df.columns), None)
    if close_col is None:
        raise ValueError("CSV must contain a 'Close' or 'Adj Close' column")
    df = df[[close_col]].rename(columns={close_col: 'Close'})
    df.dropna(inplace=True)
    return df

# Step 2: Feature engineering for ML
def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1
    df['Volatility'] = df['Return'].rolling(5).std()
    df.dropna(inplace=True)
    return df

def create_labels(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def train_predictor(df):
    features = ['ZScore', 'Momentum', 'Volatility']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    df['MLSignal'] = model.predict(X)
    return df

# Step 3: Define entry/exit signals and simulate P&L, return trades list
def simulate_strategy(df, entry_z=-1, exit_z=0, asset_name='asset'):
    df = add_features(df)
    df = create_labels(df)
    df = train_predictor(df)

    df['Position'] = 0
    df.loc[(df['ZScore'] < entry_z) & (df['MLSignal'] == 1), 'Position'] = 1
    df.loc[df['ZScore'] > exit_z, 'Position'] = 0
    df['Position'] = df['Position'].shift().fillna(0)
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['StrategyReturns'] = df['Position'] * df['Returns']
    df['CumulativeStrategy'] = (1 + df['StrategyReturns']).cumprod()

    trades = df[df['Position'].diff() != 0].copy()
    trades['Signal'] = trades['Position'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
    trades = trades[['Close', 'ZScore', 'Signal', 'Position', 'Returns', 'CumulativeStrategy']]
    trades.to_csv(f'{asset_name}_trades_log.csv')
    return df

# Step 4: Plot results and print performance metrics
def plot_strategy(df, asset_name='Asset'):
    df['CumulativeReturns'] = (1 + df['Returns']).cumprod()
    df['CumulativeStrategy'] = (1 + df['StrategyReturns']).cumprod()

    sharpe = (df['StrategyReturns'].mean() / df['StrategyReturns'].std()) * np.sqrt(252)
    print(f"{asset_name} Sharpe Ratio: {sharpe:.2f}")
    peak = df['CumulativeStrategy'].cummax()
    drawdown = (df['CumulativeStrategy'] - peak) / peak
    max_dd = drawdown.min()
    print(f"{asset_name} Max Drawdown: {max_dd:.2%}")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['CumulativeReturns'], label='Market')
    plt.plot(df.index, df['CumulativeStrategy'], label='Strategy')
    plt.legend()
    plt.title(f'{asset_name} - Mean Reversion Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

# Step 5: Run optimizer for multiple assets or Z-scores
def optimize_strategies(folder_path, entry_list=[-1, -1.5], exit_list=[-0.1, 0]):
    summary = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            asset_path = os.path.join(folder_path, file)
            asset_name = os.path.splitext(file)[0]
            print(f"\nProcessing {asset_name}")
            try:
                data = get_price_data(asset_path)
                best_sharpe = -np.inf
                best_df = None
                best_pnl = 0
                best_cagr = 0
                best_dd = 0
                for e_in in entry_list:
                    for e_out in exit_list:
                        df_copy = data.copy()
                        df_copy = simulate_strategy(df_copy, e_in, e_out, asset_name)
                        strategy_returns = df_copy['StrategyReturns']
                        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
                        cumulative = (1 + strategy_returns).cumprod()
                        total_pnl = cumulative.iloc[-1] - 1
                        days = (df_copy.index[-1] - df_copy.index[0]).days
                        cagr = (cumulative.iloc[-1]) ** (252 / days) - 1
                        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
                        max_dd = drawdown.min()
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_df = df_copy.copy()
                            best_pnl = total_pnl
                            best_cagr = cagr
                            best_dd = max_dd
                if best_df is not None:
                    plot_strategy(best_df, asset_name)
                    summary.append({
                        'Asset': asset_name,
                        'Sharpe Ratio': round(best_sharpe, 2),
                        'Total P&L': round(best_pnl * 100, 2),
                        'CAGR': round(best_cagr * 100, 2),
                        'Max Drawdown': round(best_dd * 100, 2)
                    })
            except Exception as e:
                print(f"Failed on {asset_name}: {e}")
    print("\n=== Summary Table ===")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    df_summary.to_csv('strategy_summary_table.csv', index=False)
    if not df_summary.empty:
        df_summary.set_index('Asset')[['Sharpe Ratio', 'Total P&L', 'CAGR', 'Max Drawdown']].plot(
            kind='bar', figsize=(14, 6), title='Strategy Metrics per Asset')
        plt.ylabel('Metric Value')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

# Example usage
try:
    optimize_strategies('/Users/laurentialiennart/Downloads/HistoricalPrices')
except Exception as e:
    print(f"An error occurred: {e}")
# This code will process all CSV files in the specified folder, apply the mean-reversion strategy,
# and output a summary of the results.
# Ensure the folder path is correct and contains valid CSV files with historical price data.
# Adjust the entry and exit Z-scores as needed for optimization.
# The code will also plot the cumulative returns of the strategy against the market.
# Make sure to have the required libraries installed: numpy, pandas, matplotlib, scikit-learn.
# You can install them using pip if they are not already installed:
# pip install numpy pandas matplotlib scikit-learn
