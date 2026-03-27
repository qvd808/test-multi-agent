import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── CONFIGURATION ────────────────────────────────────────────────────────────
CSV_FILE = "margin_guard_audit_20260326_2018.csv"
DOWNSAMPLE_FACTOR = 1000  # Plot every 1000th step to save memory
# ──────────────────────────────────────────────────────────────────────────────

def plot_stress_audit():
    print(f"Reading {CSV_FILE}...")
    # Read only necessary columns to save RAM
    df = pd.read_csv(CSV_FILE, usecols=['step', 'price', 'balance', 'position', 'action', 'reward', 'vetoed'])
    
    # Calculate Portfolio Value: Cash + (Position * Price)
    df['portfolio'] = df['balance'] + (df['position'] * df['price'])
    
    # Calculate Drawdown
    df['max_portfolio'] = df['portfolio'].cummax()
    df['drawdown'] = df['portfolio'] - df['max_portfolio']
    
    # Downsample for plotting
    df_plot = df.iloc[::DOWNSAMPLE_FACTOR, :]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # ── PANEL 1: The Equity Curve ──
    ax1.plot(df_plot['step'], df_plot['portfolio'], color='#2ca02c', label='Total Equity')
    ax1.fill_between(df_plot['step'], df_plot['portfolio'], 50000, 
                     where=(df_plot['portfolio'] < 50000), color='red', alpha=0.1)
    ax1.axhline(50000, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("1. Portfolio Value over 7.6M Steps (Equity Curve)", fontweight='bold')
    ax1.set_ylabel("USD")
    ax1.legend()

    # ── PANEL 2: Underwater Plot (Drawdown) ──
    ax2.fill_between(df_plot['step'], df_plot['drawdown'], 0, color='red', alpha=0.3)
    ax2.set_title("2. Drawdown Depth (The 'Pain' Chart)", fontweight='bold')
    ax2.set_ylabel("USD Below Peak")

    # ── PANEL 3: Action Distribution (Strategy Audit) ──
    # We use a rolling mean for vetoes to see if the agent "relapsed"
    df_plot['rolling_veto'] = df_plot['vetoed'].rolling(window=50).mean() * 100
    ax3.plot(df_plot['step'], df_plot['rolling_veto'], color='orange', label='Veto Rate %')
    ax3.set_title("3. Rolling Veto Rate (Compliance Consistency)", fontweight='bold')
    ax3.set_ylabel("Veto %")
    ax3.set_ylim(0, 5) # Zoom in on the low veto rate
    ax3.set_xlabel("Total Steps")

    # Add Summary Statistics box
    max_dd = df['drawdown'].min()
    final_return = ((df['portfolio'].iloc[-1] - 50000) / 50000) * 100
    
    stats_text = (f"Final Return: {final_return:+.2f}%\n"
                  f"Max Drawdown: ${abs(max_dd):,.2f}\n"
                  f"Total Steps: {len(df):,}\n"
                  f"Veto Consistency: {100 - (df['vetoed'].mean()*100):.2f}%")
    
    fig.text(0.15, 0.05, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig("stress_test_analysis.png", dpi=200)
    print("✓ Analysis complete. Chart saved to stress_test_analysis.png")

if __name__ == "__main__":
    plot_stress_audit()