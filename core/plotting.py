from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["equity_curve"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")
    fig.autofmt_xdate()
    return fig
