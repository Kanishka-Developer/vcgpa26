from __future__ import annotations

import altair as alt
import pandas as pd


def hist_numeric(df: pd.DataFrame, col: str, title: str = "Histogram", bins: int = 20):
    base = alt.Chart(df.dropna(subset=[col])).mark_bar().encode(
        alt.X(col, bin=alt.Bin(maxbins=bins), title=col),
        y="count()",
        tooltip=[alt.Tooltip(col, aggregate="count", title="Count")],
    ).properties(title=title)
    return base.interactive()


def box_numeric(df: pd.DataFrame, col: str, by: str | None = None, title: str = "Box plot"):
    data = df.dropna(subset=[col])
    enc = {
        "y": alt.Y(col, title=col),
        "tooltip": [alt.Tooltip(col, aggregate="mean", title="Mean")],
    }
    if by:
        enc["x"] = alt.X(by, title=by)
    chart = alt.Chart(data).mark_boxplot().encode(**enc).properties(title=title)
    return chart.interactive()


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "Scatter"):
    data = df.dropna(subset=[x, y])
    enc = {
        "x": alt.X(x, title=x),
        "y": alt.Y(y, title=y),
        "tooltip": [x, y],
    }
    if color and color in df.columns:
        enc["color"] = alt.Color(color)
    chart = alt.Chart(data).mark_circle(size=60, opacity=0.6).encode(**enc).properties(title=title)
    return chart.interactive()


def bar_count(df: pd.DataFrame, cat: str, title: str = "Counts"):
    data = df[df[cat].notna() & (df[cat] != "")]
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(cat, sort='-y', title=cat),
        y=alt.Y('count()', title='Count'),
        tooltip=[alt.Tooltip(cat, title=cat), alt.Tooltip('count()', title='Count')]
    ).properties(title=title)
    return chart.interactive()


def heatmap(df: pd.DataFrame, x: str, y: str, value: str, title: str = "Heatmap"):
    data = df.dropna(subset=[x, y, value])
    chart = alt.Chart(data).mark_rect().encode(
        x=alt.X(x, title=x),
        y=alt.Y(y, title=y),
        color=alt.Color(value, title=value, scale=alt.Scale(scheme='blues')),
        tooltip=[x, y, value]
    ).properties(title=title)
    return chart.interactive()
