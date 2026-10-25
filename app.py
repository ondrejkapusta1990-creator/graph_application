import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Meranie a grafy")

# Hlavička: veličina a jednotka pre každý stĺpec
st.subheader("Veličina a jednotka")
hl = st.data_editor(
    pd.DataFrame({"A": ["t", "s"], "B": ["s", "m"]},
                 index=["veličina", "jednotka"]),
    key="hlavicka")

# Dáta
st.subheader("Namerané hodnoty")
data = st.data_editor(
    pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0, 4.0],
                  "B": [0.0, 1.1, 3.9, 9.2, 15.8]}),
    num_rows="dynamic", key="data").dropna()

x, y = data["A"].to_numpy(float), data["B"].to_numpy(float)
nx, ux = hl.loc["veličina", "A"], hl.loc["jednotka", "A"]
ny, uy = hl.loc["veličina", "B"], hl.loc["jednotka", "B"]

# Nastavenia grafu
typ = st.selectbox("Trendová čiara",
                   ["žiadna", "lineárna", "kvadratická", "exponenciálna"])
farba = st.color_picker("Farba bodov", "#1f77b4")
nazov = st.text_input("Názov grafu", f"Závislosť {ny}({nx})")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers", marker=dict(size=10, color=farba),
    name="namerané",
    hovertemplate=f"{nx} = %{{x}} {ux}<br>{ny} = %{{y}} {uy}<extra></extra>"))

if typ != "žiadna" and len(x) >= 3:
    xs = np.linspace(x.min(), x.max(), 200)
    if typ == "lineárna":
        k, q = np.polyfit(x, y, 1)
        f = lambda v: k * v + q
        rovnica = f"{ny} = {k:.4g}·{nx} + {q:.4g}"
    elif typ == "kvadratická":
        a, b, c = np.polyfit(x, y, 2)
        f = lambda v: a * v**2 + b * v + c
        rovnica = f"{ny} = {a:.4g}·{nx}² + {b:.4g}·{nx} + {c:.4g}"
    else:
        b, ln_a = np.polyfit(x, np.log(y), 1)  # vyžaduje y > 0
        a = np.exp(ln_a)
        f = lambda v: a * np.exp(b * v)
        rovnica = f"{ny} = {a:.4g}·e^({b:.4g}·{nx})"
    ss_res = np.sum((y - f(x))**2)
    r2 = 1 - ss_res / np.sum((y - y.mean())**2)
    fig.add_trace(go.Scatter(x=xs, y=f(xs), mode="lines", name="trend"))
    st.success(f"{rovnica}   |   R² = {r2:.4f}")

fig.update_layout(title=nazov,
                  xaxis_title=f"{nx} ({ux})",
                  yaxis_title=f"{ny} ({uy})")
st.plotly_chart(fig, use_container_width=True)
