import ast
import operator

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Meranie a grafy", layout="centered")

# ---------------------------------------------------------------- pomocné funkcie
OPERATORY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow,
             ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNKCIE = {"sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
           "exp": np.exp, "log": np.log, "log10": np.log10, "abs": np.abs}
KONSTANTY = {"pi": np.pi, "e": np.e}


def vypocet(vyraz, premenne):
    """Bezpečné vyhodnotenie vzorca (povolené sú len čísla, premenné, + - * / ^ a funkcie)."""
    vyraz = (vyraz.replace("^", "**").replace(",", ".")
             .replace("·", "*").replace("×", "*").replace("−", "-"))

    def _v(n):
        if isinstance(n, ast.Expression):
            return _v(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in premenne:
                return premenne[n.id]
            if n.id in KONSTANTY:
                return KONSTANTY[n.id]
            raise ValueError(f"Neznáma premenná: {n.id}")
        if isinstance(n, ast.BinOp) and type(n.op) in OPERATORY:
            return OPERATORY[type(n.op)](_v(n.left), _v(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPERATORY:
            return OPERATORY[type(n.op)](_v(n.operand))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in FUNKCIE and not n.keywords):
            return FUNKCIE[n.func.id](*[_v(a) for a in n.args])
        raise ValueError("Nepovolený výraz")

    return _v(ast.parse(vyraz, mode="eval"))


def g(v):
    return f"{v:.4g}"


def znak(v):
    return f" {'+' if v >= 0 else '−'} {abs(v):.4g}"


def nacitaj_csv(subor, rezim):
    """rezim: 2 = veličina + jednotka v prvých dvoch riadkoch, 1 = len názvy, 0 = len dáta."""
    surove = pd.read_csv(subor, sep=None, engine="python", header=None,
                         dtype=str, encoding="utf-8-sig")
    surove = surove.dropna(how="all").iloc[:, :26].reset_index(drop=True)
    n = surove.shape[1]
    stlpce = [chr(65 + i) for i in range(n)]
    velicina = list(surove.iloc[0].fillna("")) if rezim >= 1 else [""] * n
    jednotka = list(surove.iloc[1].fillna("")) if rezim == 2 else [""] * n
    telo = surove.iloc[rezim:].copy()
    telo.columns = stlpce
    telo = telo.apply(lambda s: pd.to_numeric(
        s.str.replace(",", ".", regex=False).str.strip(), errors="coerce"))
    if telo.empty:
        raise ValueError("Súbor neobsahuje žiadne dáta.")
    hl = pd.DataFrame([velicina, jednotka], index=["veličina", "jednotka"], columns=stlpce)
    return hl, telo.reset_index(drop=True).astype(float)


def trend(typ, x, y, nx, ny):
    """Vráti rovnicu, parametre, funkciu trendu a jej 1. a 2. deriváciu."""
    d1 = f"d{ny}/d{nx}"
    d2 = f"d²{ny}/d{nx}²"
    nula = lambda v: np.zeros(np.shape(v))

    if typ == "konštanta":
        c = float(np.mean(y))
        return dict(parametre={"c": c}, rovnica=f"{ny} = {g(c)}",
                    derivacia=f"{d1} = 0", derivacia2=f"{d2} = 0",
                    f=lambda v: np.full(np.shape(v), c), df=nula, d2f=nula)

    if typ == "lineárna":
        k, q = np.polyfit(x, y, 1)
        return dict(parametre={"k": k, "q": q},
                    rovnica=f"{ny} = {g(k)}·{nx}{znak(q)}",
                    derivacia=f"{d1} = {g(k)}", derivacia2=f"{d2} = 0",
                    f=lambda v: k * v + q,
                    df=lambda v: np.full(np.shape(v), k), d2f=nula)

    if typ == "kvadratická":
        a, b, c = np.polyfit(x, y, 2)
        return dict(parametre={"a": a, "b": b, "c": c},
                    rovnica=f"{ny} = {g(a)}·{nx}²{znak(b)}·{nx}{znak(c)}",
                    derivacia=f"{d1} = {g(2 * a)}·{nx}{znak(b)}",
                    derivacia2=f"{d2} = {g(2 * a)}",
                    f=lambda v: a * v**2 + b * v + c,
                    df=lambda v: 2 * a * v + b,
                    d2f=lambda v: np.full(np.shape(v), 2 * a))

    if typ == "druhá odmocnina":
        a, b = np.polyfit(np.sqrt(x), y, 1)
        return dict(parametre={"a": a, "b": b},
                    rovnica=f"{ny} = {g(a)}·√{nx}{znak(b)}",
                    derivacia=f"{d1} = {g(a / 2)}/√{nx}",
                    derivacia2=f"{d2} = {g(-a / 4)}·{nx}^(−3/2)",
                    f=lambda v: a * np.sqrt(v) + b,
                    df=lambda v: a / (2 * np.sqrt(v)),
                    d2f=lambda v: -a / (4 * v**1.5))

    if typ == "nepriama úmernosť":
        # y = a/x  ->  lineárna regresia cez počiatok pre u = 1/x
        u = 1.0 / x
        a = float(np.sum(u * y) / np.sum(u * u))
        return dict(parametre={"a": a},
                    rovnica=f"{ny} = {g(a)}/{nx}",
                    derivacia=f"{d1} = {g(-a)}/{nx}²",
                    derivacia2=f"{d2} = {g(2 * a)}/{nx}³",
                    f=lambda v: a / v,
                    df=lambda v: -a / v**2,
                    d2f=lambda v: 2 * a / v**3)

    if typ == "mocninová":
        # y = a·x^b  ->  ln y = b·ln x + ln a
        b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
        a = np.exp(ln_a)
        return dict(parametre={"a": a, "b": b},
                    rovnica=f"{ny} = {g(a)}·{nx}^{g(b)}",
                    derivacia=f"{d1} = {g(a * b)}·{nx}^{g(b - 1)}",
                    derivacia2=f"{d2} = {g(a * b * (b - 1))}·{nx}^{g(b - 2)}",
                    f=lambda v: a * v**b,
                    df=lambda v: a * b * v**(b - 1),
                    d2f=lambda v: a * b * (b - 1) * v**(b - 2))

    # exponenciálna
    b, ln_a = np.polyfit(x, np.log(y), 1)
    a = np.exp(ln_a)
    return dict(parametre={"a": a, "b": b},
                rovnica=f"{ny} = {g(a)}·e^({g(b)}·{nx})",
                derivacia=f"{d1} = {g(a * b)}·e^({g(b)}·{nx})",
                derivacia2=f"{d2} = {g(a * b * b)}·e^({g(b)}·{nx})",
                f=lambda v: a * np.exp(b * v),
                df=lambda v: a * b * np.exp(b * v),
                d2f=lambda v: a * b * b * np.exp(b * v))


TRENDY = ["žiadna", "konštanta", "lineárna", "kvadratická", "druhá odmocnina",
          "nepriama úmernosť", "mocninová", "exponenciálna"]
MIN_BODOV = {"konštanta": 1, "lineárna": 2, "kvadratická": 3, "druhá odmocnina": 2,
             "nepriama úmernosť": 1, "mocninová": 2, "exponenciálna": 2}


def zat(u):
    return f"({u})" if any(ch in u for ch in "/ ·*") else u


def jednotka_der(ux, uy, rad):
    """Jednotka derivácie: napr. m/s pre 1. a m/s² pre 2. deriváciu."""
    uxr = zat(ux) if rad == 1 else zat(ux) + "²"
    if uy and ux:
        return f"{zat(uy)}/{uxr}"
    if uy:
        return uy
    if ux:
        return f"1/{uxr}"
    return ""


def uprav_graf(fig, nazov, os_x, os_y, siet):
    """Názov a legenda nad grafom, sieť v oboch smeroch, malé okraje pre mobil."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=nazov, x=0.01, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=16)),
        margin=dict(l=10, r=10, t=100, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12)))
    osi = dict(showgrid=siet, gridcolor="#c8c8c8", gridwidth=1, zeroline=True,
               zerolinecolor="#808080", showline=True, linecolor="black",
               mirror=True, automargin=True)
    fig.update_xaxes(title_text=os_x, **osi)
    fig.update_yaxes(title_text=os_y, **osi)


def graf_derivacie(xs, hodnoty, rad, nx, ny, ux, uy, os_x, farba, siet):
    oznac = f"d{ny}/d{nx}" if rad == 1 else f"d²{ny}/d{nx}²"
    jed = jednotka_der(ux, uy, rad)
    hodnoty = np.where(np.isfinite(hodnoty), hodnoty, np.nan)
    fig = go.Figure(go.Scatter(
        x=xs, y=hodnoty, mode="lines", name=f"{rad}. derivácia trendu",
        line=dict(color=farba, width=3),
        hovertemplate=f"{nx} = %{{x:.4g}} {ux}<br>{oznac} = %{{y:.4g}} {jed}<extra></extra>"))
    uprav_graf(fig, f"{rad}. derivácia: {oznac}", os_x,
               f"{oznac} ({jed})" if jed else oznac, siet)
    return fig


def vypocitaj(data, hl, vzorce):
    """Zostaví zoznam všetkých stĺpcov (namerané + vypočítané) a premenné pre vzorce."""
    stlpce, premenne, chyby = {}, {}, []
    for c in data.columns:
        nazov = hl.loc["veličina", c].strip() or c
        stlpce[c] = dict(velicina=nazov, jednotka=hl.loc["jednotka", c].strip(),
                         hodnoty=data[c].to_numpy(float), vypocitany=False)
        premenne[c] = stlpce[c]["hodnoty"]
    for c in data.columns:
        nm = stlpce[c]["velicina"]
        if nm.isidentifier() and nm not in premenne:
            premenne[nm] = stlpce[c]["hodnoty"]
    for i, vz in enumerate(vzorce):
        kluc = chr(65 + len(data.columns) + i)
        try:
            with np.errstate(all="ignore"):
                res = vypocet(vz["vzorec"], premenne)
                res = np.broadcast_to(np.asarray(res, dtype=float), (len(data),)).copy()
            res[~np.isfinite(res)] = np.nan
        except Exception as e:
            chyby.append(f"Stĺpec {vz['nazov']}: {e}")
            res = np.full(len(data), np.nan)
        stlpce[kluc] = dict(velicina=vz["nazov"], jednotka=vz["jednotka"],
                            hodnoty=res, vypocitany=True)
        premenne[kluc] = res
        if vz["nazov"].isidentifier() and vz["nazov"] not in premenne:
            premenne[vz["nazov"]] = res
    return stlpce, premenne, chyby


# ---------------------------------------------------------------- stav aplikácie
if "data_df" not in st.session_state:
    st.session_state.data_df = pd.DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0], "B": [0.0, 1.1, 3.9, 9.2, 15.8]})
    st.session_state.hl_df = pd.DataFrame(
        {"A": ["t", "s"], "B": ["s", "m"]}, index=["veličina", "jednotka"])
    st.session_state.hl_cur = st.session_state.hl_df.copy()
    st.session_state.verzia = 0   # mení sa pri zmene štruktúry (CSV, pridanie/odobratie stĺpca)
    st.session_state.edit_n = 0   # mení sa po každej úprave hodnôt v tabuľke
    st.session_state.vzorce = []

st.title("Meranie a grafy")

# ---------------------------------------------------------------- nahratie CSV
with st.expander("📂 Nahrať dáta z CSV"):
    texty = {2: "Prvý riadok = veličina, druhý = jednotka",
             1: "Prvý riadok = názvy stĺpcov (bez jednotiek)",
             0: "Súbor obsahuje iba dáta"}
    rezim = st.radio("Formát súboru", [2, 1, 0], format_func=lambda i: texty[i])
    subor = st.file_uploader("Vyberte súbor CSV", type=["csv", "txt"])
    nacitane = False
    if subor is not None:
        id_suboru = (subor.name, subor.size, rezim)
        if st.session_state.get("posledny_subor") != id_suboru:
            try:
                hl_n, data_n = nacitaj_csv(subor, rezim)
                st.session_state.hl_df = hl_n
                st.session_state.hl_cur = hl_n.copy()
                st.session_state.data_df = data_n
                st.session_state.vzorce = []
                st.session_state.verzia += 1
                st.session_state.edit_n = 0
                st.session_state.posledny_subor = id_suboru
                nacitane = True
            except Exception as e:
                st.error(f"Súbor sa nepodarilo načítať: {e}")
    if nacitane:
        st.rerun()

# ---------------------------------------------------------------- tabuľka (namerané + vypočítané)
data0 = st.session_state.data_df.reset_index(drop=True).astype(float)
n_meas = data0.shape[1]
vzorce = st.session_state.vzorce
v = st.session_state.verzia
kluce_vyp = [chr(65 + n_meas + i) for i in range(len(vzorce))]

# hlavička: veličina + jednotka (vypočítané stĺpce sú len na čítanie)
st.subheader("Veličiny a jednotky")
hl_vstup = st.session_state.hl_df.copy()
for kluc, vz in zip(kluce_vyp, vzorce):
    hl_vstup[kluc] = [vz["nazov"], vz["jednotka"]]
hl = st.data_editor(hl_vstup, key=f"hl_{v}", disabled=kluce_vyp)
hl = hl.fillna("").astype(str)
st.session_state.hl_cur = hl[list(data0.columns)].copy()

# hodnoty všetkých stĺpcov
stlpce, premenne, chyby = vypocitaj(data0, st.session_state.hl_cur, vzorce)

tab = data0.copy()
for kluc in kluce_vyp:
    tab[kluc] = stlpce[kluc]["hodnoty"]
cfg = {}
for kluc in tab.columns:
    if kluc in kluce_vyp:
        cfg[kluc] = st.column_config.NumberColumn(f"{kluc} 🧮", disabled=True)
    else:
        cfg[kluc] = st.column_config.NumberColumn(kluc)

st.subheader("Namerané hodnoty")
upravene = st.data_editor(tab, num_rows="dynamic", column_config=cfg,
                          key=f"data_{v}_{st.session_state.edit_n}")
nove = (upravene[list(data0.columns)]
        .apply(pd.to_numeric, errors="coerce")
        .reset_index(drop=True).astype(float))
if not nove.equals(data0):
    # po úprave uložíme nové hodnoty a prekreslíme, aby sa vypočítané stĺpce prepočítali
    st.session_state.data_df = nove
    st.session_state.edit_n += 1
    st.rerun()

if kluce_vyp:
    st.caption("🧮 = vypočítaný stĺpec (len na čítanie).")
for ch in chyby:
    st.error(ch)


def popis(k):
    s = stlpce[k]
    return f"{s['velicina']} ({s['jednotka']})" if s["jednotka"] else s["velicina"]


# odstránenie vypočítaného stĺpca
for i, vz in enumerate(vzorce):
    if st.button(f"🗑 Odstrániť stĺpec {vz['nazov']} = {vz['vzorec']}", key=f"del_{i}"):
        st.session_state.hl_df = st.session_state.hl_cur.copy()
        st.session_state.vzorce.pop(i)
        st.session_state.verzia += 1
        st.rerun()

# pridanie vypočítaného stĺpca
with st.expander("➕ Pridať stĺpec so vzorcom"):
    st.caption("Premenné: " + ", ".join(
        f"{k} ({stlpce[k]['velicina']})" if stlpce[k]["velicina"] != k else k
        for k in stlpce) + ". Môžete písať aj názov veličiny, napr. s/t.")
    st.caption("Operácie: + − * / ^ · funkcie: sqrt, sin, cos, tan, exp, log, log10, abs · konštanta: pi")
    with st.form("vzorec_form", clear_on_submit=True):
        n_nazov = st.text_input("Veličina nového stĺpca (napr. v)")
        n_jedn = st.text_input("Jednotka (napr. m/s)")
        n_vzorec = st.text_input("Vzorec (napr. s/t)")
        if st.form_submit_button("Pridať stĺpec") and n_vzorec.strip():
            if n_meas + len(vzorce) >= 26:
                st.error("Dosiahli ste maximálny počet stĺpcov.")
            else:
                st.session_state.hl_df = st.session_state.hl_cur.copy()
                st.session_state.vzorce.append(dict(
                    nazov=n_nazov.strip() or f"stĺpec{len(vzorce) + 1}",
                    jednotka=n_jedn.strip(), vzorec=n_vzorec.strip()))
                st.session_state.verzia += 1
                st.rerun()

# ---------------------------------------------------------------- nastavenie grafu
st.subheader("Graf")
moznosti = list(stlpce.keys())
c1, c2 = st.columns(2)
kx = c1.selectbox("Os x", moznosti, index=0, format_func=popis)
ky = c2.selectbox("Os y", moznosti, index=1 if len(moznosti) > 1 else 0, format_func=popis)

nx, ux = stlpce[kx]["velicina"], stlpce[kx]["jednotka"]
ny, uy = stlpce[ky]["velicina"], stlpce[ky]["jednotka"]

typ = st.selectbox("Trendová čiara", TRENDY)

with st.expander("🎨 Úprava grafu"):
    nazov = st.text_input("Názov grafu", f"Závislosť {ny}({nx})",
                          key=f"nazov_{kx}_{ky}_{nx}_{ny}")
    farba = st.color_picker("Farba bodov", "#1f77b4")
    farba_t = st.color_picker("Farba trendovej čiary", "#d62728")
    velkost = st.slider("Veľkosť bodov", 4, 20, 10)
    siet = st.checkbox("Zobraziť sieť", value=True)

d = pd.DataFrame({"x": stlpce[kx]["hodnoty"], "y": stlpce[ky]["hodnoty"]}).dropna()
x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)

t = None
if typ != "žiadna":
    if len(x) < MIN_BODOV[typ]:
        st.warning(f"Pre tento trend je potrebných aspoň {MIN_BODOV[typ]} body.")
    elif typ == "druhá odmocnina" and (x < 0).any():
        st.warning("Druhá odmocnina vyžaduje, aby hodnoty na osi x neboli záporné.")
    elif typ == "nepriama úmernosť" and ((x == 0).any() or (x.min() < 0 < x.max())):
        st.warning("Nepriama úmernosť nemôže mať x = 0 a hodnoty x musia mať rovnaké znamienko.")
    elif typ == "mocninová" and ((x <= 0).any() or (y <= 0).any()):
        st.warning("Mocninová funkcia vyžaduje kladné hodnoty na oboch osiach.")
    elif typ == "exponenciálna" and (y <= 0).any():
        st.warning("Exponenciálny trend vyžaduje kladné hodnoty na osi y.")
    else:
        t = trend(typ, x, y, nx, ny)

# ---------------------------------------------------------------- hlavný graf
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers", name="namerané hodnoty",
    marker=dict(size=velkost, color=farba),
    hovertemplate=f"{nx} = %{{x:.4g}} {ux}<br>{ny} = %{{y:.4g}} {uy}<extra></extra>"))

xs = None
if t is not None and len(x) > 0:
    xs = np.linspace(x.min(), x.max(), 200)
    with np.errstate(all="ignore"):
        ys = t["f"](xs)
    ys = np.where(np.isfinite(ys), ys, np.nan)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=f"trend: {typ}",
        line=dict(color=farba_t, width=3),
        hovertemplate=f"{nx} = %{{x:.4g}} {ux}<br>{ny} = %{{y:.4g}} {uy}<extra></extra>"))

uprav_graf(fig, nazov, popis(kx), popis(ky), siet)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- parametre trendu
if t is not None:
    with np.errstate(all="ignore"):
        ss_res = np.sum((y - t["f"](x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    st.success(f"{t['rovnica']}   |   R² = {g(r2)}")
    st.markdown("**Parametre:** " + ", ".join(f"{k} = {g(val)}" for k, val in t["parametre"].items()))

    # ------------------------------------------------------------ derivácie trendu
    st.markdown("**Derivácie trendovej čiary**")
    cd1, cd2 = st.columns(2)
    zobraz1 = cd1.checkbox("Zobraziť 1. deriváciu")
    zobraz2 = cd2.checkbox("Zobraziť 2. deriváciu")

    if zobraz1:
        st.info(t["derivacia"])
        with np.errstate(all="ignore"):
            h1 = t["df"](xs)
        st.plotly_chart(graf_derivacie(xs, h1, 1, nx, ny, ux, uy, popis(kx), "#2ca02c", siet),
                        width="stretch")
    if zobraz2:
        st.info(t["derivacia2"])
        with np.errstate(all="ignore"):
            h2 = t["d2f"](xs)
        st.plotly_chart(graf_derivacie(xs, h2, 2, nx, ny, ux, uy, popis(kx), "#9467bd", siet),
                        width="stretch")
