import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations

st.set_page_config(page_title="Simulación Monte Carlo", layout="wide")

# ==============================
# Sidebar - Parámetros del modelo
# ==============================
st.sidebar.header("⚙️ Parámetros del modelo")

base_goles = st.sidebar.slider("Base de goles por equipo", 0.8, 2.2, 1.4, 0.1)
factor_local = st.sidebar.slider("Ventaja de jugar en casa", 1.0, 1.5, 1.1, 0.05)

st.sidebar.subheader("Pesos ofensivos")
w_goles_marcados = st.sidebar.slider("Goles marcados", 0.0, 1.0, 0.5, 0.05)
w_goles_fuera   = st.sidebar.slider("Goles a domicilio", 0.0, 1.0, 0.2, 0.05)
w_victorias     = st.sidebar.slider("Victorias", 0.0, 1.0, 0.1, 0.05)
w_coef          = st.sidebar.slider("Coef. club", 0.0, 1.0, 0.2, 0.05)

st.sidebar.subheader("Pesos defensivos")
w_diff_goles = st.sidebar.slider("Dif. goles", 0.0, 1.0, 0.5, 0.05)
w_goles_enc  = st.sidebar.slider("Goles encajados", 0.0, 1.0, 0.3, 0.05)
w_disciplina = st.sidebar.slider("Disciplina", 0.0, 1.0, 0.2, 0.05)

num_simulaciones = st.sidebar.slider("Número de simulaciones", 500, 50000, 10000, 500)

# ==============================
# Helpers de modelo
# ==============================
def normalizar(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

# ==============================
# Carga de datos
# ==============================
st.title("⚽ Simulación Monte Carlo (liga suiza)")
st.markdown("Simulador de Monte Carlo con sliders para predecir clasificación tras última jornada.")

st.write("@diegoparicio")

st.sidebar.markdown("---")

base_df = pd.read_csv("montecarlo/champions_2025_26_clasificacion.csv")

required_cols = {
    "equipo", "puntos", "diferencia_goles", "goles_marcados", "goles_a_domicilio",
    "victorias", "victorias_domicilio", "opponents_points", "opponents_goal_difference",
    "opponents_goals_scored", "disciplinary_points", "club_coefficient",
}
missing = required_cols - set(base_df.columns)
if missing:
    st.error(f"Faltan columnas en el CSV: {missing}")
    st.stop()

# ==============================
# Normalizar pesos
# ==============================
suma_of = w_goles_marcados + w_goles_fuera + w_victorias + w_coef
suma_def = w_diff_goles + w_goles_enc + w_disciplina

if suma_of > 0:
    w_goles_marcados /= suma_of
    w_goles_fuera   /= suma_of
    w_victorias     /= suma_of
    w_coef          /= suma_of

if suma_def > 0:
    w_diff_goles /= suma_def
    w_goles_enc  /= suma_def
    w_disciplina /= suma_def

# ==============================
# Calcular fuerzas
# ==============================
df = base_df.copy()

df["fuerza_ofensiva"] = (
    normalizar(df["goles_marcados"])    * w_goles_marcados +
    normalizar(df["goles_a_domicilio"]) * w_goles_fuera   +
    normalizar(df["victorias"])         * w_victorias     +
    normalizar(df["club_coefficient"])  * w_coef
)

df["fuerza_defensiva"] = (
    normalizar(-df["diferencia_goles"])       * w_diff_goles +
    normalizar(-df["opponents_goals_scored"]) * w_goles_enc  +
    normalizar(df["disciplinary_points"])    * w_disciplina
)

st.subheader("📊 Datos base (con fuerzas calculadas)")
st.dataframe(df)

# ==============================
# Simulación de partidos
# ==============================
def simular_goles(lambda_goles):
    return np.random.poisson(max(lambda_goles, 0.1))


def simular_partido(equipo_a, equipo_b, df, factor_local, base_goles):
    a = df[df["equipo"] == equipo_a].iloc[0]
    b = df[df["equipo"] == equipo_b].iloc[0]

    lambda_a = base_goles * a["fuerza_ofensiva"] * (2 - b["fuerza_defensiva"]) * factor_local
    lambda_b = base_goles * b["fuerza_ofensiva"] * (2 - a["fuerza_defensiva"])

    return simular_goles(lambda_a), simular_goles(lambda_b)


def desempate_uefa(tabla):
    return tabla.sort_values(
        by=["puntos", "diferencia_goles", "goles_marcados", "victorias", "opponents_points", "club_coefficient"],
        ascending=[False]*6
    ).reset_index(drop=True)


def simular_liga(df, factor_local, base_goles, calendario=None):
    equipos = df["equipo"].tolist()
    puntos = {e: df.set_index("equipo").loc[e, "puntos"] for e in equipos}
    goles = {e: df.set_index("equipo").loc[e, "goles_marcados"] for e in equipos}
    dg = {e: df.set_index("equipo").loc[e, "diferencia_goles"] for e in equipos}
    victorias = {e: df.set_index("equipo").loc[e, "victorias"] for e in equipos}

    partidos = calendario if calendario is not None else list(combinations(equipos, 2))

    for a, b in partidos:
        ga, gb = simular_partido(a, b, df, factor_local, base_goles)
        goles[a] += ga
        goles[b] += gb
        dg[a] += ga - gb
        dg[b] += gb - ga
        if ga > gb:
            puntos[a] += 3
            victorias[a] += 1
        elif gb > ga:
            puntos[b] += 3
            victorias[b] += 1
        else:
            puntos[a] += 1
            puntos[b] += 1

    tabla = pd.DataFrame({
        "equipo": equipos,
        "puntos": [puntos[e] for e in equipos],
        "diferencia_goles": [dg[e] for e in equipos],
        "goles_marcados": [goles[e] for e in equipos],
        "victorias": [victorias[e] for e in equipos],
    })

    tabla = tabla.merge(df[["equipo", "opponents_points", "club_coefficient"]], on="equipo", how="left")
    return desempate_uefa(tabla)


def simular_montecarlo(df, factor_local, base_goles, num_sim, calendario=None):
    equipos = df["equipo"].tolist()
    posiciones = {e: [] for e in equipos}

    progress = st.progress(0)
    for i in range(num_sim):
        tabla = simular_liga(df, factor_local, base_goles, calendario)
        for pos, equipo in enumerate(tabla["equipo"], start=1):
            posiciones[equipo].append(pos)
        if i % max(1, num_sim // 100) == 0:
            progress.progress(i / num_sim)
    progress.progress(1.0)
    return posiciones

# ==============================
# Calendario última jornada
# ==============================
calendario_ultima_jornada = [
    ("Paris", "Newcastle"), ("Man City", "Galatasaray"), ("Liverpool", "Qarabag"),
    ("B. Dortmund", "Inter"), ("Barcelona", "Copenhagen"), ("Arsenal", "Kairat Almaty"),
    ("Leverkusen", "Villarreal"), ("Atleti", "Bodø/Glimt"), ("Benfica", "Real Madrid"),
    ("Frankfurt", "Tottenham"), ("Club Brugge", "Marseille"), ("PSV", "Bayern München"),
    ("Ajax", "Olympiacos"), ("Napoli", "Chelsea"), ("Monaco", "Juventus"),
    ("Union SG", "Atalanta"), ("Athletic Club", "Sporting CP"), ("Pafos", "Slavia Praha")
]

st.subheader("🗓️ Partidos de la última jornada")
st.write(pd.DataFrame(calendario_ultima_jornada, columns=["Local", "Visitante"]))

# ==============================
# Ejecución
# ==============================
if st.button("🚀 Simular última jornada"):
    posiciones = simular_montecarlo(df, factor_local, base_goles, num_simulaciones, calendario=calendario_ultima_jornada)

    resultados = {e: pd.Series(posiciones[e]).value_counts(normalize=True).sort_index() for e in posiciones}
    df_probs = pd.DataFrame(resultados).fillna(0)
    st.subheader("📈 Probabilidades de posición final tras última jornada")
    st.dataframe(df_probs.style.format("{:.2%}"))

    media_pos = {e: np.mean(posiciones[e]) for e in posiciones}
    df_media = pd.DataFrame({"equipo": list(media_pos.keys()), "posición_media": list(media_pos.values())}).sort_values("posición_media")
    st.subheader("🏁 Clasificación final media esperada")
    st.dataframe(df_media)

    top8_probs = {e: np.mean([p <= 8 for p in posiciones[e]]) for e in posiciones}
    df_top8 = pd.DataFrame({"equipo": list(top8_probs.keys()), "prob_top8": list(top8_probs.values())}).sort_values("prob_top8", ascending=False)
    st.subheader("🥇 Probabilidad de Top-8")
    st.dataframe(df_top8.style.format({"prob_top8": "{:.2%}"}))

    campeon_probs = df_probs.loc[1] if 1 in df_probs.index else pd.Series()
    st.subheader("📊 Gráfico de probabilidad de campeón")
    st.bar_chart(campeon_probs)
