import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import os

#  CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Meteorología UNL — Estación Galileo",
    layout="wide",
    page_icon="🌡️",
)

# Auto-refresco cada 5 minutos usando meta-refresh HTML
INTERVALO_REFRESCO_SEG = 300  # 5 minutos
st.markdown(
    f'<meta http-equiv="refresh" content="{INTERVALO_REFRESCO_SEG}">',
    unsafe_allow_html=True,
)

# Parámetros del modelo (deben coincidir con neurona_temp_galileo_completo.py)
N_PASOS   = 12    # pasos de predicción (cada uno = 5 min)
HIST_PLOT = 36    # puntos de historial a mostrar (36 × 5 min = 3 h)
PASO_MIN  = 5     # minutos entre mediciones

#  ESTILOS
st.markdown("""
<style>
    /* Fondo oscuro tipo panel científico */
    .stApp { background-color: #0F1117; color: #DDDDDD; }
    .block-container { padding-top: 1rem; }
    /* Métricas */
    [data-testid="metric-container"] {
        background: #1A1D27;
        border: 1px solid #2A2D3A;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { color: #9EAFC2 !important; font-size: 0.82rem; }
    [data-testid="stMetricValue"]  { color: #4FC3F7 !important; font-size: 1.6rem; }
    [data-testid="stMetricDelta"]  { font-size: 0.8rem; }
    /* Cabecera */
    h1 { color: #4FC3F7 !important; }
    h2, h3 { color: #9EAFC2 !important; }
    /* Separador */
    hr { border-color: #2A2D3A; }
    /* Pie de página */
    .footer { color: #555566; font-size: 0.75rem; text-align: center; padding-top: 12px; }
</style>
""", unsafe_allow_html=True)

#  TÍTULO
st.title("🌡️ Predicción de Temperatura — Estación Galileo (UNL)")
st.markdown("Panel de monitoreo en tiempo real. Se actualiza automáticamente cada **5 minutos**.")

#  CARGA DE DATOS
XLSX_FILE = "historial_galileo.xlsx"
PNG_FILE  = "galileo_prediccion_actual.png"

@st.cache_data(ttl=INTERVALO_REFRESCO_SEG)
def cargar_datos():
    try:
        df = pd.read_excel(XLSX_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ Error al leer '{XLSX_FILE}': {e}")
        return pd.DataFrame()

df = cargar_datos()

# Columnas de predicción presentes en el Excel
COLS_PRED = [f"pred_{(i+1)*5}min" for i in range(N_PASOS)]   # pred_5min … pred_60min

if df.empty:
    st.warning("⏳ Esperando a que el modelo genere datos en `historial_galileo.xlsx`…")
    st.stop()

# Descartar filas con timestamp inválido o temp_real_C inválida
df = df.dropna(subset=["timestamp", "temp_real_C"]).reset_index(drop=True)

if df.empty:
    st.warning("⏳ No hay filas con datos válidos todavía. Esperando al modelo…")
    st.stop()

#  SECCIÓN 1: MÉTRICAS ACTUALES
st.subheader("📊 Estado Actual")

# Última fila con temperatura real válida
ultima = df.iloc[-1]

# Última fila con predicciones válidas (puede ser distinta a la última fila)
def ultima_valida(col):
    s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    return s.iloc[-1] if not s.empty else None

temp_real  = float(ultima["temp_real_C"])
pred_5     = ultima_valida("pred_5min")
pred_30    = ultima_valida("pred_30min")
pred_60    = ultima_valida("pred_60min")
rmse_v     = ultima_valida("rmse_ventana")

def fmt_delta(pred, real):
    """Devuelve el delta solo si ambos valores son números válidos."""
    if pred is not None and pd.notna(pred) and pd.notna(real):
        return f"{pred - real:+.1f} °C"
    return None

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("🌡️ Temperatura Real",    f"{temp_real:.1f} °C")
col2.metric("⏱️ Predicción +5 min",   f"{pred_5:.1f} °C"  if pred_5  is not None else "—", delta=fmt_delta(pred_5,  temp_real))
col3.metric("⏱️ Predicción +30 min",  f"{pred_30:.1f} °C" if pred_30 is not None else "—", delta=fmt_delta(pred_30, temp_real))
col4.metric("⏱️ Predicción +60 min",  f"{pred_60:.1f} °C" if pred_60 is not None else "—", delta=fmt_delta(pred_60, temp_real))
col5.metric("📉 RMSE ventana",         f"{rmse_v:.3f} °C"  if rmse_v  is not None else "—")

# Timestamp seguro
ts = ultima["timestamp"]
ts_str = ts.strftime('%d/%m/%Y %H:%M') if pd.notna(ts) else "desconocido"
st.markdown(f"<p style='color:#666677;font-size:0.8rem;'>Última lectura: {ts_str}</p>", unsafe_allow_html=True)
st.markdown("---")

#  HELPER: plantilla oscura para gráficos Plotly
PLOT_LAYOUT = dict(
    paper_bgcolor="#0F1117",
    plot_bgcolor="#1A1D27",
    font=dict(color="#AAAAAA", size=11),
    xaxis=dict(
        gridcolor="#2A2D3A", showgrid=True,
        tickformat="%H:%M", zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#2A2D3A", showgrid=True,
        zeroline=False,
        title="Temperatura (°C)",
    ),
    legend=dict(
        bgcolor="#1A1D27", bordercolor="#2A2D3A",
        borderwidth=1, font=dict(size=10),
    ),
    margin=dict(l=55, r=30, t=55, b=45),
    hovermode="x unified",
)

#  SECCIÓN 2: GRÁFICO DE HORIZONTE DE PREDICCIÓN
#  (réplica interactiva del gráfico matplotlib)
st.subheader("🔮 Horizonte de Predicción con Banda de Incertidumbre")

# Últimas HIST_PLOT filas con datos reales continuos
df_hist = df.tail(HIST_PLOT).copy()

dt_actual = ultima["timestamp"]   # ya validado como no-NaT arriba
df_hist_clean = df_hist.dropna(subset=["timestamp", "temp_real_C"])
t_real    = df_hist_clean["timestamp"].tolist()
v_real    = df_hist_clean["temp_real_C"].tolist()

# Última fila con predicciones válidas
df_con_preds = df.dropna(subset=["pred_5min"]).copy()
rmse_val = float(rmse_v) if rmse_v is not None else 0.5

fig_horizon = go.Figure()

# — Temperatura real —
fig_horizon.add_trace(go.Scatter(
    x=t_real, y=v_real,
    mode="lines+markers",
    name="Temperatura real",
    line=dict(color="#4FC3F7", width=2),
    marker=dict(size=4),
    hovertemplate="%{y:.2f} °C<extra>Real</extra>",
))

# Predicciones desde la última fila con preds y timestamp válido
df_con_preds = df_con_preds.dropna(subset=["timestamp"])
if not df_con_preds.empty:
    fila_pred = df_con_preds.iloc[-1]
    dt_pred_base = fila_pred["timestamp"]

    t_preds = [dt_pred_base + timedelta(minutes=PASO_MIN * (i + 1)) for i in range(N_PASOS)]
    v_preds = []
    for col in COLS_PRED:
        val = fila_pred.get(col, float("nan"))
        v_preds.append(float(val) if pd.notna(val) else None)

    # Banda de incertidumbre que crece con el horizonte
    incert = [rmse_val * (1 + 0.15 * i) for i in range(N_PASOS)]
    v_sup  = [p + u if p is not None else None for p, u in zip(v_preds, incert)]
    v_inf  = [p - u if p is not None else None for p, u in zip(v_preds, incert)]

    # Rellenar banda (fill between)
    fig_horizon.add_trace(go.Scatter(
        x=t_preds + t_preds[::-1],
        y=v_sup + v_inf[::-1],
        fill="toself",
        fillcolor="rgba(255,138,101,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=True,
        name="Incertidumbre (±RMSE·k)",
        hoverinfo="skip",
    ))

    # Línea de predicción
    fig_horizon.add_trace(go.Scatter(
        x=t_preds, y=v_preds,
        mode="lines+markers",
        name="Predicción",
        line=dict(color="#FF8A65", width=2, dash="dash"),
        marker=dict(size=4),
        hovertemplate="%{y:.2f} °C<extra>Predicción</extra>",
    ))

    # Anotación valor actual
    if v_real:
        fig_horizon.add_annotation(
            x=t_real[-1], y=v_real[-1],
            text=f"<b>{v_real[-1]:.1f}°C</b>",
            showarrow=True, arrowhead=0,
            ax=10, ay=-22,
            font=dict(color="#4FC3F7", size=10),
        )
    # Anotación predicción final
    if v_preds[-1] is not None:
        fig_horizon.add_annotation(
            x=t_preds[-1], y=v_preds[-1],
            text=f"<b>{v_preds[-1]:.1f}°C (+{N_PASOS*PASO_MIN} min)</b>",
            showarrow=True, arrowhead=0,
            ax=10, ay=-22,
            font=dict(color="#FF8A65", size=10),
        )

# Línea vertical "Ahora"
fig_horizon.add_vline(
    x=dt_actual.timestamp() * 1000,
    line=dict(color="#FFFFFF", width=1, dash="dot"),
    annotation_text="Ahora",
    annotation_font_color="#AAAAAA",
    annotation_position="top right",
)

fig_horizon.update_layout(
    **PLOT_LAYOUT,
    title=dict(
        text=f"Estación Galileo — {dt_actual.strftime('%d/%m/%Y %H:%M')}  |  "
             f"RMSE = {rmse_val:.3f} °C  |  {len(t_real)} datos reales",
        font=dict(color="#DDDDDD", size=13),
    ),
    height=420,
)

st.plotly_chart(fig_horizon, use_container_width=True)
st.markdown("---")

#  SECCIÓN 3: REAL vs MÚLTIPLES HORIZONTES
st.subheader("📈 Evolución Real vs Predicciones (múltiples horizontes)")

df_plot = df.tail(HIST_PLOT * 3).copy()   # más contexto para esta vista

fig_multi = go.Figure()

# Real
fig_multi.add_trace(go.Scatter(
    x=df_plot["timestamp"], y=df_plot["temp_real_C"],
    mode="lines", name="Real",
    line=dict(color="#4FC3F7", width=2.5),
    hovertemplate="%{y:.2f} °C<extra>Real</extra>",
))

# Horizontes: 5, 15, 30, 60 min con colores degradados
horizontes = [
    ("pred_5min",  "+5 min",  "#66BB6A"),
    ("pred_15min", "+15 min", "#FFA726"),
    ("pred_30min", "+30 min", "#EF5350"),
    ("pred_60min", "+60 min", "#AB47BC"),
]
for col, label, color in horizontes:
    if col in df_plot.columns:
        fig_multi.add_trace(go.Scatter(
            x=df_plot["timestamp"], y=df_plot[col],
            mode="lines", name=label,
            line=dict(color=color, width=1.5, dash="dot"),
            opacity=0.85,
            hovertemplate=f"%{{y:.2f}} °C<extra>{label}</extra>",
        ))

fig_multi.update_layout(
    **PLOT_LAYOUT,
    title=dict(
        text="Temperatura real vs predicciones a distintos horizontes",
        font=dict(color="#DDDDDD", size=13),
    ),
    height=380,
)

st.plotly_chart(fig_multi, use_container_width=True)
st.markdown("---")

#  SECCIÓN 4: MÉTRICAS DE ERROR A LO LARGO DEL TIEMPO
st.subheader("📉 Evolución del Error del Modelo (RMSE y MAE)")

df_err = df.dropna(subset=["rmse_ventana"]).copy()

if not df_err.empty:
    fig_err = go.Figure()

    fig_err.add_trace(go.Scatter(
        x=df_err["timestamp"], y=df_err["rmse_ventana"],
        mode="lines", name="RMSE",
        line=dict(color="#FF8A65", width=2),
        fill="tozeroy", fillcolor="rgba(255,138,101,0.08)",
        hovertemplate="%{y:.4f} °C<extra>RMSE</extra>",
    ))

    if "mae_ventana" in df_err.columns:
        fig_err.add_trace(go.Scatter(
            x=df_err["timestamp"], y=df_err["mae_ventana"],
            mode="lines", name="MAE",
            line=dict(color="#4FC3F7", width=2, dash="dash"),
            hovertemplate="%{y:.4f} °C<extra>MAE</extra>",
        ))

    # Línea de referencia umbral 1°C
    fig_err.add_hline(
        y=1.0, line=dict(color="#EF5350", dash="longdash", width=1),
        annotation_text="Umbral reentrenamiento (1 °C)",
        annotation_font_color="#EF5350",
        annotation_position="bottom right",
    )

    # Marcar re-entrenamientos
    if "reentrenado" in df_err.columns:
        df_retrain = df_err[df_err["reentrenado"] == 1]
        if not df_retrain.empty:
            fig_err.add_trace(go.Scatter(
                x=df_retrain["timestamp"], y=df_retrain["rmse_ventana"],
                mode="markers", name="Re-entrenamiento",
                marker=dict(color="#FFD54F", size=9, symbol="star"),
                hovertemplate="%{x}<br>RMSE: %{y:.4f}<extra>Re-entrenó</extra>",
            ))

    fig_err.update_layout(
        **PLOT_LAYOUT,
        title=dict(
            text="Error del modelo en ventana deslizante",
            font=dict(color="#DDDDDD", size=13),
        ),
        height=320,
    )
    st.plotly_chart(fig_err, use_container_width=True)

st.markdown("---")

#  SECCIÓN 5: MAPA DE CALOR — ERROR POR HORA DEL DÍA
st.subheader("🗓️ Mapa de Calor — Error Absoluto por Hora del Día")

df_heat = df.dropna(subset=["error_C"]).copy()

if len(df_heat) >= 12:
    df_heat["hora"]  = df_heat["timestamp"].dt.hour
    df_heat["fecha"] = df_heat["timestamp"].dt.date
    df_heat["error_abs"] = df_heat["error_C"].abs()

    pivot = df_heat.pivot_table(
        index="fecha", columns="hora", values="error_abs", aggfunc="mean"
    )

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[str(d) for d in pivot.index],
        colorscale="YlOrRd",
        colorbar=dict(title="Error medio (°C)", tickfont=dict(color="#AAAAAA")),
        hovertemplate="Hora: %{x}<br>Fecha: %{y}<br>Error: %{z:.2f} °C<extra></extra>",
    ))
    # PLOT_LAYOUT ya define xaxis/yaxis; se fusiona en un dict para evitar duplicados
    heat_layout = {**PLOT_LAYOUT}
    heat_layout["xaxis"]  = dict(title="Hora del día", gridcolor="#2A2D3A", tickfont=dict(color="#AAAAAA"))
    heat_layout["yaxis"]  = dict(title="Fecha",        gridcolor="#2A2D3A", tickfont=dict(color="#AAAAAA"))
    heat_layout["title"]  = dict(text="Error de predicción promedio por hora del día y fecha", font=dict(color="#DDDDDD", size=13))
    heat_layout["height"] = 320
    fig_heat.update_layout(**heat_layout)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

#  SECCIÓN 6: IMAGEN MATPLOTLIB (si existe)
if os.path.exists(PNG_FILE):
    st.subheader("🖼️ Gráfico en Tiempo Real (generado por el modelo)")
    st.image(PNG_FILE, use_container_width=True)
    st.markdown("---")

#  SECCIÓN 7: TABLA DE DATOS RECIENTES
with st.expander("📋 Ver últimos registros (datos crudos)", expanded=False):
    cols_mostrar = ["timestamp", "temp_real_C", "pred_5min", "pred_30min",
                    "pred_60min", "error_C", "rmse_ventana", "mae_ventana", "reentrenado"]
    cols_presentes = [c for c in cols_mostrar if c in df.columns]
    st.dataframe(
        df[cols_presentes].tail(30).sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

#  PIE DE PÁGINA
st.markdown(
    f"<p class='footer'>Fuente: http://galileo4.unl.edu.ar — "
    f"Modelo: Neurona única | Actualización automática cada {INTERVALO_REFRESCO_SEG // 60} min</p>",
    unsafe_allow_html=True,
)
