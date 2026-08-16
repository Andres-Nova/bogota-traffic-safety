"""Tema visual compartido — Andres Nova · AI Solutions Architect."""
import streamlit as st

PORTFOLIO_URL = "https://andres-nova.github.io"

# ── Paletas de tema ───────────────────────────────────────────────────────────
_TEMAS: dict[bool, dict] = {
    True: {   # oscuro (default)
        "bg":         "#0A0A0A",
        "superficie": "#141414",
        "borde":      "#2A2A2A",
        "texto":      "#EEEEEE",
        "muted":      "#888888",
        "acento":     "#FFFFFF",
        "grid":       "#1E1E1E",
    },
    False: {  # claro
        "bg":         "#FFFFFF",
        "superficie": "#F5F5F5",
        "borde":      "#E0E0E0",
        "texto":      "#111111",
        "muted":      "#666666",
        "acento":     "#111111",
        "grid":       "#EBEBEB",
    },
}


def _paleta() -> dict:
    return _TEMAS[st.session_state.get("tema_oscuro", True)]


def toggle_tema_sidebar() -> None:
    """Renderiza toggle 🌙/☀️ en el sidebar y actualiza session_state."""
    if "tema_oscuro" not in st.session_state:
        st.session_state.tema_oscuro = True  # oscuro por defecto
    seleccion = st.sidebar.radio(
        "Tema",
        ["🌙 Oscuro", "☀️ Claro"],
        index=0 if st.session_state.tema_oscuro else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.tema_oscuro = seleccion.startswith("🌙")


def aplicar_estilo() -> None:
    """Inyecta CSS base + overrides del tema activo."""
    es_oscuro = st.session_state.get("tema_oscuro", True)
    p = _paleta()

    css_base = """
    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    .stDeployButton{display:none!important;}
    .block-container{padding-top:1.2rem!important;}
    section[data-testid="stSidebar"] > div:first-child{padding-top:1rem;}
    """

    css_claro = "" if es_oscuro else f"""
    .stApp{{background-color:{p['bg']}!important;}}
    header[data-testid="stHeader"]{{background-color:{p['bg']}!important;}}
    section[data-testid="stSidebar"]{{background-color:{p['superficie']}!important;}}
    [data-testid="stSidebarContent"]{{background-color:{p['superficie']}!important;}}
    .stMarkdown p,.stMarkdown li,.stText{{color:{p['texto']}!important;}}
    h1,h2,h3,h4,h5,h6,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,
    .stMarkdown h4,.stMarkdown h5{{color:{p['texto']}!important;}}
    [data-testid="stMetricValue"]{{color:{p['texto']}!important;}}
    [data-testid="stMetricLabel"]{{color:{p['muted']}!important;}}
    .stCaption p{{color:{p['muted']}!important;}}
    label,.stRadio label,.stSelectbox label,.stSlider label{{color:{p['muted']}!important;}}
    .streamlit-expanderHeader p{{color:{p['texto']}!important;}}
    """

    st.markdown(f"<style>{css_base}{css_claro}</style>", unsafe_allow_html=True)


def mostrar_header(titulo: str, emoji: str, descripcion: str) -> None:
    """Cabecera monocromática adaptativa con borde izquierdo."""
    aplicar_estilo()
    p = _paleta()

    st.markdown(
        f"""
        <div style="
            background:{p['superficie']};
            border-left:4px solid {p['acento']};
            border-radius:0 8px 8px 0;
            padding:18px 26px;
            margin-bottom:1.4rem;
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            flex-wrap:wrap;
            gap:12px;
        ">
          <div>
            <div style="font-size:.70rem;color:{p['muted']};
                        letter-spacing:.12em;text-transform:uppercase;
                        margin-bottom:5px;font-family:monospace;">
              Andres Nova &nbsp;·&nbsp; AI Solutions Architect
            </div>
            <h1 style="margin:0;font-size:1.5rem;font-weight:700;
                       color:{p['texto']};line-height:1.25;font-family:monospace;">
              {emoji}&nbsp;{titulo}
            </h1>
            <p style="margin:6px 0 0;color:{p['muted']};font-size:.88rem;max-width:520px;">
              {descripcion}
            </p>
          </div>
          <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener" style="
              background:rgba(128,128,128,.12);
              color:{p['texto']};
              text-decoration:none;
              padding:7px 16px;
              border-radius:4px;
              font-size:.80rem;
              font-weight:600;
              border:1px solid {p['borde']};
              white-space:nowrap;
              align-self:flex-start;
              font-family:monospace;
          ">← Portafolio</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def aplicar_tema_fig(fig):
    """Aplica el tema activo a una figura Plotly. Retorna la figura modificada."""
    p = _paleta()
    fig.update_layout(
        paper_bgcolor=p["bg"],
        plot_bgcolor=p["bg"],
        font=dict(color=p["texto"], family="monospace, sans-serif", size=12),
        xaxis=dict(
            gridcolor=p["grid"],
            linecolor=p["borde"],
            tickfont=dict(color=p["muted"]),
            title_font=dict(color=p["texto"]),
            zerolinecolor=p["borde"],
        ),
        yaxis=dict(
            gridcolor=p["grid"],
            linecolor=p["borde"],
            tickfont=dict(color=p["muted"]),
            title_font=dict(color=p["texto"]),
            zerolinecolor=p["borde"],
        ),
        legend=dict(
            font=dict(color=p["texto"]),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=p["borde"],
            borderwidth=1,
        ),
        title_font=dict(color=p["texto"]),
        coloraxis_colorbar=dict(
            tickfont=dict(color=p["texto"]),
            title_font=dict(color=p["texto"]),
        ),
    )
    return fig


def colores_tematicos() -> dict:
    """Retorna dict de colores adaptativos para el tema activo."""
    es_oscuro = st.session_state.get("tema_oscuro", True)
    if es_oscuro:
        return {
            "primario":    "#EEEEEE",
            "secundario":  "#AAAAAA",
            "terciario":   "#666666",
            "cuarto":      "#333333",
            "peligro":     "#EF4444",
            "advertencia": "#F59E0B",
            "exito":       "#22C55E",
            "neutro":      "#888888",
            "info":        "#60A5FA",
        }
    else:
        return {
            "primario":    "#111111",
            "secundario":  "#444444",
            "terciario":   "#777777",
            "cuarto":      "#BBBBBB",
            "peligro":     "#DC2626",
            "advertencia": "#D97706",
            "exito":       "#16A34A",
            "neutro":      "#888888",
            "info":        "#2563EB",
        }


def mostrar_footer() -> None:
    """Pie de página con enlace al portafolio."""
    st.divider()
    st.caption(
        f"Proyecto de portafolio · "
        f"[Andres Nova]({PORTFOLIO_URL}) — AI Solutions Architect"
    )
