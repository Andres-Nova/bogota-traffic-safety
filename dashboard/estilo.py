"""Tema visual compartido — Andres Nova · Data Architect & AI Engineer."""
import streamlit as st

PORTFOLIO_URL = "https://andres-nova.github.io"

_CSS = """
<style>
/* ── Ocultar chrome de Streamlit ───────────────── */
#MainMenu          { visibility: hidden; }
footer             { visibility: hidden; }
.stDeployButton    { display: none !important; }

/* ── Reducir padding superior excesivo ─────────── */
.block-container   { padding-top: 1.2rem !important; }

/* ── Sidebar: quitar margen superior extra ──────── */
section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
</style>
"""

def aplicar_estilo() -> None:
    """Inyecta el CSS base. Llamar al inicio de cada página."""
    st.markdown(_CSS, unsafe_allow_html=True)


def mostrar_header(titulo: str, emoji: str, descripcion: str) -> None:
    """Cabecera de marca con gradiente, título y enlace al portafolio."""
    aplicar_estilo()
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
            border-radius:12px; padding:20px 28px; margin-bottom:1.4rem;
            color:white; display:flex; justify-content:space-between;
            align-items:flex-start; flex-wrap:wrap; gap:12px;">
          <div>
            <div style="font-size:.72rem;opacity:.8;letter-spacing:.08em;
                        text-transform:uppercase;margin-bottom:6px;">
              Andres Nova &nbsp;·&nbsp; Data Architect &amp; AI Engineer
            </div>
            <h1 style="margin:0;font-size:1.55rem;font-weight:700;
                       color:white;line-height:1.25;">
              {emoji}&nbsp;{titulo}
            </h1>
            <p style="margin:7px 0 0;opacity:.88;font-size:.9rem;max-width:520px;">
              {descripcion}
            </p>
          </div>
          <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener" style="
              background:rgba(255,255,255,.15);color:white;text-decoration:none;
              padding:8px 18px;border-radius:20px;font-size:.82rem;font-weight:600;
              border:1px solid rgba(255,255,255,.35);white-space:nowrap;
              align-self:flex-start;">← Portafolio</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_footer() -> None:
    """Pie de página consistente con enlace al portafolio."""
    st.divider()
    st.caption(
        f"Proyecto de portafolio · "
        f"[Andres Nova]({PORTFOLIO_URL}) — Data Architect & AI Engineer"
    )
