# app_streamlit.py
# UI Streamlit – KPIs simplifiés (8 indicateurs) avec robustesse et pression
# VERSION UI/UX AMÉLIORÉE - Design moderne inspiré des outils d'analyse sportive

from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Plotly (avec garde-fou si absent)
PLOTLY_OK = True
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    PLOTLY_OK = False

from calc_iic import (
    list_matches,
    read_match_meta,
    top_players_by_possession,
    compute_iic_for_player,
    player_team_mode_map,   # <— pour corriger les joueurs dans 2 équipes
)

# Optionnel (si tu as ajouté le scan récursif dans calc_iic.py)
try:
    from calc_iic import list_matches_anywhere
    HAS_SCAN = True
except Exception:
    HAS_SCAN = False


# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Football Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Analyse de l'Apport Collectif - 8 Indicateurs Clés"}
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%); }
    h1 {
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2.8rem !important; margin-bottom: 0.3rem; letter-spacing: -0.02em;
    }
    h2 {
        color: #ffffff; font-weight: 700; font-size: 1.9rem !important;
        margin-top: 2.5rem; margin-bottom: 1.5rem; padding-bottom: 0.8rem;
        border-bottom: 3px solid; border-image: linear-gradient(90deg, #00ff87 0%, #60efff 100%) 1;
    }
    h3 { color: #60efff; font-weight: 600; font-size: 1.4rem !important; margin-top: 1rem; }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important; font-weight: 800 !important;
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important; color: #8b92a8 !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%); color: #0f1419; border: none;
        border-radius: 12px; padding: 0.75rem 2.5rem; font-weight: 700; font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 10px 30px rgba(0, 255, 135, 0.3);
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(0, 255, 135, 0.5); }
    .stButton > button:active { transform: translateY(-1px); }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ff6b9d 0%, #c06bff 100%); color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.8rem; font-weight: 600; font-size: 0.9rem; transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.3);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 1px solid rgba(96, 239, 255, 0.1);
    }
    .match-card {
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.08) 0%, rgba(96, 239, 255, 0.08) 100%);
        border-radius: 20px; padding: 2rem; border: 2px solid rgba(96, 239, 255, 0.2);
        margin: 1.5rem 0; backdrop-filter: blur(20px); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    .player-section {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.8) 0%, rgba(15, 20, 25, 0.8) 100%);
        border-radius: 16px; padding: 2rem; margin: 2rem 0; border-left: 5px solid;
        border-image: linear-gradient(180deg, #00ff87 0%, #60efff 100%) 1; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    .info-badge {
        display: inline-block; padding: 0.4rem 1rem;
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.2) 0%, rgba(96, 239, 255, 0.2) 100%);
        border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: #60efff;
        border: 1px solid rgba(96, 239, 255, 0.3); margin: 0.3rem;
    }
    .pressure-indicator {
        padding: 1rem; background: linear-gradient(135deg, rgba(255, 107, 157, 0.1) 0%, rgba(192, 107, 255, 0.1) 100%);
        border-left: 4px solid #ff6b9d; border-radius: 8px; margin-top: 1rem; font-size: 0.95rem; color: #ffb3d9;
    }
</style>
""", unsafe_allow_html=True)

# ========== UTILS ==========
def safe_round(x, n=1):
    if x is None:
        return None
    try:
        return round(float(x), n)
    except Exception:
        return None

def fmt_pct(x, n=1, signed=False):
    v = safe_round(x, n)
    if v is None:
        return None
    return f"{v:+.{n}f}%" if signed else f"{v:.{n}f}%"

def fmt_rate(x, n=1):
    # x attendu entre 0 et 1
    if x is None:
        return None
    try:
        return f"{round(100*float(x), n):.{n}f}%"
    except Exception:
        return None

def get_summary_value(summary: dict, preferred_key: str, fallback_keys: list):
    """Renvoie summary[preferred_key] si présent, sinon essaye les fallback_keys."""
    if summary is None:
        return None
    if preferred_key in summary and summary.get(preferred_key) is not None:
        return summary.get(preferred_key)
    for k in fallback_keys:
        if k in summary and summary.get(k) is not None:
            return summary.get(k)
    return None

def run_compute(events_csv, tracking_jsonl, player_id,
                pre_s, post_s_possession, post_s_struct, fps,
                press_t1_s, press_t2_s, press_r1_m=None, press_r2_m=None):
    """Wrapper pour compute_iic_for_player avec compatibilité d’anciennes signatures."""
    try:
        # Nouveau backend (idéal): supporte press_t1_s / press_t2_s + clés neutres
        return compute_iic_for_player(
            events_csv=events_csv,
            tracking_jsonl=tracking_jsonl,
            player_id=player_id,
            pre_s=pre_s,
            post_s_possession=post_s_possession,
            post_s_struct=post_s_struct,
            fps=fps,
            press_t1_s=press_t1_s,
            press_t2_s=press_t2_s,
            press_r1_m=press_r1_m,
            press_r2_m=press_r2_m,
        )
    except TypeError:
        try:
            # Backend intermédiaire: pas press_t1_s/press_t2_s, mais press_r1/r2 + clés anciennes
            return compute_iic_for_player(
                events_csv=events_csv,
                tracking_jsonl=tracking_jsonl,
                player_id=player_id,
                pre_s=pre_s,
                post_s_possession=post_s_possession,
                post_s_struct=post_s_struct,
                fps=fps,
                press_r1_m=press_r1_m,
                press_r2_m=press_r2_m,
            )
        except TypeError:
            # Très ancien backend
            return compute_iic_for_player(events_csv, tracking_jsonl, player_id, pre_s, post_s_possession, post_s_struct, fps)

def create_radar_chart(summary, player_name, team_name,
                       post_s_possession, post_s_struct):
    """Crée un graphique radar pour les KPI."""
    if not PLOTLY_OK:
        return None

    categories = [
        f"Possession<br>+{int(post_s_possession)}s",
        f"Δ Largeur<br>+{int(post_s_struct)}s",
        f"Δ Hauteur<br>+{int(post_s_struct)}s",
        f"Δ Compacité<br>+{int(post_s_struct)}s",
        "Δ Vitesse"
    ]

    # Compat keys: neutres OU anciennes
    poss = get_summary_value(summary, "possession_retained_rate", ["possession_retained_rate_+6s"])
    w = get_summary_value(summary, "delta_width_pct_mean", ["delta_width_pct_+5s_mean"])
    h = get_summary_value(summary, "delta_height_pct_mean", ["delta_height_pct_+5s_mean"])
    c = get_summary_value(summary, "delta_compact_pct_mean", ["delta_compact_pct_+5s_mean"])
    v = get_summary_value(summary, "delta_team_speed_pct", ["delta_team_speed_pct_0_6_vs_-3_0"])

    values = [
        (poss * 100) if poss is not None else 50,
        min(max(((w or 0) + 20) * 2.5, 0), 100) if w is not None else 50,
        min(max(((h or 0) + 20) * 2.5, 0), 100) if h is not None else 50,
        min(max(((c or 0) + 20) * 2.5, 0), 100) if c is not None else 50,
        min(max(((v or 0) + 30) * 1.5, 0), 100) if v is not None else 50,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_name,
        line=dict(color='#00ff87', width=3),
        fillcolor='rgba(0, 255, 135, 0.3)',
        marker=dict(size=8, color='#60efff')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(96, 239, 255, 0.2)', color='#8b92a8'),
            angularaxis=dict(gridcolor='rgba(96, 239, 255, 0.2)', color='#60efff'),
            bgcolor='rgba(15, 20, 25, 0.5)'
        ),
        showlegend=False,
        title=dict(text=f"<b>{player_name}</b> · {team_name}", font=dict(color='#60efff', size=18), x=0.5, xanchor='center'),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=450,
        margin=dict(t=100, b=60, l=60, r=60)
    )
    return fig

def create_comparison_chart(summaries, player_names,
                            post_s_possession, post_s_struct,
                            press_t1_s, press_t2_s, press_r1_m, press_r2_m):
    """Graphique comparatif (si Plotly dispo)."""
    if not PLOTLY_OK:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"⚽ Possession (+{int(post_s_possession)}s)",
            f"📐 Structure (+{int(post_s_struct)}s)",
            "⚡ Tempo d'équipe",
            f"🎯 Pression (+{int(press_t1_s)}s / +{int(press_t2_s)}s)"
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.15, horizontal_spacing=0.12
    )

    # Possession
    poss_vals = []
    for s in summaries:
        poss = get_summary_value(s, "possession_retained_rate", ["possession_retained_rate_+6s"])
        poss_vals.append((poss * 100) if poss is not None else 0)
    fig.add_trace(go.Bar(x=player_names, y=poss_vals, showlegend=False,
                         text=[f'{v:.1f}%' for v in poss_vals], textposition='outside'),
                  row=1, col=1)

    # Structure = moyenne (largeur + hauteur + compacité)
    struct_vals = []
    for s in summaries:
        w = get_summary_value(s, "delta_width_pct_mean", ["delta_width_pct_+5s_mean"]) or 0
        h = get_summary_value(s, "delta_height_pct_mean", ["delta_height_pct_+5s_mean"]) or 0
        c = get_summary_value(s, "delta_compact_pct_mean", ["delta_compact_pct_+5s_mean"]) or 0
        struct_vals.append((w + h + c) / 3.0)
    fig.add_trace(go.Bar(x=player_names, y=struct_vals, showlegend=False,
                         text=[f'{v:+.1f}%' for v in struct_vals], textposition='outside'),
                  row=1, col=2)

    # Tempo
    tempo_vals = []
    for s in summaries:
        v = get_summary_value(s, "delta_team_speed_pct", ["delta_team_speed_pct_0_6_vs_-3_0"])
        tempo_vals.append(v if v is not None else 0)
    fig.add_trace(go.Bar(x=player_names, y=tempo_vals, showlegend=False,
                         text=[f'{v:+.1f}%' for v in tempo_vals], textposition='outside'),
                  row=2, col=1)

    # Pression = moyenne de deux instants
    press_vals = []
    for s in summaries:
        # new keys
        p1 = get_summary_value(s, "pressure_n_r1_mean", [])  # entier
        p2 = get_summary_value(s, "pressure_n_r2_mean", [])  # entier

        # old keys fallback
        if p1 is None:
            p1 = get_summary_value(s, "n_opponents_within_3m_at_+3s_mean", [])
        if p2 is None:
            p2 = get_summary_value(s, "n_opponents_within_5m_at_+5s_mean", [])

        p1 = 0 if p1 is None else p1
        p2 = 0 if p2 is None else p2
        press_vals.append((p1 + p2) / 2.0)

    fig.add_trace(go.Bar(x=player_names, y=press_vals, showlegend=False,
                         text=[f'{v:.1f}' for v in press_vals], textposition='outside'),
                  row=2, col=2)

    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(gridcolor='rgba(96, 239, 255, 0.1)', showgrid=True)
    fig.update_layout(
        height=650,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(15, 20, 25, 0.5)',
        font=dict(color='#8b92a8'),
        title=dict(text="<b>Comparaison Multi-Joueurs</b>", x=0.5, xanchor='center'),
        margin=dict(t=100, b=80, l=60, r=60)
    )
    return fig


# ========== HEADER ==========
st.title("⚽ Football Analytics Dashboard")
st.markdown('<p class="subtitle">Analyse Avancée de l\'Apport Collectif • 8 Indicateurs Clés de Performance</p>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # --- Chargement dossiers externes ---
    if HAS_SCAN:
        mode_scan = st.checkbox("📌 Scanner récursivement (dossiers externes)", value=True)
        base_matches_dir = st.text_input("📂 Dossier racine", "data")
        st.caption("Mode scan : le dossier peut contenir des sous-dossiers, on détecte les matchs automatiquement.")
    else:
        mode_scan = False
        base_matches_dir = st.text_input("📂 Dossier des matchs", "data/matches")
        st.caption("Mode strict : structure attendue data/matches/<match_id>/...")

    st.markdown("### ⏱️ Paramètres temporels")
    pre_s = st.number_input("Fenêtre AVANT (s)", min_value=0.0, value=1.0, step=0.5,
                            help="Référence t0−1s pour Δ structure")

    post_s_struct = st.number_input("Horizon structure (+s)", min_value=1.0, value=5.0, step=0.5)
    post_s_possession = st.number_input("Horizon possession (+s)", min_value=1.0, value=6.0, step=0.5)

    fps = st.number_input("FPS (tracking)", min_value=1, value=10, step=1)

    st.markdown("### 🎯 Paramètres de pression")
    # temps pression (optionnel si le backend les supporte)
    press_t1_s = st.number_input("Temps pression 1 (s)", min_value=0.5, value=3.0, step=0.5)
    press_r1_m = st.number_input(f"Rayon pression 1 (m) @+{int(press_t1_s)}s", min_value=2.0, value=3.0, step=0.5)

    press_t2_s = st.number_input("Temps pression 2 (s)", min_value=0.5, value=5.0, step=0.5)
    press_r2_m = st.number_input(f"Rayon pression 2 (m) @+{int(press_t2_s)}s", min_value=3.0, value=5.0, step=0.5)

st.markdown("---")

# ========== MAIN FLOW ==========
if st.button("🔄 CHARGER LES MATCHS", use_container_width=True):
    with st.spinner("⚡ Chargement en cours..."):
        if HAS_SCAN and mode_scan:
            st.session_state["matches_list"] = list_matches_anywhere(base_matches_dir)
        else:
            st.session_state["matches_list"] = list_matches(base_matches_dir)
    st.success("✅ Matchs chargés avec succès!")

matches_list = st.session_state.get("matches_list", None)

if matches_list:
    labels = [lab for _, _, lab in matches_list]
    choice = st.selectbox("🎯 Sélectionner un match", labels, index=0)
    mid, mpath, _ = next((i, p, l) for (i, p, l) in matches_list if l == choice)

    home, away, date = read_match_meta(mpath)
    home = home or "Home"
    away = away or "Away"

    # Match card
    st.markdown(f"""
    <div class="match-card">
        <h2 style="margin-top: 0; border: none; font-size: 2rem;">🏟️ {home} <span style="color: #60efff;">vs</span> {away}</h2>
        <div style="margin-top: 1rem;">
            <span class="info-badge">📅 {date if date else 'Date inconnue'}</span>
            <span class="info-badge">🆔 {mid}</span>
        </div>
        <p style="color: #5a6173; font-size: 0.85rem; margin-top: 1.5rem; margin-bottom: 0;">📁 {mpath}</p>
    </div>
    """, unsafe_allow_html=True)

    # Check files
    match_dir = Path(mpath)
    events_csv = next(match_dir.glob("*_dynamic_events.csv"), None)
    tracking_jsonl = next(match_dir.glob("*_tracking_extrapolated.jsonl"), None)
    if not (events_csv and tracking_jsonl):
        st.error("⚠️ Fichiers du match incomplets"); st.stop()

    # -------- Players selection (corrigé : équipe majoritaire + déduplication) --------
    st.markdown("## 👥 Sélection des joueurs")
    st.markdown("Choisissez jusqu'à 2 joueurs par équipe pour l'analyse")

    events_df = pd.read_csv(str(events_csv), low_memory=False)
    p2t_map = player_team_mode_map(events_df)  # player_id -> team_shortname modalité majoritaire

    tops_raw = top_players_by_possession(str(events_csv), top_n=16)  # plus large, puis filtre
    tops = tops_raw.copy()
    tops["team_mode"] = tops["player_in_possession_id"].map(p2t_map)
    tops = tops[~tops["team_mode"].isna()].copy()
    tops["team_shortname"] = tops["team_mode"]
    tops = tops.drop_duplicates(subset=["player_in_possession_id"])

    teams = list(tops["team_shortname"].dropna().unique())
    if len(teams) > 2:
        counts = tops["team_shortname"].value_counts().index.tolist()
        teams = counts[:2]

    if not teams:
        st.warning("Aucune équipe détectée dans les events."); st.stop()

    cols = st.columns(min(2, len(teams)))
    selections = {}
    for idx, team in enumerate(teams):
        with cols[idx]:
            st.markdown(f"### {team}")
            tdf = (tops[tops["team_shortname"] == team]
                   .sort_values("n_events", ascending=False)
                   .head(8)
                   .reset_index(drop=True))
            picks = []
            for _, r in tdf.iterrows():
                pid = int(r["player_in_possession_id"])
                nm = r["player_in_possession_name"]
                ne = int(r["n_events"])
                label = f"**{nm}** · `{pid}` · *{ne} touches*"
                if st.checkbox(label, value=(len(picks) < 2), key=f"pick_{mid}_{team}_{pid}"):
                    picks.append((pid, nm, team))
            selections[team] = picks

    st.markdown("---")

    # -------- Calculate KPIs --------
    if st.button("📊 CALCULER LES KPI", use_container_width=True):
        all_summaries, all_names = [], []

        for team, pinfo in selections.items():
            if not pinfo:
                continue
            st.markdown(f"## 📈 Résultats – {team}")

            for pid, pname, tname in pinfo:
                with st.spinner(f"⚡ Analyse de {pname}..."):
                    df_out, summary = run_compute(
                        str(events_csv), str(tracking_jsonl),
                        player_id=pid,
                        pre_s=float(pre_s),
                        post_s_possession=float(post_s_possession),
                        post_s_struct=float(post_s_struct),
                        fps=int(fps),
                        press_t1_s=float(press_t1_s),
                        press_t2_s=float(press_t2_s),
                        press_r1_m=float(press_r1_m),
                        press_r2_m=float(press_r2_m),
                    )

                if not summary:
                    st.warning(f"❌ Aucune touche pour {pname}")
                    continue

                all_summaries.append(summary)
                all_names.append(f"{pname} ({tname})")

                # ---- Valeurs KPI (compatibles vieux/nouveau back) ----
                poss = get_summary_value(summary, "possession_retained_rate", ["possession_retained_rate_+6s"])
                w = get_summary_value(summary, "delta_width_pct_mean", ["delta_width_pct_+5s_mean"])
                h = get_summary_value(summary, "delta_height_pct_mean", ["delta_height_pct_+5s_mean"])
                c = get_summary_value(summary, "delta_compact_pct_mean", ["delta_compact_pct_+5s_mean"])
                v = get_summary_value(summary, "delta_team_speed_pct", ["delta_team_speed_pct_0_6_vs_-3_0"])

                # Pression : nouveau back => pressure_n_r1_mean / pressure_n_r2_mean
                p1 = get_summary_value(summary, "pressure_n_r1_mean", [])
                p2 = get_summary_value(summary, "pressure_n_r2_mean", [])
                # fallback back ancien
                if p1 is None:
                    p1 = get_summary_value(summary, "n_opponents_within_3m_at_+3s_mean", [])
                if p2 is None:
                    p2 = get_summary_value(summary, "n_opponents_within_5m_at_+5s_mean", [])

                st.markdown(f'<div class="player-section">', unsafe_allow_html=True)
                st.markdown(f"### {pname} — {tname}")

                # Métriques - Ligne 1
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎯 Touches", summary.get("n_touches"))
                c2.metric(f"⚽ Possession (+{int(post_s_possession)}s)", fmt_rate(poss, 1) or "—")
                c3.metric(f"↔️ Δ Largeur (+{int(post_s_struct)}s)", fmt_pct(w, 1, signed=True) or "—")
                c4.metric(f"↕️ Δ Hauteur (+{int(post_s_struct)}s)", fmt_pct(h, 1, signed=True) or "—")

                # Métriques - Ligne 2
                c5, c6, c7, c8 = st.columns(4)
                c5.metric(f"🎯 Δ Compacité (+{int(post_s_struct)}s)", fmt_pct(c, 1, signed=True) or "—")
                c6.metric("⚡ Δ Vitesse collective", fmt_pct(v, 1, signed=True) or "—")
                c7.metric(f"🔴 Pression @ +{int(press_t1_s)}s (≤{press_r1_m}m)", str(int(p1)) if p1 is not None else "—")
                c8.metric(f"🟠 Pression @ +{int(press_t2_s)}s (≤{press_r2_m}m)", str(int(p2)) if p2 is not None else "—")

                # Indicateur de pression (part des touches sous pression) — compat colonnes
                if not df_out.empty:
                    under1 = None
                    under2 = None

                    # Nouveau back (colonnes neutres)
                    if "pressure_n_r1" in df_out.columns:
                        s = df_out["pressure_n_r1"].dropna()
                        if not s.empty:
                            under1 = (s > 0).mean()
                    if "pressure_n_r2" in df_out.columns:
                        s = df_out["pressure_n_r2"].dropna()
                        if not s.empty:
                            under2 = (s > 0).mean()

                    # Back ancien (colonnes avec +3/+5)
                    if under1 is None and "n_opponents_within_3m_at_+3s" in df_out.columns:
                        s = df_out["n_opponents_within_3m_at_+3s"].dropna()
                        if not s.empty:
                            under1 = (s > 0).mean()
                    if under2 is None and "n_opponents_within_5m_at_+5s" in df_out.columns:
                        s = df_out["n_opponents_within_5m_at_+5s"].dropna()
                        if not s.empty:
                            under2 = (s > 0).mean()

                    parts = []
                    if under1 is not None:
                        parts.append(f"{round(100*under1)}% à +{int(press_t1_s)}s")
                    if under2 is not None:
                        parts.append(f"{round(100*under2)}% à +{int(press_t2_s)}s")
                    if parts:
                        st.markdown(
                            f'<div class="pressure-indicator">🎯 <strong>Sous pression (≥1 adversaire)</strong> : {" • ".join(parts)}</div>',
                            unsafe_allow_html=True
                        )

                # Radar chart
                st.markdown("#### 📊 Profil de performance")
                if PLOTLY_OK:
                    radar = create_radar_chart(summary, pname, tname, post_s_possession, post_s_struct)
                    st.plotly_chart(radar, use_container_width=True)
                else:
                    st.info("📦 Installe plotly pour voir les graphiques : `pip install plotly`")

                # Export buttons (keys uniques)
                st.markdown("#### 💾 Export des données")
                colA, colB = st.columns(2)
                with colA:
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Télécharger Détails (CSV)",
                        data=csv_bytes,
                        file_name=f"KPI_details_{mid}_{pid}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{mid}_{pid}"
                    )
                with colB:
                    jbytes = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
                    st.download_button(
                        "📥 Télécharger Résumé (JSON)",
                        data=jbytes,
                        file_name=f"KPI_summary_{mid}_{pid}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl_json_{mid}_{pid}"
                    )

                with st.expander("📋 Voir les données brutes (DataFrame complet)"):
                    st.dataframe(df_out, use_container_width=True, height=400)

                st.markdown('</div>', unsafe_allow_html=True)

        # Comparison chart
        if len(all_summaries) >= 2:
            st.markdown("---")
            st.markdown("## 🔄 Comparaison multi-joueurs")
            st.markdown("Vue d'ensemble comparative des profils de performance")
            if PLOTLY_OK:
                comparison = create_comparison_chart(
                    all_summaries, all_names,
                    post_s_possession, post_s_struct,
                    press_t1_s, press_t2_s, press_r1_m, press_r2_m
                )
                st.plotly_chart(comparison, use_container_width=True)
            else:
                st.info("📦 Installe plotly pour voir les graphiques : `pip install plotly`")

        # Méthodologie
        st.markdown("---")
        with st.expander("📖 Méthodologie & définitions (8 KPI)"):
            st.markdown(f"""
            <div style="background: rgba(26, 31, 46, 0.5); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(96, 239, 255, 0.2);">
            <strong>Fenêtres :</strong> Δ structure t0−{pre_s}s → t0+{post_s_struct}s • Possession à t0+{post_s_possession}s<br><br>
            <strong>Structure :</strong> ↔️ largeur (Y) • ↕️ hauteur (X) • 🎯 compacité (distance au centre de gravité) — en % d'évolution<br>
            <strong>Tempo :</strong> ⚡ Δ vitesse collective (baseline −3→0s vs 0→6s) — en % d'évolution<br>
            <strong>Pression :</strong> 🔴 +{int(press_t1_s)}s (≤{press_r1_m}m) • 🟠 +{int(press_t2_s)}s (≤{press_r2_m}m) — nombre d’adversaires autour du porteur
            </div>
            """, unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">⚽</div>
        <h2 style="border: none; margin-bottom: 1rem;">Bienvenue sur le Dashboard d'Analyse</h2>
        <p style="font-size: 1.1rem; color: #8b92a8; max-width: 600px; margin: 0 auto 2rem;">
            Cliquez sur <strong style="color: #00ff87;">"CHARGER LES MATCHS"</strong> pour commencer
            l'analyse approfondie de l'apport collectif de vos joueurs.
        </p>
        <p style="color:#8b92a8;">
            Astuce : si tu veux charger des matchs depuis un autre dossier, active le mode "Scanner récursivement" (si dispo).
        </p>
    </div>
    """, unsafe_allow_html=True)
