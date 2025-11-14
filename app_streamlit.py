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
def run_compute(events_csv, tracking_jsonl, player_id,
                pre_s, post_s_possession, post_s_struct, fps,
                press_r1_m=None, press_r2_m=None):
    """Wrapper pour compute_iic_for_player avec compatibilité d’anciennes signatures."""
    try:
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
        # fallback anciennes signatures
        return compute_iic_for_player(events_csv, tracking_jsonl, player_id, pre_s, post_s_possession, post_s_struct, fps)

def create_radar_chart(summary, player_name, team_name):
    """Crée un graphique radar moderne pour les KPI d'un joueur."""
    categories = ['Possession<br>Conservée', 'Δ Largeur', 'Δ Hauteur', 'Δ Compacité', 'Δ Vitesse']
    values = [
        summary.get('possession_retained_rate_+6s', 0) * 100 if summary.get('possession_retained_rate_+6s') else 50,
        min(max((summary.get('delta_width_pct_+5s_mean', 0) + 20) * 2.5, 0), 100) if summary.get('delta_width_pct_+5s_mean') is not None else 50,
        min(max((summary.get('delta_height_pct_+5s_mean', 0) + 20) * 2.5, 0), 100) if summary.get('delta_height_pct_+5s_mean') is not None else 50,
        min(max((summary.get('delta_compact_pct_+5s_mean', 0) + 20) * 2.5, 0), 100) if summary.get('delta_compact_pct_+5s_mean') is not None else 50,
        min(max((summary.get('delta_team_speed_pct_0_6_vs_-3_0', 0) + 30) * 1.5, 0), 100) if summary.get('delta_team_speed_pct_0_6_vs_-3_0') is not None else 50,
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name=player_name,
        line=dict(color='#00ff87', width=3), fillcolor='rgba(0, 255, 135, 0.3)',
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
        paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)', height=450, margin=dict(t=100, b=60, l=60, r=60)
    )
    return fig

def create_comparison_chart(summaries, player_names):
    """Crée un graphique de comparaison élégant entre joueurs."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('⚽ Possession Conservée', '📐 Structure Collective', '⚡ Tempo d\'Équipe', '🎯 Pression Subie'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],[{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.15, horizontal_spacing=0.12
    )
    colors = ['#00ff87', '#60efff', '#ff6b9d', '#c06bff']

    # Possession
    possession_vals = [s.get('possession_retained_rate_+6s', 0) * 100 if s.get('possession_retained_rate_+6s') else 0 for s in summaries]
    fig.add_trace(go.Bar(x=player_names, y=possession_vals, marker=dict(color=colors[0]), showlegend=False,
                         text=[f'{v:.1f}%' for v in possession_vals], textposition='outside'), row=1, col=1)

    # Structure (moyenne des 3 deltas)
    struct_vals = []
    for s in summaries:
        w = s.get('delta_width_pct_+5s_mean', 0) if s.get('delta_width_pct_+5s_mean') is not None else 0
        h = s.get('delta_height_pct_+5s_mean', 0) if s.get('delta_height_pct_+5s_mean') is not None else 0
        c = s.get('delta_compact_pct_+5s_mean', 0) if s.get('delta_compact_pct_+5s_mean') is not None else 0
        struct_vals.append((w + h + c) / 3)
    fig.add_trace(go.Bar(x=player_names, y=struct_vals, marker=dict(color=colors[1]), showlegend=False,
                         text=[f'{v:+.1f}%' for v in struct_vals], textposition='outside'), row=1, col=2)

    # Tempo
    tempo_vals = [s.get('delta_team_speed_pct_0_6_vs_-3_0', 0) if s.get('delta_team_speed_pct_0_6_vs_-3_0') is not None else 0 for s in summaries]
    fig.add_trace(go.Bar(x=player_names, y=tempo_vals, marker=dict(color=colors[2]), showlegend=False,
                         text=[f'{v:+.1f}%' for v in tempo_vals], textposition='outside'), row=2, col=1)

    # Pression
    press_vals = []
    for s in summaries:
        p3 = s.get('n_opponents_within_3m_at_+3s_mean', 0) if s.get('n_opponents_within_3m_at_+3s_mean') is not None else 0
        p5 = s.get('n_opponents_within_5m_at_+5s_mean', 0) if s.get('n_opponents_within_5m_at_+5s_mean') is not None else 0
        press_vals.append((p3 + p5) / 2)
    fig.add_trace(go.Bar(x=player_names, y=press_vals, marker=dict(color=colors[3]), showlegend=False,
                         text=[f'{v:.1f}' for v in press_vals], textposition='outside'), row=2, col=2)

    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(gridcolor='rgba(96, 239, 255, 0.1)', showgrid=True)
    fig.update_layout(height=650, paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(15, 20, 25, 0.5)',
                      font=dict(color='#8b92a8'), title=dict(text="<b>Comparaison Multi-Joueurs</b>", x=0.5, xanchor='center'),
                      margin=dict(t=100, b=80, l=60, r=60))
    return fig

# ========== HEADER ==========
st.title("⚽ Football Analytics Dashboard")
st.markdown('<p class="subtitle">Analyse Avancée de l\'Apport Collectif • 8 Indicateurs Clés de Performance</p>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    base_matches_dir = st.text_input("📂 Dossier des matchs", "data/matches")
    st.markdown("### ⏱️ Paramètres Temporels")
    pre_s = st.number_input("Fenêtre AVANT (s)", min_value=0.0, value=1.0, step=0.5,
                            help="Référence t0−1s pour Δ structure")
    post_s_struct = st.number_input("Horizon structure (+s)", min_value=1.0, value=5.0, step=0.5)
    post_s_possession = st.number_input("Horizon possession (+s)", min_value=1.0, value=6.0, step=0.5)
    fps = st.number_input("FPS (tracking)", min_value=1, value=10, step=1)
    st.markdown("### 🎯 Paramètres de Pression")
    press_r1_m = st.number_input("Rayon immédiat (m) @+3s", min_value=2.0, value=3.0, step=0.5)
    press_r2_m = st.number_input("Rayon soutenu (m) @+5s", min_value=3.0, value=5.0, step=0.5)

st.markdown("---")

# ========== MAIN FLOW ==========
if st.button("🔄 CHARGER LES MATCHS", use_container_width=True):
    with st.spinner("⚡ Chargement en cours..."):
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
    st.markdown("## 👥 Sélection des Joueurs")
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
                if st.checkbox(label, value=(len(picks) < 2), key=f"{team}_{pid}"):
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
                        press_r1_m=float(press_r1_m),
                        press_r2_m=float(press_r2_m),
                    )
                if not summary:
                    st.warning(f"❌ Aucune touche pour {pname}")
                    continue

                all_summaries.append(summary)
                all_names.append(f"{pname} ({tname})")

                st.markdown(f'<div class="player-section">', unsafe_allow_html=True)
                st.markdown(f"### {pname} — {tname}")

                # Métriques - Ligne 1
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎯 Touches", summary.get("n_touches"))
                c2.metric(f"⚽ Possession (+{int(post_s_possession)}s)",
                          f"{round(100*summary.get('possession_retained_rate_+6s',0),1)}%")
                c3.metric(f"↔️ Δ Largeur (+{int(post_s_struct)}s)",
                          None if summary.get("delta_width_pct_+5s_mean") is None else f"{round(summary.get('delta_width_pct_+5s_mean'),1)}%")
                c4.metric(f"↕️ Δ Hauteur (+{int(post_s_struct)}s)",
                          None if summary.get("delta_height_pct_+5s_mean") is None else f"{round(summary.get('delta_height_pct_+5s_mean'),1)}%")

                # Métriques - Ligne 2
                c5, c6, c7, c8 = st.columns(4)
                c5.metric(f"🎯 Δ Compacité (+{int(post_s_struct)}s)",
                          None if summary.get("delta_compact_pct_+5s_mean") is None else f"{round(summary.get('delta_compact_pct_+5s_mean'),1)}%")
                c6.metric("⚡ Δ Vitesse Collective",
                          None if summary.get("delta_team_speed_pct_0_6_vs_-3_0") is None else f"{round(summary.get('delta_team_speed_pct_0_6_vs_-3_0'),1)}%")
                c7.metric(f"🔴 Pression @ +3s",
                          None if summary.get("n_opponents_within_3m_at_+3s_mean") is None else f"{summary.get('n_opponents_within_3m_at_+3s_mean')} adv.")
                c8.metric(f"🟠 Pression @ +5s",
                          None if summary.get("n_opponents_within_5m_at_+5s_mean") is None else f"{summary.get('n_opponents_within_5m_at_+5s_mean')} adv.")

                # Indicateur de pression (part des touches sous pression)
                if not df_out.empty:
                    under3_share = None
                    under5_share = None
                    if "n_opponents_within_3m_at_+3s" in df_out.columns:
                        s = df_out["n_opponents_within_3m_at_+3s"].dropna()
                        if not s.empty: under3_share = (s > 0).mean()
                    if "n_opponents_within_5m_at_+5s" in df_out.columns:
                        s = df_out["n_opponents_within_5m_at_+5s"].dropna()
                        if not s.empty: under5_share = (s > 0).mean()
                    if under3_share is not None or under5_share is not None:
                        parts = []
                        if under3_share is not None: parts.append(f"{round(100*under3_share)}% à +3s")
                        if under5_share is not None: parts.append(f"{round(100*under5_share)}% à +5s")
                        st.markdown(f'<div class="pressure-indicator">🎯 <strong>Sous pression (≥1 adversaire)</strong> : {" • ".join(parts)}</div>', unsafe_allow_html=True)

                # Radar chart
                st.markdown("#### 📊 Profil de Performance")
                if PLOTLY_OK:
                    radar = create_radar_chart(summary, pname, tname)
                    st.plotly_chart(radar, use_container_width=True)
                else:
                    st.info("📦 Installe plotly pour voir les graphiques : `pip install plotly`")

                # Export buttons (keys uniques)
                st.markdown("#### 💾 Export des Données")
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
            st.markdown("## 🔄 Comparaison Multi-Joueurs")
            st.markdown("Vue d'ensemble comparative des profils de performance")
            if PLOTLY_OK:
                comparison = create_comparison_chart(all_summaries, all_names)
                st.plotly_chart(comparison, use_container_width=True)
            else:
                st.info("📦 Installe plotly pour voir les graphiques : `pip install plotly`")

        # Méthodologie
        st.markdown("---")
        with st.expander("📖 Méthodologie & Définitions des Indicateurs"):
            st.markdown(f"""
            <div style="background: rgba(26, 31, 46, 0.5); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(96, 239, 255, 0.2);">
            <strong>Fenêtres :</strong> Δ structure t0−{pre_s}s → t0+{post_s_struct}s • Possession à t0+{post_s_possession}s<br>
            ↔️ Δ Largeur (Y) • ↕️ Δ Hauteur (X) • 🎯 Δ Compacité (distance au centre de gravité)<br>
            ⚡ Δ Vitesse collective : 0→6s vs −3→0s (fallback centroïde si <4 joueurs valides)<br>
            🔴 +3s (≤{press_r1_m} m) • 🟠 +5s (≤{press_r2_m} m) — nombre d’adversaires autour du porteur
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
    </div>
    """, unsafe_allow_html=True)
