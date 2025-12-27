# -*- coding: utf-8 -*-
# app_streamlit.py
# UI Streamlit – KPIs simplifiés (8 indicateurs) avec robustesse et pression
# Design moderne inspiré outils d'analyse sportive
#
# ✅ Upload ZIP uniquement + extraction safe + limite 1GB décompressé
# ❌ DEBUG retiré

from pathlib import Path
import json
import os
import shutil
import tempfile
import zipfile

import pandas as pd
import streamlit as st

# Plotly (optionnel)
PLOTLY_OK = True
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    PLOTLY_OK = False

# ✅ IMPORT ROBUSTE: on importe le module, pas une liste de symboles
import calc_iic as backend


# ===================== ZIP SAFE EXTRACT =====================
def _upload_base_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "scfu_uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_safe_zip_path(dest_dir: Path, member_name: str) -> bool:
    member_path = Path(member_name)
    if member_path.is_absolute():
        return False
    resolved = (dest_dir / member_path).resolve()
    return str(resolved).startswith(str(dest_dir.resolve()))


def _clear_previous_upload():
    old = st.session_state.get("uploaded_extract_dir")
    if old:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass
    st.session_state["uploaded_extract_dir"] = None
    st.session_state["uploaded_root"] = None
    st.session_state["matches_list"] = None


def safe_extract_zip(
    uploaded_file,
    dest_dir: Path,
    max_files: int = 8000,
    max_total_uncompressed: int = 1024 * 1024 * 1024,  # ✅ 1GB extrait
):
    """
    Extrait un ZIP de manière safe:
      - limite nb fichiers
      - limite taille décompressée totale
      - protection zip-slip
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    tmp_zip_path = dest_dir / "_upload.zip"
    with open(tmp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    total = 0
    with zipfile.ZipFile(tmp_zip_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ValueError(f"ZIP trop volumineux: {len(infos)} fichiers (max {max_files}).")

        # pre-check + zip-slip
        for info in infos:
            if info.is_dir():
                continue
            total += int(info.file_size or 0)
            if total > max_total_uncompressed:
                raise ValueError(
                    f"ZIP trop lourd une fois extrait (~{total/1024/1024:.1f} MB). "
                    f"Max autorisé: {max_total_uncompressed/1024/1024:.0f} MB."
                )
            if not _is_safe_zip_path(dest_dir, info.filename):
                raise ValueError("ZIP invalide (chemins dangereux détectés).")

        # extraction
        for info in infos:
            if info.is_dir():
                continue
            if not _is_safe_zip_path(dest_dir, info.filename):
                continue
            out_path = (dest_dir / info.filename).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    try:
        tmp_zip_path.unlink(missing_ok=True)
    except Exception:
        pass


def _pick_biggest(paths):
    pp = []
    for p in paths:
        if p and Path(p).exists():
            pp.append(Path(p))
    if not pp:
        return None
    return str(max(pp, key=lambda x: x.stat().st_size))


def _unpack_match_entry(entry):
    """
    Backend list_matches_anywhere retourne: (mid, match_dir, label)
    On repère les fichiers dans match_dir (choix du plus gros si plusieurs)
    Retour: (mid, mpath, label, events_csv, tracking_jsonl)
    """
    if len(entry) != 3:
        raise ValueError("Format match entry non supporté (attendu: (mid, match_dir, label))")

    mid, mpath, label = entry
    match_dir = Path(mpath)

    ev_candidates = [str(p) for p in match_dir.glob("*_dynamic_events.csv")]
    tr_candidates = [str(p) for p in match_dir.glob("*_tracking_extrapolated.jsonl")]

    ev = _pick_biggest(ev_candidates)
    tr = _pick_biggest(tr_candidates)
    return mid, str(match_dir), label, ev, tr


# ===================== CONFIG STREAMLIT =====================
st.set_page_config(
    page_title="Football Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Analyse de l'Apport Collectif - 8 Indicateurs"},
)

# ===================== CSS (design inchangé + sidebar blanc) =====================
st.markdown(
    """
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

    .subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: -0.4rem;
        margin-bottom: 1.6rem;
        font-weight: 500;
    }

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

    /* Sidebar texte blanc */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: rgba(255,255,255,0.80) !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
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
</style>
""",
    unsafe_allow_html=True,
)


# ===================== UTILS =====================
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
    if x is None:
        return None
    try:
        return f"{round(100 * float(x), n):.{n}f}%"
    except Exception:
        return None


def get_summary_value(summary: dict, preferred_key: str, fallback_keys: list):
    if summary is None:
        return None
    if preferred_key in summary and summary.get(preferred_key) is not None:
        return summary.get(preferred_key)
    for k in fallback_keys:
        if k in summary and summary.get(k) is not None:
            return summary.get(k)
    return None


def run_compute(
    events_csv,
    tracking_jsonl,
    player_id,
    pre_s,
    post_s_possession,
    post_s_struct,
    fps,
    press_t1_s,
    press_t2_s,
    press_r1_m=None,
    press_r2_m=None,
):
    return backend.compute_iic_for_player(
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


def create_radar_chart(summary, player_name, team_name, post_s_possession, post_s_struct):
    if not PLOTLY_OK:
        return None

    categories = [
        f"Possession +{int(post_s_possession)}s",
        f"Delta Largeur +{int(post_s_struct)}s",
        f"Delta Hauteur +{int(post_s_struct)}s",
        f"Delta Compacite +{int(post_s_struct)}s",
        "Delta Vitesse",
    ]

    poss = get_summary_value(summary, "possession_retained_rate", [])
    w = get_summary_value(summary, "delta_width_pct_mean", [])
    h = get_summary_value(summary, "delta_height_pct_mean", [])
    c = get_summary_value(summary, "delta_compact_pct_mean", [])
    v = get_summary_value(summary, "delta_team_speed_pct", [])

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
        fill="toself",
        name=player_name,
        line=dict(color="#00ff87", width=3),
        fillcolor="rgba(0, 255, 135, 0.3)",
        marker=dict(size=8, color="#60efff"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(96, 239, 255, 0.2)", color="#8b92a8"),
            angularaxis=dict(gridcolor="rgba(96, 239, 255, 0.2)", color="#60efff"),
            bgcolor="rgba(15, 20, 25, 0.5)",
        ),
        showlegend=False,
        title=dict(text=f"<b>{player_name}</b> - {team_name}", font=dict(color="#60efff", size=18), x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        height=450,
        margin=dict(t=90, b=60, l=60, r=60),
    )
    return fig


# ===================== HEADER =====================
st.title("Football Analytics Dashboard")
st.markdown("<p class='subtitle'>Analyse avancee de l'apport collectif - 8 indicateurs cles</p>", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    st.markdown("### Upload ZIP (dataset externe)")
    uploaded_zip = st.file_uploader("ZIP (structure libre)", type=["zip"])

    if uploaded_zip is not None:
        st.caption("Conseil: évite les ZIP trop gros sur Streamlit Cloud (upload + extraction).")
        if st.button("Utiliser ce ZIP", use_container_width=True):
            with st.spinner("Extraction du ZIP..."):
                _clear_previous_upload()
                extract_dir = _upload_base_dir() / f"upload_{os.getpid()}_{uploaded_zip.size}"
                safe_extract_zip(uploaded_zip, extract_dir)
                st.session_state["uploaded_extract_dir"] = str(extract_dir)
                st.session_state["uploaded_root"] = str(extract_dir)
            st.success("ZIP chargé. Clique sur CHARGER LES MATCHS.")

    st.markdown("---")

    if st.session_state.get("uploaded_root"):
        base_matches_dir = st.text_input("Dossier racine (ZIP extrait)", st.session_state["uploaded_root"], disabled=True)
        mode = st.selectbox("Mode listing", ["Scan (recommande)"], index=0)
    else:
        base_matches_dir = st.text_input("Dossier racine (local)", "data")
        mode = st.selectbox("Mode listing", ["Scan (recommande)", "Strict (data/matches)"], index=0)

    st.markdown("### Parametres temporels")
    pre_s = st.number_input("Fenetre AVANT (s)", min_value=0.0, value=1.0, step=0.5)
    post_s_struct = st.number_input("Horizon structure (+s)", min_value=1.0, value=5.0, step=0.5)
    post_s_possession = st.number_input("Horizon possession (+s)", min_value=1.0, value=6.0, step=0.5)
    fps = st.number_input("FPS (tracking)", min_value=1, value=10, step=1)

    st.markdown("### Parametres de pression")
    press_t1_s = st.number_input("Temps pression 1 (s)", min_value=0.5, value=3.0, step=0.5)
    press_r1_m = st.number_input(f"Rayon pression 1 (m) @+{int(press_t1_s)}s", min_value=2.0, value=3.0, step=0.5)
    press_t2_s = st.number_input("Temps pression 2 (s)", min_value=0.5, value=5.0, step=0.5)
    press_r2_m = st.number_input(f"Rayon pression 2 (m) @+{int(press_t2_s)}s", min_value=3.0, value=5.0, step=0.5)

st.markdown("---")

# ===================== MAIN FLOW =====================
if st.button("CHARGER LES MATCHS", use_container_width=True):
    with st.spinner("Chargement..."):
        if mode.startswith("Strict"):
            st.session_state["matches_list"] = backend.list_matches(str(Path(base_matches_dir) / "matches"))
        else:
            st.session_state["matches_list"] = backend.list_matches_anywhere(base_matches_dir)
    st.success("Matchs charges.")

matches_list = st.session_state.get("matches_list", None)

if matches_list:
    labels = [_unpack_match_entry(e)[2] for e in matches_list]
    choice = st.selectbox("Selectionner un match", labels, index=0)

    chosen = next(e for e in matches_list if _unpack_match_entry(e)[2] == choice)
    mid, mpath, _, events_csv, tracking_jsonl = _unpack_match_entry(chosen)

    home, away, date = backend.read_match_meta(mpath)
    home = home or "Home"
    away = away or "Away"

    st.markdown(f"""
    <div class="match-card">
        <h2 style="margin-top: 0; border: none; font-size: 2rem;">{home} <span style="color: #60efff;">vs</span> {away}</h2>
        <div style="margin-top: 1rem;">
            <span class="info-badge">Date: {date if date else 'inconnue'}</span>
            <span class="info-badge">ID: {mid}</span>
        </div>
        <p style="color: #5a6173; font-size: 0.85rem; margin-top: 1.5rem; margin-bottom: 0;">Dossier: {mpath}</p>
    </div>
    """, unsafe_allow_html=True)

    if not (events_csv and tracking_jsonl):
        st.error("Fichiers incomplets: events/tracking introuvables.")
        st.stop()

    # Affiche chemins + tailles (pour éviter les mauvais matchings)
    try:
        ev_size = round(Path(events_csv).stat().st_size / (1024 * 1024), 2)
    except Exception:
        ev_size = None
    try:
        tr_size = round(Path(tracking_jsonl).stat().st_size / (1024 * 1024), 2)
    except Exception:
        tr_size = None

    st.markdown("### Fichiers utilises")
    st.write(f"Events CSV: `{events_csv}` ({ev_size} MB)" if ev_size is not None else f"Events CSV: `{events_csv}`")
    st.write(f"Tracking JSONL: `{tracking_jsonl}` ({tr_size} MB)" if tr_size is not None else f"Tracking JSONL: `{tracking_jsonl}`")

    # Petit check rapide du tracking (détecte souvent un mauvais fichier / LFS pointer)
    try:
        with open(tracking_jsonl, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(1200)
        if ("\"frame\"" not in head) or ("\"player_data\"" not in head):
            st.warning("Le tracking selectionne ne ressemble pas a un JSONL SkillCorner (frame/player_data). KPI vides.")
    except Exception:
        st.warning("Impossible de lire un apercu du tracking.")

    events_df = pd.read_csv(str(events_csv), low_memory=False)

    # Selection joueurs
    st.markdown("## Selection des joueurs")
    st.markdown("Choisis jusqu'a 2 joueurs par equipe")

    p2t_map = backend.player_team_mode_map(events_df)

    tops_raw = backend.top_players_by_possession(str(events_csv), top_n=16)
    tops = tops_raw.copy()
    tops["team_mode"] = tops["player_in_possession_id"].map(p2t_map)
    tops = tops[~tops["team_mode"].isna()].copy()
    tops["team_shortname"] = tops["team_mode"]
    tops = tops.drop_duplicates(subset=["player_in_possession_id"])

    teams = list(tops["team_shortname"].dropna().unique())
    if len(teams) > 2:
        teams = tops["team_shortname"].value_counts().index.tolist()[:2]

    if not teams:
        st.warning("Aucune equipe detectee.")
        st.stop()

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
                label = f"{nm} - {pid} ({ne} touches)"
                if st.checkbox(label, value=(len(picks) < 2), key=f"pick_{mid}_{team}_{pid}"):
                    picks.append((pid, nm, team))
            selections[team] = picks

    st.markdown("---")

    if st.button("CALCULER LES KPI", use_container_width=True):
        for team, pinfo in selections.items():
            if not pinfo:
                continue
            st.markdown(f"## Resultats - {team}")

            for pid, pname, tname in pinfo:
                with st.spinner(f"Analyse de {pname}..."):
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
                    st.warning(f"Aucune touche pour {pname}")
                    continue

                poss = get_summary_value(summary, "possession_retained_rate", [])
                w = get_summary_value(summary, "delta_width_pct_mean", [])
                h = get_summary_value(summary, "delta_height_pct_mean", [])
                c = get_summary_value(summary, "delta_compact_pct_mean", [])
                v = get_summary_value(summary, "delta_team_speed_pct", [])
                p1 = get_summary_value(summary, "pressure_n_r1_mean", [])
                p2 = get_summary_value(summary, "pressure_n_r2_mean", [])

                st.markdown('<div class="player-section">', unsafe_allow_html=True)
                st.markdown(f"### {pname} - {tname}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Touches", summary.get("n_touches"))
                c2.metric(f"Possession +{int(post_s_possession)}s", fmt_rate(poss, 1) or "-")
                c3.metric(f"Delta Largeur +{int(post_s_struct)}s", fmt_pct(w, 1, signed=True) or "-")
                c4.metric(f"Delta Hauteur +{int(post_s_struct)}s", fmt_pct(h, 1, signed=True) or "-")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric(f"Delta Compacite +{int(post_s_struct)}s", fmt_pct(c, 1, signed=True) or "-")
                c6.metric("Delta Vitesse", fmt_pct(v, 1, signed=True) or "-")
                c7.metric(f"Pression +{int(press_t1_s)}s", str(int(p1)) if p1 is not None else "-")
                c8.metric(f"Pression +{int(press_t2_s)}s", str(int(p2)) if p2 is not None else "-")

                if PLOTLY_OK:
                    st.markdown("#### Profil (radar)")
                    radar = create_radar_chart(summary, pname, tname, post_s_possession, post_s_struct)
                    st.plotly_chart(radar, use_container_width=True)

                st.markdown("#### Export")
                colA, colB = st.columns(2)
                with colA:
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Telecharger details CSV",
                        data=csv_bytes,
                        file_name=f"KPI_details_{mid}_{pid}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{mid}_{pid}",
                    )
                with colB:
                    jbytes = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
                    st.download_button(
                        "Telecharger resume JSON",
                        data=jbytes,
                        file_name=f"KPI_summary_{mid}_{pid}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl_json_{mid}_{pid}",
                    )

                with st.expander("Voir les donnees brutes"):
                    st.dataframe(df_out, use_container_width=True, height=400)

                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">⚽</div>
        <h2 style="border: none; margin-bottom: 1rem;">Bienvenue</h2>
        <p style="font-size: 1.1rem; color: rgba(255,255,255,0.75); max-width: 700px; margin: 0 auto 2rem;">
            Uploade un ZIP dans la sidebar, clique "Utiliser ce ZIP", puis "CHARGER LES MATCHS".
        </p>
    </div>
    """, unsafe_allow_html=True)
