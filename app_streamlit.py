# -*- coding: utf-8 -*-
# app_streamlit.py
# UI Streamlit – Indice d'Apport Collectif (IAC)
#
# ✅ Upload ZIP uniquement + extraction safe + limite 1GB décompressé
# ✅ CSS MINIMAL: on garde le thème Streamlit "de base"
# ✅ Sidebar inputs lisibles
# ✅ Crédit visible (fond clair + texte noir)
# ✅ Tooltips KPI (bulle visible au survol)
# ✅ Pression affichée en décimal (évite 0 trop souvent)

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
    max_total_uncompressed: int = 1024 * 1024 * 1024,  # 1GB extrait
):
    dest_dir.mkdir(parents=True, exist_ok=True)

    tmp_zip_path = dest_dir / "_upload.zip"
    with open(tmp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    total = 0
    with zipfile.ZipFile(tmp_zip_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ValueError(f"ZIP trop volumineux: {len(infos)} fichiers (max {max_files}).")

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
    page_title="Indice d’Apport Collectif (IAC)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Indice d'Apport Collectif (IAC) - KPIs"},
)

# ===================== CSS MINIMAL =====================
# ⚠️ On évite tout style global sur le background.
# On touche uniquement: sidebar inputs, credit card, tooltip KPI.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* ✅ Sidebar: inputs lisibles (Streamlit 1.50) */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  background: #f4f6f8 !important;
  color: #111827 !important;
  border: 1px solid #cfd4dc !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] input {
  -webkit-text-fill-color: #111827 !important;
  caret-color: #111827 !important;
}
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
  color: #6b7280 !important;
}

/* Selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #f4f6f8 !important;
  border: 1px solid #cfd4dc !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
  color: #111827 !important;
  font-weight: 500 !important;
}
[data-baseweb="popover"] * { color: #111827 !important; }

/* ✅ Crédit visible (fond clair + texte noir) */
.credit-card {
  margin-top: 0.35rem;
  margin-bottom: 1.1rem;
  padding: 0.9rem 1rem;
  background: #ffffff;
  border: 1px solid rgba(17, 24, 39, 0.12);
  border-left: 4px solid #22c55e;
  color: #111827;
  font-size: 0.92rem;
  line-height: 1.35rem;
  border-radius: 12px;
  max-width: 1100px;
}
.credit-card b { color: #111827; }

/* ✅ Tooltips KPI visibles */
.kpi-wrap { position: relative; display: inline-block; width: 100%; }
.kpi-label { cursor: help; }
.kpi-tip {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  left: 0;
  top: -0.2rem;
  transform: translateY(-100%);
  max-width: 420px;
  background: rgba(17, 24, 39, 0.96);
  border: 1px solid rgba(17, 24, 39, 0.12);
  color: rgba(255,255,255,0.95);
  padding: 0.55rem 0.7rem;
  border-radius: 10px;
  font-size: 0.9rem;
  line-height: 1.25rem;
  z-index: 9999;
  box-shadow: 0 12px 32px rgba(0,0,0,0.2);
}
.kpi-wrap:hover .kpi-tip { visibility: visible; opacity: 1; transition: opacity 0.12s ease-in; }

@media (max-width: 900px){
  .kpi-tip { max-width: 320px; }
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
    press_r1_m,
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
        press_t2_s=press_t1_s,
        press_r1_m=press_r1_m,
        press_r2_m=press_r1_m,
    )


def kpi_box(label: str, value: str, tooltip: str):
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    label_esc = esc(label)
    value_esc = esc(value)
    tip_esc = esc(tooltip)

    st.markdown(
        f"""
<div class="kpi-wrap">
  <div class="kpi-tip">{tip_esc}</div>
  <div class="kpi-label" style="font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; color:#6b7280; font-weight:700;">
    {label_esc}
  </div>
  <div style="font-size:2.2rem; font-weight:800; color:#10b981;">
    {value_esc}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def create_radar_chart(summary, player_name, team_name, post_s_possession, post_s_struct):
    if not PLOTLY_OK:
        return None

    categories = [
        f"Possession +{int(post_s_possession)}s",
        f"Delta Largeur +{int(post_s_struct)}s",
        f"Delta Hauteur +{int(post_s_struct)}s",
        f"Delta Compacité +{int(post_s_struct)}s",
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
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=player_name,
        )
    )
    fig.update_layout(height=450, margin=dict(t=60, b=40, l=40, r=40), showlegend=False)
    return fig


# ===================== HEADER =====================
st.title("Indice d’Apport Collectif (IAC)")
st.caption("8 indicateurs simples pour lire l’impact collectif d’une touche (avec tracking)")

# ✅ Crédit visible sous le titre (texte noir)
st.markdown(
    """
<div class="credit-card">
  <b>Crédit données</b> — Initiative open-source conjointe <b>SkillCorner</b> × <b>PySport</b>.
  Merci de créditer <b>SkillCorner</b> et <b>PySport</b> lors de toute utilisation.
</div>
""",
    unsafe_allow_html=True,
)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    st.markdown("### Upload ZIP (dataset externe)")
    uploaded_zip = st.file_uploader("ZIP (structure libre)", type=["zip"])

    if uploaded_zip is not None:
        st.caption("Limites extraction: ~1.0 GB une fois extrait (max 8000 fichiers).")
        try:
            st.caption(f"Taille du ZIP: ~{uploaded_zip.size/1024/1024:.1f} MB")
        except Exception:
            pass

        if st.button("Utiliser ce ZIP", use_container_width=True):
            with st.spinner("Extraction du ZIP..."):
                _clear_previous_upload()
                extract_dir = _upload_base_dir() / f"upload_{os.getpid()}_{getattr(uploaded_zip,'size',0)}"
                safe_extract_zip(uploaded_zip, extract_dir)
                st.session_state["uploaded_extract_dir"] = str(extract_dir)
                st.session_state["uploaded_root"] = str(extract_dir)
            st.success("ZIP chargé. Clique sur CHARGER LES MATCHS.")

    st.markdown("---")

    if st.session_state.get("uploaded_root"):
        base_matches_dir = st.text_input("Dossier racine (ZIP extrait)", st.session_state["uploaded_root"], disabled=True)
        mode = st.selectbox("Mode listing", ["Scan (recommandé)"], index=0)
    else:
        base_matches_dir = st.text_input("Dossier racine (local)", "data")
        mode = st.selectbox("Mode listing", ["Scan (recommandé)", "Strict (data/matches)"], index=0)

    st.markdown("### Paramètres temporels")
    pre_s = st.number_input("Fenêtre AVANT (s)", min_value=0.0, value=1.0, step=0.5)
    post_s_struct = st.number_input("Horizon structure (+s)", min_value=1.0, value=5.0, step=0.5)
    post_s_possession = st.number_input("Horizon possession (+s)", min_value=1.0, value=6.0, step=0.5)
    fps = st.number_input("FPS (tracking)", min_value=1, value=10, step=1)

    st.markdown("### Paramètres de pression")
    press_t1_s = st.number_input("Temps pression (s)", min_value=0.5, value=3.0, step=0.5)
    press_r1_m = st.number_input(f"Rayon pression (m) @+{int(press_t1_s)}s", min_value=2.0, value=3.0, step=0.5)

st.markdown("---")

# ===================== MAIN FLOW =====================
if st.button("CHARGER LES MATCHS", use_container_width=True):
    with st.spinner("Chargement..."):
        if mode.startswith("Strict"):
            st.session_state["matches_list"] = backend.list_matches(str(Path(base_matches_dir) / "matches"))
        else:
            st.session_state["matches_list"] = backend.list_matches_anywhere(base_matches_dir)
    st.success("Matchs chargés.")

matches_list = st.session_state.get("matches_list", None)

if matches_list:
    labels = [_unpack_match_entry(e)[2] for e in matches_list]
    choice = st.selectbox("Sélectionner un match", labels, index=0)

    chosen = next(e for e in matches_list if _unpack_match_entry(e)[2] == choice)
    mid, mpath, _, events_csv, tracking_jsonl = _unpack_match_entry(chosen)

    home, away, date = backend.read_match_meta(mpath)
    home = home or "Home"
    away = away or "Away"

    st.subheader(f"{home} vs {away}")
    if date:
        st.caption(f"Date: {date} — ID: {mid}")

    if not (events_csv and tracking_jsonl):
        st.error("Fichiers incomplets: events/tracking introuvables.")
        st.stop()

    st.markdown("### Fichiers utilisés")
    st.write(f"Events CSV: `{events_csv}`")
    st.write(f"Tracking JSONL: `{tracking_jsonl}`")

    events_df = pd.read_csv(str(events_csv), low_memory=False)

    st.markdown("## Sélection des joueurs")
    st.caption("Choisis jusqu’à 2 joueurs par équipe")

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
        st.warning("Aucune équipe détectée.")
        st.stop()

    cols = st.columns(min(2, len(teams)))
    selections = {}
    for idx, team in enumerate(teams):
        with cols[idx]:
            st.markdown(f"### {team}")
            tdf = (
                tops[tops["team_shortname"] == team]
                .sort_values("n_events", ascending=False)
                .head(8)
                .reset_index(drop=True)
            )
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
        TIP_TOUCHES = "Nombre de prises de balle (touches) détectées dans les events."
        TIP_POSSESSION = f"Sur ces touches, % où l’équipe garde le ballon à +{int(post_s_possession)}s."
        TIP_PRESSURE = (
            f"Nombre moyen d’adversaires proches du porteur à +{int(press_t1_s)}s "
            f"(rayon {press_r1_m:.1f} m)."
        )
        TIP_SPEED = "Variation du rythme collectif après la touche (0→6s comparé à -3→0s)."
        TIP_WIDTH = f"Évolution de l’écartement (largeur) de l’équipe à +{int(post_s_struct)}s."
        TIP_HEIGHT = f"Évolution de la profondeur (hauteur) de l’équipe à +{int(post_s_struct)}s."
        TIP_COMPACT = f"Évolution du resserrement (compacité) de l’équipe à +{int(post_s_struct)}s."

        for team, pinfo in selections.items():
            if not pinfo:
                continue
            st.markdown(f"## Résultats — {team}")

            for pid, pname, tname in pinfo:
                with st.spinner(f"Analyse de {pname}..."):
                    df_out, summary = run_compute(
                        str(events_csv),
                        str(tracking_jsonl),
                        player_id=pid,
                        pre_s=float(pre_s),
                        post_s_possession=float(post_s_possession),
                        post_s_struct=float(post_s_struct),
                        fps=int(fps),
                        press_t1_s=float(press_t1_s),
                        press_r1_m=float(press_r1_m),
                    )

                if not summary:
                    st.warning(f"Aucune touche pour {pname}")
                    continue

                poss = get_summary_value(summary, "possession_retained_rate", [])
                w = get_summary_value(summary, "delta_width_pct_mean", [])
                h = get_summary_value(summary, "delta_height_pct_mean", [])
                c = get_summary_value(summary, "delta_compact_pct_mean", [])
                v = get_summary_value(summary, "delta_team_speed_pct", [])

                # Pression affichée en décimal (plus informatif)
                pressure_mean = None
                try:
                    if df_out is not None and "pressure_n_r1" in df_out.columns and df_out["pressure_n_r1"].notna().any():
                        pressure_mean = float(df_out["pressure_n_r1"].dropna().mean())
                except Exception:
                    pressure_mean = None

                st.markdown(f"### {pname} — {tname}")

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    kpi_box("Touches", str(summary.get("n_touches", "—")), TIP_TOUCHES)
                with r2:
                    kpi_box(f"Possession +{int(post_s_possession)}s", fmt_rate(poss, 1) or "—", TIP_POSSESSION)
                with r3:
                    kpi_box(
                        f"Pression +{int(press_t1_s)}s",
                        (f"{pressure_mean:.1f}" if pressure_mean is not None else "—"),
                        TIP_PRESSURE,
                    )
                with r4:
                    kpi_box("Delta Vitesse", fmt_pct(v, 1, signed=True) or "—", TIP_SPEED)

                b1, b2, b3 = st.columns(3)
                with b1:
                    kpi_box(f"Delta Largeur +{int(post_s_struct)}s", fmt_pct(w, 1, signed=True) or "—", TIP_WIDTH)
                with b2:
                    kpi_box(f"Delta Hauteur +{int(post_s_struct)}s", fmt_pct(h, 1, signed=True) or "—", TIP_HEIGHT)
                with b3:
                    kpi_box(f"Delta Compacité +{int(post_s_struct)}s", fmt_pct(c, 1, signed=True) or "—", TIP_COMPACT)

                if PLOTLY_OK:
                    st.markdown("#### Profil (radar)")
                    radar = create_radar_chart(summary, pname, tname, post_s_possession, post_s_struct)
                    st.plotly_chart(radar, use_container_width=True)

else:
    st.info("Uploade un ZIP dans la sidebar, clique sur « Utiliser ce ZIP », puis « CHARGER LES MATCHS ».")
