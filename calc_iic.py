# calc_iic.py — Backend IIC (8 KPI) — secondes dynamiques + scan dossiers externes
# Dépendances: pandas
# Fichiers attendus par match: *_dynamic_events.csv et *_tracking_extrapolated.jsonl

import json
import math
from statistics import mean
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import pandas as pd


# ============================================================
# 1) TOUCHES (harmonisation liste joueurs vs analyse)
# ============================================================
def extract_touches(df: pd.DataFrame, player_id: Optional[int] = None) -> pd.DataFrame:
    """
    Retourne les 'touches' harmonisées à partir de dynamic_events.csv.
    - Filtre player_in_possession_id (si donné)
    - Garde le jeu en cours (is_in_play==True/1) si la colonne existe
    - Déduplique par frame_start (sécurise les doublons éventuels)
    """
    d = df.copy()

    if "player_in_possession_id" not in d.columns:
        return d.iloc[0:0].copy()

    d = d.dropna(subset=["player_in_possession_id"])
    d["player_in_possession_id"] = d["player_in_possession_id"].astype(int)

    if player_id is not None:
        d = d[d["player_in_possession_id"] == int(player_id)]

    if "is_in_play" in d.columns:
        d = d[d["is_in_play"].astype(str).isin(["1", "True", "true", "TRUE"])]

    if "frame_start" in d.columns:
        sort_cols = [c for c in ["period", "frame_start"] if c in d.columns]
        if sort_cols:
            d = d.sort_values(sort_cols, na_position="last")
        d = d.drop_duplicates(subset=["player_in_possession_id", "frame_start"], keep="first")

    return d


# ============================================================
# 2) TEAM MAPPING (évite joueur dans 2 équipes)
# ============================================================
def player_team_mode_map(df: pd.DataFrame) -> Dict[int, str]:
    """
    Associe player_id -> team_shortname via la modalité la plus fréquente dans les events.
    Utilise plusieurs colonnes si elles existent, mais surtout player_in_possession_id.
    """
    mapping_counts: Dict[int, Counter] = {}

    candidates = [c for c in ["player_in_possession_id", "player_id", "player_targeted_id"] if c in df.columns]
    if "team_shortname" not in df.columns:
        return {}

    for col_id in candidates:
        sub = df[[col_id, "team_shortname"]].dropna()
        if sub.empty:
            continue
        sub[col_id] = sub[col_id].astype(int)
        for pid, team in zip(sub[col_id], sub["team_shortname"]):
            mapping_counts.setdefault(pid, Counter())
            mapping_counts[pid][team] += 1

    return {pid: cnt.most_common(1)[0][0] for pid, cnt in mapping_counts.items()}


def opponent_of(team: str, teams: List[str]) -> Optional[str]:
    for t in teams:
        if t != team:
            return t
    return None


# ============================================================
# 3) OUTILS GÉOMÉTRIE / % / robustesse frames
# ============================================================
def _range(vals):
    vals = [v for v in vals if v is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def _team_lateral_width(positions):
    ys = [y for x, y, pid in positions if y is not None]
    return _range(ys)


def _team_longitudinal_height(positions):
    xs = [x for x, y, pid in positions if x is not None]
    return _range(xs)


def _compacity(positions):
    xs = [x for x, y, pid in positions if x is not None]
    ys = [y for x, y, pid in positions if y is not None]
    if not xs or not ys:
        return None, (None, None)
    cx, cy = mean(xs), mean(ys)
    d = [math.hypot(x - cx, y - cy) for x, y, pid in positions if x is not None and y is not None]
    return (mean(d) if d else None), (cx, cy)


def _dist(a, b):
    if a is None or b is None:
        return None
    x1, y1 = a
    x2, y2 = b
    if None in (x1, y1, x2, y2):
        return None
    return math.hypot(x2 - x1, y2 - y1)


def _pct_change(after, before, eps=1e-6):
    if after is None or before is None or abs(before) < eps:
        return None
    return 100.0 * (after - before) / before


def _nearest_key(dct, target, tol=2):
    if not dct:
        return None
    if target in dct:
        return target
    cand = min(dct.keys(), key=lambda k: abs(k - target))
    return cand if abs(cand - target) <= tol else None


def _team_centroid(positions):
    xs = [x for x, y, p in positions if x is not None]
    ys = [y for x, y, p in positions if y is not None]
    if not xs or not ys:
        return None
    return (mean(xs), mean(ys))


# ============================================================
# 4) META MATCH (home/away/date) — priorité matches.json
# ============================================================
def _team_label_from(obj):
    if isinstance(obj, dict):
        for k in ("short_name", "shortName", "name", "clubName", "acronym"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        if "id" in obj:
            return f"Team {obj['id']}"
        return "Unknown"
    if isinstance(obj, str):
        return obj.strip() or "Unknown"
    return "Unknown"


def _date_label_from_str(s: str):
    if not s:
        return None
    s = s.strip()
    candidates = [s, s.replace("Z", "+00:00"), s.split("T")[0]]
    for c in candidates:
        try:
            dt = datetime.fromisoformat(c)
            return dt.date().isoformat()
        except Exception:
            pass
    fmts = [
        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%Y", "%d-%m-%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date().isoformat()
        except Exception:
            continue
    return None


def _load_matches_index_from_any_parent(match_dir: Path):
    candidates = [
        match_dir / "matches.json",
        match_dir.parent / "matches.json",
        match_dir.parent.parent / "matches.json",
    ]
    for f in candidates:
        try:
            if f.exists():
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                idx = {}
                if isinstance(data, list):
                    for e in data:
                        _id = e.get("id") or e.get("match_id") or e.get("matchId")
                        if _id is not None:
                            idx[str(_id)] = e
                elif isinstance(data, dict):
                    if "matches" in data and isinstance(data["matches"], list):
                        for e in data["matches"]:
                            _id = e.get("id") or e.get("match_id") or e.get("matchId")
                            if _id is not None:
                                idx[str(_id)] = e
                    else:
                        for k, e in data.items():
                            if isinstance(e, dict):
                                idx[str(k)] = e
                return idx
        except Exception:
            continue
    return {}


def _extract_home_away_date_from_index(entry: dict):
    home = None
    away = None
    date_str = None

    if isinstance(entry.get("home_team"), (dict, str)):
        home = _team_label_from(entry.get("home_team"))
    if isinstance(entry.get("away_team"), (dict, str)):
        away = _team_label_from(entry.get("away_team"))

    for k in ("home_team_name", "homeTeamName", "home", "homeTeam", "home_name"):
        if not home and isinstance(entry.get(k), str):
            home = entry[k]
    for k in ("away_team_name", "awayTeamName", "away", "awayTeam", "away_name"):
        if not away and isinstance(entry.get(k), str):
            away = entry[k]

    for dk in ("utc_kickoff_time", "kickoff_time", "kickoff", "date", "match_date", "game_date", "utc_date", "start_time"):
        if date_str:
            break
        if entry.get(dk):
            date_str = _date_label_from_str(str(entry[dk]))

    return home, away, date_str


def read_match_meta(match_dir: str):
    md = Path(match_dir)
    mid = md.name

    # 0) matches.json (priorité)
    idx = _load_matches_index_from_any_parent(md)
    if idx and mid in idx:
        h, a, d = _extract_home_away_date_from_index(idx[mid])
        home_idx, away_idx, date_idx = h, a, d
    else:
        home_idx = away_idx = date_idx = None

    # 1) {id}_match.json (fallback)
    home_json = away_json = date_json = None
    mjson = next(md.glob(f"{mid}_match.json"), None)
    if mjson:
        try:
            with open(mjson, "r", encoding="utf-8") as f:
                data = json.load(f)
            home_obj = data.get("home_team") or data.get("homeTeam") or data.get("home")
            away_obj = data.get("away_team") or data.get("awayTeam") or data.get("away")
            if home_obj:
                home_json = _team_label_from(home_obj)
            if away_obj:
                away_json = _team_label_from(away_obj)

            for k in ["date", "kickoff_time", "kickoff", "start_time",
                      "match_date", "utc_date", "utc_kickoff_time", "gameDate"]:
                if k in data and data[k]:
                    date_json = _date_label_from_str(str(data[k]))
                    break
        except Exception:
            pass

    # 2) dynamic_events.csv (fallback date)
    date_csv = None
    if date_json is None and date_idx is None:
        ev = next(md.glob("*_dynamic_events.csv"), None)
        if ev:
            try:
                head = pd.read_csv(ev, nrows=10, low_memory=False)
                for col in ["match_date", "game_date", "date", "kickoff_time", "utc_kickoff_time", "start_time", "timestamp", "datetime"]:
                    if col in head.columns and head[col].notna().any():
                        date_csv = _date_label_from_str(str(head[col].dropna().iloc[0]))
                        if date_csv:
                            break
            except Exception:
                pass

    home = home_idx or home_json
    away = away_idx or away_json
    date_str = date_idx or date_json or date_csv
    return home, away, date_str


# ============================================================
# 5) LISTING MATCHS
# ============================================================
def list_matches(base_dir: str):
    """
    Mode 'strict' (structure SkillCorner): base_dir = .../data/matches
    """
    base = Path(base_dir)
    matches_dirs = sorted([p for p in base.glob("*") if p.is_dir()])
    idx_global = _load_matches_index_from_any_parent(base)

    labels = []
    for m in matches_dirs:
        mid = m.name
        home = away = date = None

        if idx_global and mid in idx_global:
            h, a, d = _extract_home_away_date_from_index(idx_global[mid])
            home, away, date = h or home, a or away, d or date

        h2, a2, d2 = read_match_meta(str(m))
        home = home or h2
        away = away or a2
        date = date or d2

        if home and away:
            label = f"{home} vs {away}" + (f" — {date}" if date else "") + f" ({mid})"
        else:
            label = mid + (f" — {date}" if date else "")

        labels.append((mid, str(m), label))
    return labels


def list_matches_anywhere(root_dir: str):
    """
    Mode 'scan': root_dir peut être N'IMPORTE quel dossier.
    On détecte un match dès qu'un dossier contient:
      - *_dynamic_events.csv
      - *_tracking_extrapolated.jsonl
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    matches = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        ev = list(d.glob("*_dynamic_events.csv"))
        tr = list(d.glob("*_tracking_extrapolated.jsonl"))
        if not (ev and tr):
            continue

        mid = d.name
        home, away, date = read_match_meta(str(d))
        home = home or "Home"
        away = away or "Away"
        label = f"{home} vs {away}" + (f" — {date}" if date else "") + f" ({mid})"
        matches.append((mid, str(d), label))

    matches.sort(key=lambda x: x[2])
    return matches


# ============================================================
# 6) TOP JOUEURS PAR TOUCHES (cohérent)
# ============================================================
def top_players_by_possession(events_csv: str, top_n: int = 8):
    df = pd.read_csv(events_csv, low_memory=False)

    touches = extract_touches(df, player_id=None)
    if touches.empty:
        return pd.DataFrame(columns=["team_shortname", "player_in_possession_id", "player_in_possession_name", "n_events"])

    if "team_shortname" not in touches.columns:
        touches["team_shortname"] = "TEAM"

    # Mapping joueur->équipe (majoritaire) pour éviter les anomalies
    p2t = player_team_mode_map(touches)
    if p2t:
        touches["team_shortname_fix"] = touches["player_in_possession_id"].map(p2t)
        touches.loc[touches["team_shortname_fix"].notna(), "team_shortname"] = touches["team_shortname_fix"]
        touches = touches.drop(columns=["team_shortname_fix"], errors="ignore")

    grp = (
        touches.groupby(["team_shortname", "player_in_possession_id", "player_in_possession_name"])
        .size()
        .reset_index(name="n_events")
    )

    return (
        grp.sort_values(["team_shortname", "n_events"], ascending=[True, False])
        .groupby("team_shortname")
        .head(top_n)
    )


# ============================================================
# 7) CALCUL IIC (8 KPI) — SECONDES DYNAMIQUES
# ============================================================
def compute_iic_for_player(
    events_csv: str,
    tracking_jsonl: str,
    player_id: int,
    pre_s: float = 1.0,
    post_s_possession: float = 6.0,
    post_s_struct: float = 5.0,
    fps: int = 10,
    press_t1_s: float = 3.0,   # temps pression 1 (ex +3s)
    press_t2_s: float = 5.0,   # temps pression 2 (ex +5s)
    press_r1_m: float = 3.0,   # rayon pression 1
    press_r2_m: float = 5.0,   # rayon pression 2
):
    """
    8 KPI:
    1 touches
    2 possession conservée (+post_s_possession)
    3 Δ largeur (+post_s_struct) en %
    4 Δ hauteur (+post_s_struct) en %
    5 Δ compacité (+post_s_struct) en %
    6 Δ vitesse collective (baseline -3→0 vs 0→+6) en %
    7 pression à +press_t1_s (rayon press_r1_m) en entier
    8 pression à +press_t2_s (rayon press_r2_m) en entier
    """
    df = pd.read_csv(events_csv, low_memory=False)

    teams_in_game = [t for t in df.get("team_shortname", pd.Series()).dropna().unique().tolist() if t]
    p2t = player_team_mode_map(df)

    touches = extract_touches(df, player_id=player_id).reset_index(drop=True)
    if touches.empty:
        return pd.DataFrame(), {}

    team_short = p2t.get(player_id)
    if not team_short and "team_shortname" in touches.columns:
        try:
            team_short = touches["team_shortname"].mode().iloc[0]
        except Exception:
            team_short = None
    team_short = team_short or "TEAM"

    opp_team = opponent_of(team_short, teams_in_game) if teams_in_game else None

    # Offsets frames
    pre_off = int(float(pre_s) * fps)
    f_poss = int(float(post_s_possession) * fps)
    f_struct = int(float(post_s_struct) * fps)

    f_m3 = int(3.0 * fps)  # baseline -3s
    f_p1 = int(float(press_t1_s) * fps)
    f_p2 = int(float(press_t2_s) * fps)
    f_p6 = int(6.0 * fps)  # post 0→6

    windows, needed = [], set()
    for _, r in touches.iterrows():
        f0 = int(r["frame_start"])
        f_pre = max(0, f0 - pre_off)
        f_beg = max(0, f0 - f_m3)
        f1 = f0 + f_p1
        f2 = f0 + f_p2
        f6 = f0 + f_p6
        f_poss_abs = f0 + f_poss
        f_struct_abs = f0 + f_struct

        windows.append((f0, f_beg, f_pre, f1, f2, f6, f_poss_abs, f_struct_abs))
        needed.update([f_beg, f_pre, f0, f1, f2, f6, f_poss_abs, f_struct_abs])

    positions_by_frame_team: Dict[int, List[Tuple[float, float, int]]] = {}
    positions_by_frame_opp: Dict[int, List[Tuple[float, float, int]]] = {}
    player_pos_by_frame: Dict[int, Tuple[float, float]] = {}
    possession_by_frame: Dict[int, str] = {}
    player_group = None  # home/away selon tracking

    with open(tracking_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fr = obj.get("frame")
            if fr is None:
                continue

            # détecter le "group" du porteur quand il a le ballon
            if player_group is None:
                poss = obj.get("possession", {}) or {}
                if poss.get("player_id") == player_id:
                    player_group = poss.get("group")

            if fr in needed:
                plist = obj.get("player_data", []) or []
                team_list, opp_list = [], []

                for p in plist:
                    pid = p.get("player_id")
                    if pid is None:
                        continue
                    pid = int(pid)
                    tname = p2t.get(pid)

                    tup = (p.get("x"), p.get("y"), pid)

                    if tname == team_short:
                        team_list.append(tup)
                    elif opp_team and tname == opp_team:
                        opp_list.append(tup)

                    if pid == player_id:
                        player_pos_by_frame[fr] = (p.get("x"), p.get("y"))

                positions_by_frame_team[fr] = team_list
                positions_by_frame_opp[fr] = opp_list
                possession_by_frame[fr] = (obj.get("possession", {}) or {}).get("group")

    def _v_avg(pA, pB, dt):
        d = _dist(pA, pB)
        return (d / dt) if (d is not None and dt > 0) else None

    rows = []
    for f0, f_beg, f_pre, f1, f2, f6, f_poss_abs, f_struct_abs in windows:
        k_beg = _nearest_key(positions_by_frame_team, f_beg, tol=2)
        k_0 = _nearest_key(positions_by_frame_team, f0, tol=2)
        k_6 = _nearest_key(positions_by_frame_team, f6, tol=2)
        k_pre = _nearest_key(positions_by_frame_team, f_pre, tol=2)
        k_str = _nearest_key(positions_by_frame_team, f_struct_abs, tol=2)

        team_beg = positions_by_frame_team.get(k_beg, [])
        team_0 = positions_by_frame_team.get(k_0, [])
        team_6 = positions_by_frame_team.get(k_6, [])
        pre_team = positions_by_frame_team.get(k_pre, [])
        postS_team = positions_by_frame_team.get(k_str, [])

        # Structure (% vs t0-pre_s)
        width_pre = _team_lateral_width(pre_team)
        width_postS = _team_lateral_width(postS_team)
        delta_width_pct = _pct_change(width_postS, width_pre)

        height_pre = _team_longitudinal_height(pre_team)
        height_postS = _team_longitudinal_height(postS_team)
        delta_height_pct = _pct_change(height_postS, height_pre)

        comp_pre, _ = _compacity(pre_team)
        comp_post, _ = _compacity(postS_team)
        delta_comp_pct = _pct_change(comp_post, comp_pre)

        # Tempo collectif : (0→6) vs (−3→0)
        map_beg = {pid: (x, y) for x, y, pid in team_beg}
        map_0 = {pid: (x, y) for x, y, pid in team_0}
        map_6 = {pid: (x, y) for x, y, pid in team_6}
        common = set(map_beg) & set(map_0) & set(map_6)

        deltas_speed_pct = []
        for pid in common:
            a = map_beg[pid]
            b = map_0[pid]
            d = map_6[pid]
            if None in (*a, *b, *d):
                continue
            v_base = _v_avg(a, b, 3.0)  # -3→0
            v_0_6 = _v_avg(b, d, 6.0)   # 0→6
            if v_base is None or v_0_6 is None or v_base < 1e-6:
                continue
            deltas_speed_pct.append(100.0 * (v_0_6 - v_base) / v_base)

        team_speed_pct = mean(deltas_speed_pct) if deltas_speed_pct else None

        # Fallback centroïde si <4 joueurs valides
        if (team_speed_pct is None) or (len(deltas_speed_pct) < 4):
            c_beg = _team_centroid(team_beg)
            c_0 = _team_centroid(team_0)
            c_6 = _team_centroid(team_6)
            if c_beg and c_0 and c_6:
                v_base = _v_avg(c_beg, c_0, 3.0)
                v_0_6 = _v_avg(c_0, c_6, 6.0)
                if v_base and v_base >= 1e-6 and v_0_6:
                    team_speed_pct = 100.0 * (v_0_6 - v_base) / v_base

        # Pression : adversaires dans rayon autour du porteur à +t1/+t2
        k1o = _nearest_key(positions_by_frame_opp, f1, tol=2)
        k2o = _nearest_key(positions_by_frame_opp, f2, tol=2)
        opp_1 = positions_by_frame_opp.get(k1o, [])
        opp_2 = positions_by_frame_opp.get(k2o, [])

        k1p = _nearest_key(player_pos_by_frame, f1, tol=2)
        k2p = _nearest_key(player_pos_by_frame, f2, tol=2)
        p1 = player_pos_by_frame.get(k1p)
        p2 = player_pos_by_frame.get(k2p)

        n_opp_r1 = None
        n_opp_r2 = None

        if p1 and p1[0] is not None and p1[1] is not None:
            n = 0
            for x, y, pid in opp_1:
                if x is None or y is None:
                    continue
                if math.hypot(x - p1[0], y - p1[1]) <= float(press_r1_m):
                    n += 1
            n_opp_r1 = n

        if p2 and p2[0] is not None and p2[1] is not None:
            n = 0
            for x, y, pid in opp_2:
                if x is None or y is None:
                    continue
                if math.hypot(x - p2[0], y - p2[1]) <= float(press_r2_m):
                    n += 1
            n_opp_r2 = n

        # Possession conservée à +post_s_possession
        k_poss = _nearest_key(possession_by_frame, f_poss_abs, tol=2)
        poss_post = possession_by_frame.get(k_poss)
        retained = int(poss_post == player_group) if (player_group is not None and poss_post is not None) else None

        rows.append({
            "frame_start": f0,

            "delta_width_pct": delta_width_pct,
            "delta_height_pct": delta_height_pct,
            "delta_compact_pct": delta_comp_pct,
            "delta_team_speed_pct": team_speed_pct,

            "pressure_n_r1": n_opp_r1,
            "pressure_n_r2": n_opp_r2,
            "possession_retained": retained,
        })

    out = pd.DataFrame(rows)

    # Summary (clés neutres + params)
    vals_r1 = out["pressure_n_r1"].dropna().to_list() if "pressure_n_r1" in out.columns else []
    vals_r2 = out["pressure_n_r2"].dropna().to_list() if "pressure_n_r2" in out.columns else []

    summary = {
        "n_touches": int(len(out)),
        "possession_retained_rate": float(out["possession_retained"].dropna().mean()) if out.get("possession_retained") is not None and out["possession_retained"].notna().any() else None,

        "delta_width_pct_mean": float(out["delta_width_pct"].dropna().mean()) if out.get("delta_width_pct") is not None and out["delta_width_pct"].notna().any() else None,
        "delta_height_pct_mean": float(out["delta_height_pct"].dropna().mean()) if out.get("delta_height_pct") is not None and out["delta_height_pct"].notna().any() else None,
        "delta_compact_pct_mean": float(out["delta_compact_pct"].dropna().mean()) if out.get("delta_compact_pct") is not None and out["delta_compact_pct"].notna().any() else None,

        "delta_team_speed_pct": float(out["delta_team_speed_pct"].dropna().mean()) if out.get("delta_team_speed_pct") is not None and out["delta_team_speed_pct"].notna().any() else None,

        # Pression en ENTIER
        "pressure_n_r1_mean": int(round(mean(vals_r1))) if vals_r1 else None,
        "pressure_n_r2_mean": int(round(mean(vals_r2))) if vals_r2 else None,

        # paramètres utilisés (pour l'UI)
        "pre_s": float(pre_s),
        "post_s_possession": float(post_s_possession),
        "post_s_struct": float(post_s_struct),
        "press_t1_s": float(press_t1_s),
        "press_t2_s": float(press_t2_s),
        "press_r1_m": float(press_r1_m),
        "press_r2_m": float(press_r2_m),
        "fps": int(fps),
        "team": team_short,
    }

    return out, summary
