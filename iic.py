# calc_iic.py
# KPIs "apport collectif" — version simplifiée
# 1) n_touches
# 2) possession_retained_rate_+6s
# 3) delta_width_pct_+5s_mean           (largeur %)
# 4) delta_compact_pct_+5s_mean         (compacité %)
# 5) delta_team_speed_pct_0_6_vs_-3_0   (tempo collectif %)
# 6) n_opponents_within_3m_at_+3s_mean  (pression immédiate)
# 7) n_opponents_within_5m_at_+5s_mean  (pression soutenue)

import json, math
from statistics import mean
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd

# ---------- Helpers équipe / mapping ----------
def player_team_mode_map(df: pd.DataFrame) -> Dict[int, str]:
    mapping_counts: Dict[int, Counter] = {}
    for col_id in ["player_in_possession_id", "player_id", "player_targeted_id"]:
        if col_id in df.columns:
            sub = df[[col_id, "team_shortname"]].dropna()
            if sub.empty: continue
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

# ---------- Géométrie ----------
def _range(vals):
    vals = [v for v in vals if v is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None

def _team_lateral_width(positions):
    ys = [y for x, y, pid in positions if y is not None]
    return _range(ys)

def _compacity(positions):
    xs = [x for x, y, pid in positions if x is not None]
    ys = [y for x, y, pid in positions if y is not None]
    if not xs or not ys:
        return None, (None, None)
    cx, cy = mean(xs), mean(ys)
    d = [math.hypot(x - cx, y - cy) for x, y, pid in positions if x is not None and y is not None]
    return (mean(d) if d else None), (cx, cy)

def _dist(a, b):
    if a is None or b is None: return None
    x1, y1 = a; x2, y2 = b
    if None in (x1, y1, x2, y2): return None
    return math.hypot(x2 - x1, y2 - y1)

def _pct_change(after, before, eps=1e-6):
    if after is None or before is None or abs(before) < eps:
        return None
    return 100.0 * (after - before) / before

# ---------- Meta match ----------
def _team_label_from(obj):
    if isinstance(obj, dict):
        for k in ("short_name","shortName","name","clubName","acronym"):
            v = obj.get(k)
            if isinstance(v,str) and v.strip(): return v.strip()
        if "id" in obj: return f"Team {obj['id']}"
        return "Unknown"
    if isinstance(obj,str): return obj.strip() or "Unknown"
    return "Unknown"

def _date_label(d):
    from datetime import datetime
    if not d: return None
    if isinstance(d,str):
        try:
            dt = datetime.fromisoformat(d.replace("Z","+00:00"))
            return dt.date().isoformat()
        except Exception:
            return d.split("T",1)[0] if "T" in d else d
    return str(d)

def read_match_meta(match_dir: str):
    md = Path(match_dir); mid = md.name
    mjson = next(md.glob(f"{mid}_match.json"), None)
    if not mjson: return None, None, None
    try:
        with open(mjson,"r",encoding="utf-8") as f: data = json.load(f)
        home_obj = data.get("home_team") or data.get("homeTeam") or data.get("home_team_name") or data.get("home")
        away_obj = data.get("away_team") or data.get("awayTeam") or data.get("away_team_name") or data.get("away")
        date_obj = data.get("date") or data.get("kickoff_time") or data.get("match_date")
        return _team_label_from(home_obj), _team_label_from(away_obj), _date_label(date_obj)
    except Exception:
        return None, None, None

def list_matches(base_dir: str):
    base = Path(base_dir)
    matches_dirs = sorted([p for p in base.glob("*") if p.is_dir()])
    labels = []
    for m in matches_dirs:
        mid = m.name
        home, away, date = read_match_meta(str(m))
        if home and away:
            label = f"{home} vs {away}" + (f" — {date}" if date else "") + f" ({mid})"
        else:
            ev = next(m.glob("*_dynamic_events.csv"), None)
            label = mid
            if ev:
                try:
                    tmp = pd.read_csv(ev, usecols=["team_shortname"], nrows=500)
                    teams = [t for t in tmp["team_shortname"].dropna().unique().tolist() if t]
                    if len(teams) >= 2: label = f"{teams[0]} vs {teams[1]} — {mid}"
                    elif len(teams) == 1: label = f"{teams[0]} — {mid}"
                except Exception: pass
        labels.append((mid, str(m), label))
    return labels

def top_players_by_possession(events_csv: str, top_n: int = 8):
    df = pd.read_csv(events_csv, low_memory=False)
    if "player_in_possession_id" not in df.columns:
        return pd.DataFrame(columns=["team_shortname","player_in_possession_id","player_in_possession_name","n_events"])
    d = df.dropna(subset=["player_in_possession_id"])
    grp = (d.groupby(["team_shortname","player_in_possession_id","player_in_possession_name"])
             .size().reset_index(name="n_events"))
    return (grp.sort_values(["team_shortname","n_events"], ascending=[True,False])
              .groupby("team_shortname").head(top_n))

# ---------- Calcul principal ----------
def compute_iic_for_player(
    events_csv: str,
    tracking_jsonl: str,
    player_id: int,
    pre_s: float = 1.0,            # référence structure (t0-1s)
    post_s_possession: float = 6.0,
    post_s_struct: float = 5.0,
    fps: int = 10,
    press_r1_m: float = 3.0,       # rayon pression immédiate à +3s
    press_r2_m: float = 5.0,       # rayon pression soutenue à +5s
):
    """
    Retour: (df_detail, summary) avec 7 KPI simplifiés.
    """
    df = pd.read_csv(events_csv, low_memory=False)
    teams_in_game = [t for t in df["team_shortname"].dropna().unique().tolist() if t]

    touches = (df[df["player_in_possession_id"] == player_id]
               .sort_values(["period", "frame_start"])
               .reset_index(drop=True))
    if touches.empty:
        return pd.DataFrame(), {}

    p2t = player_team_mode_map(df)
    team_short = p2t.get(
        player_id,
        df.loc[df["player_in_possession_id"] == player_id, "team_shortname"].mode().iloc[0]
    )
    opp_team = opponent_of(team_short, teams_in_game)

    # Frames requis
    pre_off = int(pre_s * fps)          # -1s
    f_poss = int(post_s_possession * fps)  # +6s
    f_struct = int(post_s_struct * fps)    # +5s
    f_m3 = int(3.0 * fps)                  # baseline -3->0
    f_p3 = int(3.0 * fps)                  # +3
    f_p5 = int(5.0 * fps)                  # +5
    f_p6 = int(6.0 * fps)                  # +6

    windows, needed = [], set()
    for _, r in touches.iterrows():
        f0 = int(r["frame_start"])
        f_pre = max(0, f0 - pre_off)
        f_beg = max(0, f0 - f_m3)           # t0-3
        f_3   = f0 + f_p3                   # t0+3
        f_5   = f0 + f_p5                   # t0+5
        f_6   = f0 + f_p6                   # t0+6
        f_poss_abs   = f0 + f_poss
        f_struct_abs = f0 + f_struct
        windows.append((f0, f_beg, f_pre, f_3, f_5, f_6, f_poss_abs, f_struct_abs))
        needed.update([f_beg, f_pre, f0, f_3, f_5, f_6, f_poss_abs, f_struct_abs])

    positions_by_frame_team: Dict[int, List[Tuple[float,float,int]]] = {}
    positions_by_frame_opp : Dict[int, List[Tuple[float,float,int]]] = {}
    player_pos_by_frame: Dict[int, Tuple[float,float]] = {}
    possession_by_frame: Dict[int, str] = {}
    player_group = None

    with open(tracking_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            fr = obj.get("frame")
            if fr is None: continue

            if player_group is None:
                poss = obj.get("possession", {}) or {}
                if poss.get("player_id") == player_id:
                    player_group = poss.get("group")

            if fr in needed:
                plist = obj.get("player_data", []) or []
                team_list, opp_list = [], []
                for p in plist:
                    pid = p.get("player_id")
                    if pid is None: continue
                    pid = int(pid)
                    tname = p2t.get(pid)
                    tup = (p.get("x"), p.get("y"), pid)
                    if tname == team_short: team_list.append(tup)
                    elif tname == opp_team: opp_list.append(tup)
                    if pid == player_id:
                        player_pos_by_frame[fr] = (p.get("x"), p.get("y"))
                positions_by_frame_team[fr] = team_list
                positions_by_frame_opp[fr]  = opp_list
                possession_by_frame[fr] = (obj.get("possession", {}) or {}).get("group")

            if (len(positions_by_frame_team) >= len(needed)) and (player_group is not None):
                break

    def _v_avg(pA, pB, dt):
        d = _dist(pA, pB)
        return (d / dt) if (d is not None and dt > 0) else None

    rows = []
    for f0, f_beg, f_pre, f3, f5, f6, f_poss_abs, f_struct_abs in windows:
        # Equipe
        team_beg = positions_by_frame_team.get(f_beg, [])
        team_0   = positions_by_frame_team.get(f0, [])
        team_3   = positions_by_frame_team.get(f3, [])
        team_5   = positions_by_frame_team.get(f5, [])
        team_6   = positions_by_frame_team.get(f6, [])
        pre_team   = positions_by_frame_team.get(f_pre, [])
        postS_team = positions_by_frame_team.get(f_struct_abs, [])

        # Structure (% vs pré t0-1s)
        width_pre   = _team_lateral_width(pre_team)
        width_postS = _team_lateral_width(postS_team)
        delta_width_pct = _pct_change(width_postS, width_pre)

        comp_pre, _ = _compacity(pre_team)
        comp_post, _ = _compacity(postS_team)
        delta_comp_pct = _pct_change(comp_post, comp_pre)

        # Tempo collectif % : (0→6) vs (−3→0)
        map_beg = {pid:(x,y) for x,y,pid in team_beg}
        map_0   = {pid:(x,y) for x,y,pid in team_0}
        map_6   = {pid:(x,y) for x,y,pid in team_6}
        common  = set(map_beg) & set(map_0) & set(map_6)

        deltas_speed_pct = []
        for pid in common:
            a = map_beg[pid]; b = map_0[pid]; d = map_6[pid]
            if None in (*a,*b,*d): continue
            v_base = _v_avg(a, b, 3.0)     # -3 -> 0
            v_0_6  = _v_avg(b, d, 6.0)     # 0  -> 6
            if v_base is None or v_0_6 is None or v_base < 1e-6: continue
            deltas_speed_pct.append(100.0 * (v_0_6 - v_base) / v_base)
        team_speed_pct = mean(deltas_speed_pct) if deltas_speed_pct else None

        # Pression adverse autour du porteur
        opp_3 = positions_by_frame_opp.get(f3, [])
        opp_5 = positions_by_frame_opp.get(f5, [])
        p3 = player_pos_by_frame.get(f3)
        p5 = player_pos_by_frame.get(f5)
        n_opp_3m, n_opp_5m = None, None

        if p3 and p3[0] is not None and p3[1] is not None:
            n = 0
            for x,y,pid in opp_3:
                if x is None or y is None: continue
                if math.hypot(x - p3[0], y - p3[1]) <= press_r1_m: n += 1
            n_opp_3m = n

        if p5 and p5[0] is not None and p5[1] is not None:
            n = 0
            for x,y,pid in opp_5:
                if x is None or y is None: continue
                if math.hypot(x - p5[0], y - p5[1]) <= press_r2_m: n += 1
            n_opp_5m = n

        # Possession +6s
        poss_post = possession_by_frame.get(f_poss_abs)
        retained = int(poss_post == player_group) if (player_group is not None and poss_post is not None) else None

        rows.append({
            "frame_start": f0,
            "f_-3": f_beg, "f_pre": f_pre, "f0": f0, "f3": f3, "f5": f5, "f6": f6,
            "post_frame_possession": f_poss_abs, "post_frame_struct": f_struct_abs,

            "delta_width_pct_+5s": delta_width_pct,
            "delta_compact_pct_+5s": delta_comp_pct,
            "team_speed_pct_0_6_vs_-3_0": team_speed_pct,

            "n_opponents_within_3m_at_+3s": n_opp_3m,
            "n_opponents_within_5m_at_+5s": n_opp_5m,

            "possession_retained_+6s": retained,
        })

    out = pd.DataFrame(rows)

    summary = {
        "n_touches": int(len(out)),
        "possession_retained_rate_+6s": float(out["possession_retained_+6s"].dropna().mean()) if out["possession_retained_+6s"].notna().any() else None,

        "delta_width_pct_+5s_mean": float(out["delta_width_pct_+5s"].dropna().mean()) if out["delta_width_pct_+5s"].notna().any() else None,
        "delta_compact_pct_+5s_mean": float(out["delta_compact_pct_+5s"].dropna().mean()) if out["delta_compact_pct_+5s"].notna().any() else None,

        "delta_team_speed_pct_0_6_vs_-3_0": float(out["team_speed_pct_0_6_vs_-3_0"].dropna().mean()) if out["team_speed_pct_0_6_vs_-3_0"].notna().any() else None,

        "n_opponents_within_3m_at_+3s_mean": float(out["n_opponents_within_3m_at_+3s"].dropna().mean()) if out["n_opponents_within_3m_at_+3s"].notna().any() else None,
        "n_opponents_within_5m_at_+5s_mean": float(out["n_opponents_within_5m_at_+5s"].dropna().mean()) if out["n_opponents_within_5m_at_+5s"].notna().any() else None,

        "team": team_short,
        "press_r1_m": float(press_r1_m),
        "press_r2_m": float(press_r2_m),
        "post_s_struct": float(post_s_struct),
        "post_s_possession": float(post_s_possession),
    }
    return out, summary
