import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from src.utils.data_utils import df_filter_by_conditions, normalize_coordinates, get_zone_id
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.stats import chi2_contingency

zone_patches = {
    # out of court zones
    13: (-10, 134, 71, 144),
    14: (-10, 113, 0, 134),   12: (61, 113, 71, 134),
    15: (-10, 90, 0, 113),  11: (61, 90, 71, 113),
    16: (-10, 67, 0, 90),  10: (61, 67, 71, 90),
    
    # downcourt zones
    2: (0, 45, 20, 67),   7: (20, 45, 41, 67),   1: (41, 45, 61, 67),
    6: (0, 22, 20, 45),   8: (20, 22, 41, 45),   5: (41, 22, 61, 45),
    4: (0, 0, 20, 22),      9: (20, 0, 41, 22),      3: (41, 0, 61, 22),
    
    # error Zone -1 
    -1: (0, 67.1, 61, 134)
}

def draw_court_background_green(ax, full_wid=61.0, half_len=67.0):
    COURT_COLOR = '#90C080' 
    LINE_COLOR = 'white'        
    lw = 2 

    ax.set_facecolor(COURT_COLOR)

    rect = Rectangle((0, 0), full_wid, half_len, linewidth=lw, edgecolor=LINE_COLOR, facecolor='none', zorder=1)
    ax.add_patch(rect)
    
    margin = 4.6
    ax.plot([margin, margin], [0, half_len], color=LINE_COLOR, linewidth=lw, zorder=1)
    ax.plot([full_wid-margin, full_wid-margin], [0, half_len], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    long_service_dist = 7.6
    ax.plot([0, full_wid], [long_service_dist, long_service_dist], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    short_service_y = half_len - 19.8
    ax.plot([0, full_wid], [short_service_y, short_service_y], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    mid_x = full_wid / 2
    ax.plot([mid_x, mid_x], [0, short_service_y], color=LINE_COLOR, linewidth=lw, zorder=1)

    text_style = {'color': 'white', 'fontsize': 10, 'fontweight': 'bold', 'zorder': 5}
    offset = 1.0
    ax.text(0 - offset, 0 - offset, '(0,0)', ha='right', va='top', **text_style)
    ax.text(full_wid + offset, 0 - offset, '(61,0)', ha='left', va='top', **text_style)
    ax.text(0 - offset, half_len + offset, '(0,67)', ha='right', va='bottom', **text_style)

def draw_failure_heatmap_green(ax, failure_rate, total_attempts, total_failures, title, cmap_name='Reds'):
    draw_court_background_green(ax, full_wid=61.0, half_len=67.0)
    
    valid_zones = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    norm = Normalize(vmin=0, vmax=12) 
    cmap = plt.get_cmap(cmap_name)
    
    for zid in valid_zones:
        rate = failure_rate.get(zid, 0)
        count = total_attempts.get(zid, 0)
        fail_count = total_failures.get(zid, 0)
        
        if zid not in zone_patches: continue
        patch_data = zone_patches[zid]
        if len(patch_data) == 4:
            x, y, x2, y2 = patch_data
            w, h = x2-x, y2-y
            color = cmap(norm(rate))
            
            rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='none', alpha=0.85, zorder=0.5)
            ax.add_patch(rect)
            
            if count > 0:
                label = f"{rate:.1f}%\n({int(fail_count)}/{int(count)})"
                
                text_color = 'white' if rate > 7 else 'black' 
                
                ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                        color=text_color, fontsize=10, weight='bold', zorder=6)
                        
    ax.set_title(title, fontsize=14, fontweight='bold', pad=5)
    ax.set_xlim(-8, 69)
    ax.set_ylim(-2, 70)
    ax.set_aspect('equal')
    ax.axis('off')

def analyze_defensive_failure_rate(df, match_id=None, set_num=None, 
                                   start_rally_id=None, end_rally_id=None, event_type=None):
    try:
        df_filtered, filter_suffix = df_filter_by_conditions(
            df, match_id=match_id, set_num=set_num, 
            start_rally_id=start_rally_id, end_rally_id=end_rally_id, event_type=event_type
        )
    except NameError:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        print("No matching data found.")
        return
        
    cols = ['shot_num', 'shot_count', 'score_team', 'set_win', 'match_id', 'player', 'A', 'B', 'C', 'D']
    for c in cols:
        if c in df_filtered.columns: df_filtered[c] = pd.to_numeric(df_filtered[c], errors='coerce')
    
    df_filtered = df_filtered[df_filtered['shot_count'] > 1]

    reason_col = 'lose_reason'
    if reason_col not in df_filtered.columns and 'lose_reason_x' in df_filtered.columns:
        reason_col = 'lose_reason_x'
        
    if reason_col in df_filtered.columns:
        df_filtered[reason_col] = df_filtered[reason_col].astype(str).str.strip()

    last_shots = df_filtered[df_filtered['shot_num'] == df_filtered['shot_count']].copy()
    if 'score_team' not in last_shots.columns and 'score_team_x' in last_shots.columns:
        last_shots['score_team'] = last_shots['score_team_x']
        
    set_winners = last_shots.groupby(['match_id', 'set_num', 'score_team']).size().unstack(fill_value=0)
    if 0.0 not in set_winners.columns: set_winners[0.0] = 0
    if 1.0 not in set_winners.columns: set_winners[1.0] = 0
    
    match_win_counts = {}
    for idx, row in set_winners.iterrows():
        mid = idx[0]
        s_winner = 0.0 if row[0.0] > row[1.0] else 1.0
        if mid not in match_win_counts: match_win_counts[mid] = {0.0: 0, 1.0: 0}
        match_win_counts[mid][s_winner] += 1
    match_winner_map = {mid: (0.0 if c[0.0] > c[1.0] else 1.0) for mid, c in match_win_counts.items()}

    def get_shot_context(row):
        pid = row['player']
        if pid == row['A'] or pid == row['B']: hitter_team = 0.0
        else: hitter_team = 1.0
        
        receiver_team = 1.0 - hitter_team
        
        m_winner = match_winner_map.get(row['match_id'], 0.0)
        if receiver_team == m_winner: recv_fmt = row['match_Winner_formation']
        else: recv_fmt = row['match_Loser_formation']
        
        is_failure = False
        if row['shot_num'] == row['shot_count']:
            set_win = row['set_win']
            score = row.get('score_team', row.get('score_team_x'))
            
            if not pd.isna(set_win) and not pd.isna(score):
                if (set_win == 0 and score == 0) or (set_win == 1 and score == 1):
                    rally_winner = 0.0
                else:
                    rally_winner = 1.0
                
                if rally_winner != receiver_team:
                    reason = str(row.get(reason_col, '')).strip()
                    if reason != '犯規' and reason != '' and reason.lower() != 'nan':
                        if reason in ['對手落地致勝', '落地致勝']:
                            is_failure = True

        norm_param = 'AB' if receiver_team == 0.0 else 'CD'
        return pd.Series([recv_fmt, is_failure, norm_param])

    df_shots = df_filtered.copy()
    df_shots[['recv_fmt', 'is_failure', 'norm_param']] = df_shots.apply(get_shot_context, axis=1)

    norm_list = []
    for team_param in ['AB', 'CD']:
        sub = df_shots[df_shots['norm_param'] == team_param].copy()
        if not sub.empty:
            sub = normalize_coordinates(sub, team=team_param)
            norm_list.append(sub)
            
    if not norm_list: 
        print("No valid coordinate data found.")
        return
    df_norm = pd.concat(norm_list)

    df_norm['heatmap_x'] = df_norm['return_x']
    df_norm['heatmap_y'] = df_norm['return_y']
    df_norm = df_norm.dropna(subset=['heatmap_x', 'heatmap_y'])
    
    try:
        df_norm['zone_id'] = df_norm.apply(lambda r: get_zone_id(r['heatmap_x'], r['heatmap_y']), axis=1)
    except NameError:
        print("Error: get_zone_id function not found")
        return
    
    all_zones = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    attempts_grp = df_norm.groupby(['recv_fmt', 'zone_id']).size()
    failures_grp = df_norm[df_norm['is_failure']].groupby(['recv_fmt', 'zone_id']).size()
    
    def get_rates(fmt):
        if fmt == 'Overall':
            att = df_norm.groupby('zone_id').size().reindex(all_zones, fill_value=0)
            fail = df_norm[df_norm['is_failure']].groupby('zone_id').size().reindex(all_zones, fill_value=0)
        else:
            att = attempts_grp.get(fmt, pd.Series()).reindex(all_zones, fill_value=0)
            fail = failures_grp.get(fmt, pd.Series()).reindex(all_zones, fill_value=0)
        
        rate = (fail / att * 100).fillna(0)
        return rate, att, fail

    rate_attack, att_attack, fail_attack = get_rates('Attack')
    rate_defense, att_defense, fail_defense = get_rates('Defense')
    rate_overall, att_overall, fail_overall = get_rates('Overall')
    
    def calc_overall_rate(fail, att):
        total_fail = fail.sum()
        total_att = att.sum()
        return (total_fail / total_att * 100) if total_att > 0 else 0.0

    ovr_rate_attack = calc_overall_rate(fail_attack, att_attack)
    ovr_rate_defense = calc_overall_rate(fail_defense, att_defense)
    ovr_rate_overall = calc_overall_rate(fail_overall, att_overall)

    if event_type == 1:
        event_title = "Men's Doubles"
        file_suffix = "MD"
    elif event_type == 2:
        event_title = "Women's Doubles"
        file_suffix = "WD"
    elif event_type == 3:
        event_title = "Mixed Doubles"
        file_suffix = "XD"
    else:
        event_title = "Overall (MD & WD)"
        file_suffix = "All"

    # 2. 定義單張圖表輸出函數
    def save_single_heatmap(rate_data, att_data, fail_data, title_text, filename):
        fig, ax = plt.subplots(figsize=(6, 6)) 
        
        draw_failure_heatmap_green(ax, rate_data, att_data, fail_data, title_text, 'Reds')
        
        fig.patch.set_facecolor('#90C080')
        plt.tight_layout()
        os.makedirs('./img/Coordination Analysis', exist_ok=True)
        plt.savefig(f'./img/Coordination Analysis/{filename}', dpi=400)
        plt.show() 

    save_single_heatmap(
        rate_attack, att_attack, fail_attack,
        f"{event_title} - Offensive ({ovr_rate_attack:.1f}%)",
        f'defensive_vulnerability_offensive_{file_suffix}.pdf'
    )

    save_single_heatmap(
        rate_defense, att_defense, fail_defense,
        f"{event_title} - Defensive ({ovr_rate_defense:.1f}%)",
        f'defensive_vulnerability_defensive_{file_suffix}.pdf'
    )

    save_single_heatmap(
        rate_overall, att_overall, fail_overall,
        f"{event_title} - Overall ({ovr_rate_overall:.1f}%)",
        f'defensive_vulnerability_overall_{file_suffix}.pdf'
    )