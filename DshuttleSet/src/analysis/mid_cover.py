import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from src.utils.data_utils import df_filter_by_conditions, normalize_coordinates, get_zone_id
from src.visualization.plot_utils import _plot_stacked_bar, draw_badminton_court, draw_zone_heatmap

def midcover_analysis(df, match_id=None, 
                                set_num=None, 
                                start_rally_id=None, 
                                end_rally_id=None):
    """
    分析「攻擊陣型下的中位覆蓋防守成功率」。
    V2:
    - 根據 'match_Winner_formation'/'match_Loser_formation' 判斷擊球者陣型
    - 根據 'match_Winner_mid_cover'/'match_Loser_mid_cover' 判斷擊球者中位覆蓋
    """
    
    # --- 1. 執行 "通用" 篩選 ---
    df_filtered, filter_title_suffix = df_filter_by_conditions(
        df,
        match_id=match_id,
        set_num=set_num,
        start_rally_id=start_rally_id,
        end_rally_id=end_rally_id,
        player_id=None, 
        team=None       
    )

    if df_filtered.empty:
        print(f"在指定的篩選條件下 ({filter_title_suffix}) 找不到任何數據。")
        return
        
    print(f"--- 顯示中位覆蓋防守率分析報告: {filter_title_suffix} ---")

    # --- 2. 檢查必要欄位 ---
    # 【修正 1】: 添加您指定的 'mid_cover' 欄位
    required_cols = [
        'player', 'A', 'B', 'C', 'D', 'shot_num', 'shot_count',
        'match_Winner_formation', 'match_Loser_formation', 
        'match_Winner_mid_cover', 'match_Loser_mid_cover',
        'score_team' 
    ]
    
    if not all(col in df_filtered.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_filtered.columns]
        print(f"錯誤: 缺少必要欄位無法分析: {missing}。")
        return
        
    # --- 3. 提取 A, B, C, D 四位球員的 ID ---
    try:
        player_id_A = float(df_filtered['A'].iloc[0])
        player_id_B = float(df_filtered['B'].iloc[0])
        player_id_C = float(df_filtered['C'].iloc[0])
        player_id_D = float(df_filtered['D'].iloc[0])
    except (IndexError, ValueError) as e:
        print(f"錯誤: 無法獲取 A/B/C/D player ID 或轉換為 float: {e}")
        return
        
    # 將 Player ID 映射到 Team ID (0.0 = A/B, 1.0 = C/D)
    player_id_to_team = {
        player_id_A: 0.0, player_id_B: 0.0,
        player_id_C: 1.0, player_id_D: 1.0
    }
    
    df_analysis = df_filtered.copy()
    
    # --- 4. 準備分析欄位 ---
    df_analysis['player'] = pd.to_numeric(df_analysis['player'], errors='coerce')
    df_analysis['score_team'] = pd.to_numeric(df_analysis['score_team'], errors='coerce')
    df_analysis['shot_num'] = pd.to_numeric(df_analysis['shot_num'], errors='coerce')
    df_analysis['shot_count'] = pd.to_numeric(df_analysis['shot_count'], errors='coerce')
    
    # 【修正 2】: 轉換您指定的中位覆蓋欄位為布林值
    df_analysis['match_Winner_mid_cover'] = df_analysis['match_Winner_mid_cover'].astype(str).str.lower().isin(['true', '1'])
    df_analysis['match_Loser_mid_cover'] = df_analysis['match_Loser_mid_cover'].astype(str).str.lower().isin(['true', '1'])

    # 4a. 獲取擊球者所屬的隊伍 (0 或 1)
    df_analysis['player_team'] = df_analysis['player'].map(player_id_to_team)
    
    # 4b. 對手的隊伍是否贏了這回合 (即我方防守失敗)
    df_analysis['did_opponent_team_win'] = (df_analysis['player_team'] != df_analysis['score_team'])
    
    # 4c. 判斷這拍距離回合結束還有幾拍 
    df_analysis['shots_to_end'] = df_analysis['shot_count'] - df_analysis['shot_num']
    
    # 4d. 定義「防守失敗」
    df_analysis['is_defense_failure'] = ( 
        (df_analysis['did_opponent_team_win']) &
        (df_analysis['shots_to_end'].isin([1,2,3]))
    )
    
    # --- 5. 【關鍵修正 3】: 建立 'player_formation' 和 'player_mid_cover' ---
    # 找出整場比賽的 Winner Team ID (0.0 或 1.0)
    try:
        # 找出最後一拍
        df_last_shots = df_analysis[df_analysis['shot_num'] == df_analysis['shot_count']]
        # 統計得分
        team_0_wins = (df_last_shots['score_team'] == 0).sum()
        team_1_wins = (df_last_shots['score_team'] == 1).sum()
        
        winner_team_id = 0.0
            
        loser_team_id = 1.0
        
        print(f"分析: Team {winner_team_id} (A/B) 被識別為 Winner, Team {loser_team_id} (C/D) 被識別為 Loser。")

    except Exception as e:
        print(f"錯誤: 無法判斷整場比賽的 Winner (基於 score_team): {e}")
        return

    # A. 建立 'player_formation'
    # 如果擊球者 (player_team) 是 Winner (winner_team_id)，則使用 match_Winner_formation
    df_analysis['player_formation'] = np.where(
        df_analysis['player_team'] == winner_team_id, 
        df_analysis['match_Winner_formation'], 
        df_analysis['match_Loser_formation']
    )
    
    # B. 建立 'player_mid_cover'
    # 如果擊球者 (player_team) 是 Winner (winner_team_id)，則使用 match_Winner_mid_cover
    df_analysis['player_mid_cover'] = np.where(
        df_analysis['player_team'] == winner_team_id, 
        df_analysis['match_Winner_mid_cover'], 
        df_analysis['match_Loser_mid_cover']
    )

    # --- 6. 執行計算 ---
    
    # 6a. **篩選**: 我們只關心「攻擊陣型」下的擊球
    df_attack = df_analysis[df_analysis['player_formation'] == 'Attack'].copy()
    
    if df_attack.empty:
        print("警告: 找不到任何 'Attack' 陣型下的擊球數據。")
        return

    # 6b. **計算分母 (Denominator)**: 在攻擊陣型下，不同中位覆蓋的總擊球數
    #    (【修正 4】: groupby 'player_mid_cover' 而不是 'is_midcover')
    total_attempts = df_attack.groupby(
        ['player_team', 'player_mid_cover']
    ).size().rename('total_attempts')
    
    # 6c. **計算分子 (Numerator)**: 在攻擊陣型下，防守失敗的總次數
    total_failures = df_attack[df_attack['is_defense_failure'] == True].groupby(
        ['player_team', 'player_mid_cover']
    ).size().rename('total_failures')

    # 6d. **合併與計算**
    stats = pd.concat([total_attempts, total_failures], axis=1).fillna(0)
    
    # 避免除以零
    stats['failure_rate'] = stats.apply(
        lambda row: (row['total_failures'] / row['total_attempts']) * 100 if row['total_attempts'] > 0 else 0,
        axis=1
    )
    stats['success_rate'] = 100 - stats['failure_rate']

    # --- 7. 準備繪圖數據 ---
    # 重設索引以便繪圖 (索引是 'player_mid_cover')
    plot_data = stats['success_rate'].unstack(level='player_team').fillna(0)
    
    # 幫 Team ID 命名
    team_map = {
        0.0: 'Team A/B',
        1.0: 'Team C/D'
    }
    plot_data = plot_data.rename(columns=team_map)
    
    # 確保 True (有覆蓋) 和 False (無覆蓋) 索引都存在
    if True not in plot_data.index:
        plot_data.loc[True] = 0.0
    if False not in plot_data.index:
        plot_data.loc[False] = 0.0
        
    plot_data = plot_data.reindex([True, False])
    plot_data.index = ['中位有覆蓋 (True)', '中位無覆蓋 (False)']

    print("\n--- 分析結果 (防守成功率) ---")
    print(stats)
    print(plot_data)
    
    bool_indices = [True, False]
    # --- 8. 繪製兩張圖表 ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        (ax1, ax2) = axes.flatten()
        
        cover_colors = ["#4285F4", '#DB4437'] # 藍色=有覆蓋, 紅色=無覆蓋
        
        # 繪製 Team A/B
        if 'Team A/B' in plot_data.columns:
            team_data_ab = plot_data['Team A/B']
            team_data_ab.plot(kind='bar', ax=ax1, color=cover_colors, width=0.7)
            
            ax1.set_title(f'Team A/B 防守成功率 (當處於攻擊陣型時)', fontsize=14)
            ax1.set_ylabel('防守成功率 (%)', fontsize=12)
            ax1.set_ylim(0, 115) # 增加一點高度給標籤
            ax1.set_xticklabels(plot_data.index, rotation=0)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加標籤
            for i, val in enumerate(team_data_ab):
                # 1. 百分比 (頂部)
                ax1.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=11, weight='bold')
                
                # 2. 【新增】 失敗數/總數 (中間)
                current_cover_status = bool_indices[i]
                try:
                    # 從原始 stats 獲取數據
                    fail = int(stats.loc[(0.0, current_cover_status), 'total_failures'])
                    att = int(stats.loc[(0.0, current_cover_status), 'total_attempts'])
                    label_text = f"{att-fail}/{att}"
                except KeyError:
                    label_text = "0/0"
                
                # 寫在 bar 的中間 (val/2)，如果 bar 太矮，顯示在稍微上面一點避免重疊
                y_pos = val / 2 if val > 10 else val + 8 
                text_color = 'white' if val > 20 else 'black' # 如果 bar 很深，字用白色
                ax1.text(i, y_pos, label_text, ha='center', va='center', fontsize=10, color=text_color, weight='bold')

        # ---------------- Team C/D (1.0) ----------------
        if 'Team C/D' in plot_data.columns:
            team_data_cd = plot_data['Team C/D']
            team_data_cd.plot(kind='bar', ax=ax2, color=cover_colors, width=0.7)
            
            ax2.set_title(f'Team C/D 防守成功率 (當處於攻擊陣型時)', fontsize=14)
            ax2.set_xticklabels(plot_data.index, rotation=0)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

            for i, val in enumerate(team_data_cd):
                # 1. 百分比
                ax2.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=11, weight='bold')
                
                # 2. 【新增】 失敗數/總數
                current_cover_status = bool_indices[i]
                try:
                    fail = int(stats.loc[(1.0, current_cover_status), 'total_failures'])
                    att = int(stats.loc[(1.0, current_cover_status), 'total_attempts'])
                    label_text = f"{att-fail}/{att}"
                except KeyError:
                    label_text = "0/0"
                
                y_pos = val / 2 if val > 10 else val + 8
                text_color = 'white' if val > 20 else 'black'
                ax2.text(i, y_pos, label_text, ha='center', va='center', fontsize=10, color=text_color, weight='bold')

        fig.suptitle(f'攻擊陣型下的中位覆蓋防守成功率\n防守成功次數/總防守次數\n{filter_title_suffix}', fontsize=16, y=1.05)
        fig.tight_layout()
        
        plot_filename = f'midcover_defense_success_rate_{filter_title_suffix.replace(" ", "_").replace(":", "")}.png'
        fig.savefig(plot_filename, bbox_inches='tight', dpi=150)
        plt.show()
        print(f"中位覆蓋分析圖表已儲存為: {plot_filename}")

    except Exception as e:
        print(f"繪圖時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

from scipy.stats import chi2_contingency # 新增統計模組

def global_midcover_analysis(df, match_id=None, 
                                set_num=None, 
                                start_rally_id=None, 
                                end_rally_id=None,
                                selected_type=['對手落地致勝'],
                                event_type=None,
                                shot_to_end=[1,2,3]):
    """    
    邏輯更新:
    1. 使用 set_win 精確判斷 Team 0 (Winner) 與 Team 1 (Loser)。
    2. 只計算「反應補位」: 當我方攻擊(N拍)時，下一拍(N+1拍)對手回球當下，我方中位是否有補位。
    3. 防守失敗判定加入「致勝球」篩選 (例如: 對手落地致勝)。
    4. 【新增】整合卡方檢定 (Chi-Square Test) 以驗證補位與失分是否有顯著關聯。
    """
    
    df_filtered, filter_title_suffix = df_filter_by_conditions(
        df,
        match_id=match_id,
        set_num=set_num,
        start_rally_id=start_rally_id,
        end_rally_id=end_rally_id,
        player_id=None, 
        team=None,
        event_type=event_type    
    )

    if df_filtered.empty:
        print(f"在指定的篩選條件下 ({filter_title_suffix}) 找不到任何數據。")
        return
        
    print(f"--- 顯示反應補位防守效益分析 (Global Reaction Cover): {filter_title_suffix} ---")

    # --- 2. 檢查必要欄位 ---
    required_cols = [
        'player', 'A', 'B', 'C', 'D', 'shot_num', 'shot_count',
        'match_Winner_formation', 'match_Loser_formation', 
        'match_Winner_mid_cover', 'match_Loser_mid_cover',
        'score_team', 'match_id', 'set_id', 'rally_id', 'set_win'
    ]
    
    # 相容性處理 (欄位名稱可能帶有 _x)
    if 'score_team' not in df_filtered.columns and 'score_team_x' in df_filtered.columns:
        df_filtered['score_team'] = df_filtered['score_team_x']
    
    # 嘗試尋找失分原因欄位 (lose_reason 或 lose_reason_x)
    lose_reason_col = 'lose_reason'

    if lose_reason_col:
        required_cols.append(lose_reason_col)
    else:
        print("警告: 找不到 'lose_reason' 欄位，將無法進行致勝球的精確篩選。")

    if not all(col in df_filtered.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_filtered.columns]
        print(f"錯誤: 缺少必要欄位無法分析: {missing}。")
        return

    df_analysis = df_filtered.copy()
    
    # --- 3. 準備分析欄位 ---
    cols_to_numeric = ['player', 'score_team', 'shot_num', 'shot_count', 'match_id', 'A', 'B', 'C', 'D', 'set_win']
    for col in cols_to_numeric:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')

    # 確保排序正確 (Match -> Set -> Rally -> Shot)
    df_analysis = df_analysis.sort_values(by=['match_id', 'set_id', 'rally_id', 'shot_num'])

    # 轉換中位覆蓋欄位為布林值
    df_analysis['match_Winner_mid_cover'] = df_analysis['match_Winner_mid_cover'].astype(str).str.lower().isin(['true', '1'])
    df_analysis['match_Loser_mid_cover'] = df_analysis['match_Loser_mid_cover'].astype(str).str.lower().isin(['true', '1'])

    # 4a. 判斷擊球者所屬隊伍 (使用新邏輯)
    # 0.0 對應 team_Winner_...
    # 1.0 對應 team_Loser_...
    def get_player_team_logic(row):
        pid = row['player']
        try:
            set_win = int(row['set_win'])
        except:
            return np.nan
            
        if set_win == 0:
            # Team 0 = A/B (Winner), Team 1 = C/D (Loser)
            if pid == row['A'] or pid == row['B']:
                return 0.0
            elif pid == row['C'] or pid == row['D']:
                return 1.0
        elif set_win == 1:
            # Team 0 = C/D (Winner), Team 1 = A/B (Loser)
            # 注意: 這裡的 0.0/1.0 是為了對應 team_Winner/team_Loser 欄位
            if pid == row['A'] or pid == row['B']:
                return 1.0 # A/B 是輸家 (對應 Team 1)
            elif pid == row['C'] or pid == row['D']:
                return 0.0 # C/D 是贏家 (對應 Team 0)
        return np.nan

    df_analysis['player_team'] = df_analysis.apply(get_player_team_logic, axis=1)

    # 4b. 防守失敗判定 (包含致勝球篩選)
    # 如果 score_team (rally winner) != player_team，代表我方輸了這一球
    df_analysis['did_opponent_team_win'] = (df_analysis['player_team'] != df_analysis['score_team'])
    df_analysis['shots_to_end'] = df_analysis['shot_count'] - df_analysis['shot_num']
    
    # 判斷是否為指定球種得分
    is_lose_type = False
    if lose_reason_col:
        # 清理字串並檢查
        df_analysis[lose_reason_col] = df_analysis[lose_reason_col].astype(str).str.strip()
        is_lose_type = df_analysis[lose_reason_col].isin(selected_type)

    # 失敗定義更新：
    # 1. 對手贏了這一球 AND
    # 2. (原本邏輯: 3拍內結束) AND (新邏輯: 是被對手打出落地致勝球)
    df_analysis['is_defense_failure'] = ( 
        (df_analysis['did_opponent_team_win']) &
        (
            (df_analysis['shots_to_end'].isin(shot_to_end)) &
            (is_lose_type)
        )
    )
    
    # --- 5. 建立「陣型」與「反應補位」 ---
    
    # 5a. 基礎陣型 (當前 Row N)
    df_analysis['player_formation'] = np.where(
        df_analysis['player_team'] == 0.0, 
        df_analysis['match_Winner_formation'], 
        df_analysis['match_Loser_formation']
    )
    
    # 5b. 計算「反應補位 (Reaction-Cover)」(Row N+1)
    # Shift 資料，獲取下一拍的資訊
    df_analysis['next_rally_id'] = df_analysis['rally_id'].shift(-1)
    df_analysis['next_winner_mid_cover'] = df_analysis['match_Winner_mid_cover'].shift(-1)
    df_analysis['next_loser_mid_cover'] = df_analysis['match_Loser_mid_cover'].shift(-1)
    
    # 計算 Reaction Cover
    conditions = [
        (df_analysis['rally_id'] != df_analysis['next_rally_id']), # 最後一拍 -> 無效
        (df_analysis['player_team'] == 0.0), # 我是 Team 0 -> 看下一拍的 Winner Cover
        (df_analysis['player_team'] == 1.0)  # 我是 Team 1 -> 看下一拍的 Loser Cover
    ]
    
    choices = [
        np.nan, # 最後一拍設為 NaN
        df_analysis['next_winner_mid_cover'],
        df_analysis['next_loser_mid_cover']
    ]
    
    df_analysis['is_reaction_cover'] = np.select(conditions, choices, default=np.nan)

    # --- 6. 執行統計 ---
    
    # 篩選「攻擊陣型」
    df_attack = df_analysis[df_analysis['player_formation'] == 'Attack'].copy()
    
    if df_attack.empty:
        print("警告: 找不到任何 'Attack' 陣型下的擊球數據。")
        return
    else:
        print(f"分析中: 共 {len(df_attack)} 筆 'Attack' 陣型下的擊球數據。")

    def calculate_stats(data, cover_col_name):
        # 排除 NaN
        valid_data = data.dropna(subset=[cover_col_name])
        valid_data[cover_col_name] = valid_data[cover_col_name].astype(bool)
        
        total = valid_data.groupby(cover_col_name).size()
        fails = valid_data[valid_data['is_defense_failure'] == True].groupby(cover_col_name).size()
        
        stats_df = pd.concat([total, fails], axis=1).fillna(0)
        stats_df.columns = ['total_attempts', 'total_failures']
        
        stats_df['failure_rate'] = (stats_df['total_failures'] / stats_df['total_attempts']) * 100
        stats_df['success_rate'] = 100 - stats_df['failure_rate']
        
        for status in [True, False]:
            if status not in stats_df.index:
                stats_df.loc[status] = [0, 0, 0, 0]
        
        return stats_df.reindex([True, False])

    # 只計算反應補位
    stats_reaction = calculate_stats(df_attack, 'is_reaction_cover')

    print("\n--- 反應補位 (Reaction-Cover) 統計 ---")
    print(stats_reaction)

    # --- 7. 統計檢定 (卡方獨立性檢定) ---
    try:
        # 建構列聯表 (Contingency Table)
        #              失敗 (Fail)   成功 (Success)
        # 有補位 (True)   a             b
        # 無補位 (False)  c             d
        
        fail_covered = int(stats_reaction.loc[True, 'total_failures'])
        total_covered = int(stats_reaction.loc[True, 'total_attempts'])
        success_covered = total_covered - fail_covered
        
        fail_no_cover = int(stats_reaction.loc[False, 'total_failures'])
        total_no_cover = int(stats_reaction.loc[False, 'total_attempts'])
        success_no_cover = total_no_cover - fail_no_cover
        
        obs = np.array([
            [fail_covered, success_covered], 
            [fail_no_cover, success_no_cover]
        ])
        
        # 執行卡方檢定
        chi2, p_val, dof, expected = chi2_contingency(obs)
        
        print(f"\n--- 統計檢定結果 (Chi-Square Test) ---")
        print(f"p-value: {p_val:.5f}")
        
        sig_text = ""
        if p_val < 0.05:
            sig_text = "(顯著相關 *)"
            print(">> 結果顯著：反應補位與否與防守結果有顯著關聯 (p < 0.05)。")
        else:
            sig_text = "(不顯著)"
            print(">> 結果不顯著：無法證明反應補位與防守結果有關 (p >= 0.05)。")
    except Exception as e:
        print(f"統計檢定執行錯誤: {e}")
        p_val = 1.0
        sig_text = "(檢定失敗)"

    # --- 8. 繪圖 ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 準備數據
        labels = ['Successful Compensation', 'Failed Compensation']
        success_rates = [stats_reaction.loc[True, 'success_rate'], stats_reaction.loc[False, 'success_rate']]
        colors = ['#34A853', '#EA4335'] # 綠(有), 紅(無)
        
        bars = ax.bar(labels, success_rates, color=colors, width=0.6)
        
        title_str = f'Court Coverage Defense Success Rate (Offensive Formation)\n{filter_title_suffix} (p-value: {p_val:.4f} ***)'
        ax.set_title(title_str, fontsize=14, weight='bold')
        ax.set_ylabel('Defense Success Rate (%)', fontsize=12)
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 標註數據
        for i, bar in enumerate(bars):
            height = bar.get_height()
            status = [True, False][i]
            
            # 1. 成功率
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold')
            
            # 2. 成功數/總數
            fail = int(stats_reaction.loc[status, 'total_failures'])
            att = int(stats_reaction.loc[status, 'total_attempts'])
            success = att - fail
            label_text = f"Success: {success} / {att}"
            
            y_pos = height / 2 if height > 10 else height + 5
            text_color = 'white' if height > 20 else 'black'
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    label_text, ha='center', va='center', fontsize=11, color=text_color, weight='bold')

        plt.tight_layout()
        plot_filename = f'./img/midcover/global_midcover_reaction_{filter_title_suffix.replace(" ", "_")}.png'
        fig.savefig(plot_filename, dpi=150)
        plt.show()
        print(f"圖表已儲存為: {plot_filename}")
        
    except Exception as e:
        print(f"繪圖錯誤: {e}")
        import traceback
        traceback.print_exc()