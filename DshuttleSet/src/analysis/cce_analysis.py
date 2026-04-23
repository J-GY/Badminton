import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data_utils import df_filter_by_conditions
from scipy.stats import chi2_contingency 

def global_cce_analysis(df, match_id=None, 
                                set_num=None, 
                                start_rally_id=None, 
                                end_rally_id=None,
                                selected_type=['對手落地致勝'],
                                event_type=None,
                                shot_to_end=[1,2,3],
                                selected_formations='Attack'):
    
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
        
    required_cols = [
        'player', 'A', 'B', 'C', 'D', 'shot_num', 'shot_count',
        'match_Winner_formation', 'match_Loser_formation', 
        'match_Winner_mid_cover', 'match_Loser_mid_cover',
        'score_team', 'match_id', 'set_id', 'rally_id', 'set_win'
    ]
    
    if not all(col in df_filtered.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_filtered.columns]
        print(f"錯誤: 缺少必要欄位無法分析: {missing}。")
        return
    
    df_analysis = df_filtered.copy()
    
    # 篩選「陣型」
    df_selected_formation = df_analysis[df_analysis['player_formation'] == selected_formations].copy()
    
    if df_selected_formation.empty:
        print("警告: 找不到任何 'Attack' 陣型下的擊球數據。")
        return
    else:
        print(f"分析中: 共 {len(df_selected_formation)} 筆 {selected_formations} 陣型下的擊球數據。")

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
    stats_reaction = calculate_stats(df_selected_formation, 'is_reaction_cover')

    print("\n--- Reaction-Cover ---")
    print(stats_reaction)
    # (Contingency Table)
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
    
    print(f"\n[卡方檢定與效應量]")
    print(f"Chi-square: {chi2:.2f}")
    print(f"p-value: {p_val:.5f}")

    print(f"\n--- (Chi-Square Test) ---")
    print(f"p-value: {p_val:.5f}")
        
# --- 8. 繪圖 ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = ['Successful Compensation', 'Failed Compensation']
        success_rates = [stats_reaction.loc[True, 'success_rate'], stats_reaction.loc[False, 'success_rate']]
        colors = ['#34A853', '#EA4335'] # 綠(有), 紅(無)
        
        bars = ax.bar(labels, success_rates, color=colors, width=0.6)
        
        title_str = f'Court Coverage Defense Success Rate ({selected_formations} Formation)\n{filter_title_suffix} (p-value: {p_val:.4f} ***)'
        ax.set_title(title_str, fontsize=14, weight='bold', pad=15)
        ax.set_ylabel('Defense Success Rate (%)', fontsize=12, weight='bold')
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
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

        plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)

        plt.savefig('Court_Coverage.pdf', dpi=400)
        plt.show()
        
    except Exception as e:
        print(f"繪圖錯誤: {e}")