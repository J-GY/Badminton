import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from src.utils.data_utils import X_BINS, Y_BINS_BOTTOM, Y_BINS_TOP

def draw_badminton_court(ax, title=""):        
    # 外框
    ax.plot([0, 61], [0, 0], 'k-')
    ax.plot([0, 61], [134, 134], 'k-')
    ax.plot([0, 0], [0, 134], 'k-')
    ax.plot([61, 61], [0, 134], 'k-')
    
    # 球網
    ax.plot([0, 61], [67, 67], 'k--', alpha=0.8)
    
    # 雙打邊線
    ax.plot([4.6, 4.6], [0, 134], 'k-', alpha=0.6)
    ax.plot([57.4, 57.4], [0, 134], 'k-', alpha=0.6)
    
    # 前發球線
    ax.plot([0, 61], [46.6, 46.6], 'k-', alpha=0.6)
    ax.plot([0, 61], [87.4, 87.4], 'k-', alpha=0.6)
    
    # 雙打後發球線
    ax.plot([0, 61], [7.6, 7.6], 'k-', alpha=0.6)
    ax.plot([0, 61], [126.4, 126.4], 'k-', alpha=0.6)
    
    # 中線
    ax.plot([30.5, 30.5], [0, 46.6], 'k-', alpha=0.6)
    ax.plot([30.5, 30.5], [87.4, 134], 'k-', alpha=0.6)
    
    ax.set_xlim(-10, 71)
    ax.set_ylim(-10, 144)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

def _plot_stacked_bar(df_team, 
                      player_ids,          # <-- 傳入 ID, e.g., [101, 102]
                      player_names,        # <-- 傳入名稱, e.g., ['A', 'B']
                      team_name, colors, col, filter_title_suffix):
    """
    內部輔助函數：繪製單一隊伍的堆疊長條圖並標註百分比。
    (V2: 更新為使用 player_ids 和 player_names)
    """
    col_cleaned = col.strip()
    
    # 1. 計算「該圖表」的總擊球數 (用於計算總百分比)
    total_all_shots_in_plot = df_team[col].dropna().shape[0]
    if total_all_shots_in_plot == 0:
        print(f"跳過 {team_name} 的 '{col_cleaned}'：總擊球數為 0。")
        return

    # 2. 計算每個球員在每個球種上的次數
    counts_df = df_team.groupby([col, 'player']).size().unstack(fill_value=0)
    
    # 3. 確保兩位球員的 ID 欄位都存在
    if player_ids[0] not in counts_df: counts_df[player_ids[0]] = 0
    if player_ids[1] not in counts_df: counts_df[player_ids[1]] = 0
        
    # 4. 計算統計數據
    counts_df['bar_total'] = counts_df[player_ids[0]] + counts_df[player_ids[1]]
    
    counts_df['pct_of_all_shots'] = (counts_df['bar_total'] / total_all_shots_in_plot) * 100
    
    counts_df[f'pct_individual_{player_ids[0]}'] = (counts_df[player_ids[0]] / counts_df['bar_total']).fillna(0) * 100
    counts_df[f'pct_individual_{player_ids[1]}'] = (counts_df[player_ids[1]] / counts_df['bar_total']).fillna(0) * 100
    
    # 5. 決定要繪製的數據 (Top 20 或全部)
    if len(counts_df) > 20:
        counts_to_plot = counts_df.nlargest(20, 'bar_total')
        plot_title = f'{team_name} {col_cleaned} 分佈 (Top 20)\n{filter_title_suffix}'
    else:
        counts_to_plot = counts_df
        plot_title = f'{team_name} {col_cleaned} 分佈\n{filter_title_suffix}'
        
    # 6. 按總數排序
    counts_to_plot = counts_to_plot.sort_values(by='bar_total', ascending=False)

    # 7. 繪製堆疊長條圖
    ax = counts_to_plot[[player_ids[0], player_ids[1]]].plot(
        kind='bar', 
        stacked=True, 
        color=colors, 
        figsize=(10, 6)
    )
    
    # 8. 【核心】標註百分比
    ymax = counts_to_plot['bar_total'].max() * 1.15
    ax.set_ylim(top=ymax)
    min_height_for_text = ymax * 0.04 
    
    for i, ball_type in enumerate(counts_to_plot.index):
        count_p1 = counts_to_plot.loc[ball_type, player_ids[0]]
        count_p2 = counts_to_plot.loc[ball_type, player_ids[1]]
        
        pct_total = counts_to_plot.loc[ball_type, 'pct_of_all_shots']
        bar_total_height = count_p1 + count_p2
        ax.text(i, bar_total_height + (ymax * 0.01), 
                f'{pct_total:.1f}%', 
                ha='center', color='black', fontsize=9)
        
        pct_p1_indiv = counts_to_plot.loc[ball_type, f'pct_individual_{player_ids[0]}']
        y_pos_p1 = count_p1 / 2
        if count_p1 > min_height_for_text:
            ax.text(i, y_pos_p1, f'{pct_p1_indiv:.1f}%', 
                    ha='center', va='center', color='white', weight='bold', fontsize=8)
        
        pct_p2_indiv = counts_to_plot.loc[ball_type, f'pct_individual_{player_ids[1]}']
        y_pos_p2 = count_p1 + (count_p2 / 2)
        if count_p2 > min_height_for_text:
            ax.text(i, y_pos_p2, f'{pct_p2_indiv:.1f}%', 
                    ha='center', va='center', color='white', weight='bold', fontsize=8)

    # 9. 調整圖表外觀 
    ax.set_title(plot_title, fontsize=14)
    ax.set_ylabel('計次 (Count)')
    ax.set_xlabel(col_cleaned)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.legend(player_names, title='Player') 
    
    plot_filename = f'stacked_bar_{team_name}_{col_cleaned}_{filter_title_suffix}.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"堆疊長條圖 '{col_cleaned}' ({team_name}) 已儲存為: {plot_filename}")
    plt.show()

def draw_zone_heatmap(ax, zone_rates_series, cmap_name, title):
    """
    在 ax 上繪製分區著色熱圖。
    """
    # 繪製基礎球場 (含邊界)
    draw_badminton_court(ax, title)
    
    # 1. 準備顏色映射
    vmax = max(zone_rates_series.max(), 10.0) # 至少 10%
    if vmax > 50: vmax = 50 
    
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    for zone_id, patch_info in zone_patches.items():
        rate = zone_rates_series.get(zone_id, 0)
        text_color = 'white' if rate > (vmax * 0.6) else 'black'
        
        # 計算顏色
        if zone_id > 9: # 界外
            color = '#404040' 
            text_color = 'white'
        elif zone_id == -1: # 錯誤
            color = "#000000"
            text_color = 'white'
        else: # 界內
            color = cmap(norm(rate)) 
            
        label = f'{rate:.1f}%'
        
        if not (isinstance(patch_info, (tuple, list)) and len(patch_info) == 4):
            continue
            
        (x_min, y_min, x_max, y_max) = patch_info
        w = x_max - x_min
        h = y_max - y_min
        
        if w <= 0 or h <= 0:
            continue

        rect = Rectangle((x_min, y_min), w, h, color=color, alpha=0.8, ec='white', lw=0.5)
        ax.add_patch(rect)
        
        # 在中心添加標籤
        ax.text(x_min + w/2, y_min + h/2, label, 
                ha='center', va='center', color=text_color, fontsize=9, weight='bold')
            
zone_patches = {
    # 界外
    13: (-10, 134, 71, 144),
    14: (-10, 113, 0, 134),   12: (61, 113, 71, 134),
    15: (-10, 90, 0, 113),  11: (61, 90, 71, 113),
    16: (-10, 67, 0, 90),  10: (61, 67, 71, 90),
    
    # 下半場
    2: (0, 45, 20, 67),   7: (20, 45, 41, 67),   1: (41, 45, 61, 67),
    6: (0, 22, 20, 45),   8: (20, 22, 41, 45),   5: (41, 22, 61, 45),
    4: (0, 0, 20, 22),      9: (20, 0, 41, 22),      3: (41, 0, 61, 22),
    
    # Zone -1 錯誤區域
    -1: (0, 67.1, 61, 134)
}
