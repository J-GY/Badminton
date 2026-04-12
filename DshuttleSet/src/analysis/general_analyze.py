import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

def plot_badminton_shot_analysis(
    df_raw, 
    z_axis_max_limit=None, 
    column_map=None,
    zone_map=None,
    shottype_map=None,
    save_dir_3d='./img/3dplot',
    save_dir_bar='./img/angle_grouped'
):
    """
    Analyze badminton data and plot 3D scatter plot and grouped bar chart
    
    Args:
        df_raw (pd.DataFrame): 包含原始數據的 DataFrame
        z_axis_max_limit (int, optional): 3D圖 Z軸的最大值。若未提供，將依據數據自動調整。
        column_map (dict, optional): 欄位對應表
        zone_map (dict, optional): 場區對應表
        shottype_map (dict, optional): 球種對應表
        save_dir_3d (str): 3D 圖表存檔目錄
        save_dir_bar (str): 柱狀圖存檔目錄
    """
    
    if column_map is None:
        column_map = {
            'Angle': 'Partner_Angle',
            'Zone': 'Hitting_Zone', 
            'ShotType_Numeric': 'ball_type' 
        }

    if zone_map is None:
        zone_map = {
            1: 'Forecourt', 2: 'Forecourt', 7: 'Forecourt',
            5: 'Midcourt', 6: 'Midcourt', 8: 'Midcourt',
            3: 'Backcourt', 4: 'Backcourt', 9: 'Backcourt',
            '1': 'Forecourt', '2': 'Forecourt', '7': 'Forecourt',
            '5': 'Midcourt', '6': 'Midcourt', '8': 'Midcourt',
            '3': 'Backcourt', '4': 'Backcourt', '9': 'Backcourt'
        }

    if shottype_map is None:
        shottype_map = {
            # '殺球': '攻擊性擊球', 
            # '推撲球': '攻擊性擊球', 
            # '切球': '攻擊性擊球',
            # '網前小球': '防守性擊球', 
            # '長球': '防守性擊球', 
            # '平球': '防守性擊球', 
            # '挑球': '防守性擊球',
            '殺球': 'Attack Shot', 
            '推撲球': 'Attack Shot', 
            '切球': 'Attack Shot',
            '網前小球': 'Defense Shot', 
            '長球': 'Defense Shot', 
            '平球': 'Defense Shot', 
            '挑球': 'Defense Shot',
        }

    Y_CATEGORIES = ['Forecourt', 'Midcourt', 'Backcourt']
    # COLOR_CATEGORIES = ['攻擊性擊球', '防守性擊球']
    COLOR_CATEGORIES = ['Attack Shot', 'Defense Shot']
    y_mapping = {cat: i for i, cat in enumerate(Y_CATEGORIES)}
    # colors_list = {'攻擊性擊球': 'red', '防守性擊球': 'blue'} 
    colors_list = {'Attack Shot': 'red', 'Defense Shot': 'blue'} 

    try:
        font_path = fm.findfont(fm.FontProperties(family='Microsoft JhengHei'))
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        print("Unable to find 'Microsoft JhengHei', the Chinese characters in the chart may be displayed as boxes.")
    plt.rcParams['axes.unicode_minus'] = False 

    try:
        df = pd.DataFrame()
        df['Angle'] = df_raw[column_map['Angle']]
        df['Zone'] = df_raw[column_map['Zone']].map(zone_map)
        df['ShotType'] = df_raw[column_map['ShotType_Numeric']].map(shottype_map)
        df['Count'] = 1 
        
        initial_count = len(df)
        
        df = df[df['Angle'] != -404.0]
        df.dropna(subset=['Angle', 'Zone', 'ShotType'], inplace=True)
        df = df[(df['Angle'] >= 0) & (df['Angle'] <= 90)]
        
        final_count = len(df)
        
        print(f"Successfully loaded data, initial number of events: {initial_count}")
        print(f"Filtered out Angle=-404.0 and invalid rows, final number of valid shots: {final_count}")
        
        if final_count == 0:
            print("Warning: The number of valid data after processing is zero. Please check the content and column mapping of the DataFrame passed in!")
            return
            
    except KeyError as e:
        print(f"Error: DataFrame does not contain the column {e}. Please check the column_map setting.")
        return

    df_agg = df.groupby(['Angle', 'Zone', 'ShotType']).agg(
        Count=('Count', 'sum')
    ).reset_index()

    df_agg['Zone_Numeric'] = df_agg['Zone'].map(y_mapping)
    max_count = df_agg['Count'].max()

    print("\n--- Data Distribution Summary ---")
    print("Shot Type Ratio (based on actual count sum):\n", df_agg.groupby('ShotType')['Count'].sum().pipe(lambda x: x / x.sum()).round(2))
    print("Zone Distribution of Defensive Shots (based on actual count sum):\n", df_agg[df_agg['ShotType'] == 'Defense Shot'].groupby('Zone')['Count'].sum().pipe(lambda x: x / x.sum()).round(2))
    print("--------------------\n")

    os.makedirs(save_dir_3d, exist_ok=True)
    os.makedirs(save_dir_bar, exist_ok=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for shot_type, group in df_agg.groupby('ShotType'):
        color = colors_list.get(shot_type, 'gray') 
        ax.scatter(
            group['Angle'],
            group['Zone_Numeric'],
            group['Count'],
            c=color,
            marker='o',
            alpha=0.8,
            s=50 + (group['Count'] / max_count) * 200, 
            label=shot_type 
        )

    ax.set_xlabel('Angle', fontsize=12, labelpad=10)
    ax.set_xlim(0, 90)
    ax.set_ylabel('Hitting zone', fontsize=12, labelpad=10)
    ax.set_yticks(list(y_mapping.values()))
    ax.set_yticklabels(Y_CATEGORIES)
    ax.set_zlabel('Count', fontsize=12, labelpad=10)
    
    if z_axis_max_limit is not None:
        ax.set_zlim(0, z_axis_max_limit)
    else:
        ax.set_zlim(0, max_count * 1.1) 

    try:
        plt.tight_layout(pad=2.0)
    except UserWarning:
        pass
    zone_col_name = column_map['Zone']
    save_path_3d = os.path.join(save_dir_3d, f'3d_{zone_col_name}_plot_by_ball_type.png')
    plt.savefig(save_path_3d, bbox_inches='tight')
    plt.show()

    N_BINS = 5
    bins = np.linspace(0, 90, N_BINS + 1)
    bin_labels = [f'{int(bins[i])}-{int(bins[i+1])}°' for i in range(N_BINS)]

    df_agg['Angle_Group'] = pd.cut(df_agg['Angle'], bins=bins, labels=bin_labels, include_lowest=True)
    summary_df = df_agg.groupby(['Angle_Group', 'ShotType'])['Count'].sum().unstack(fill_value=0)

    for cat in COLOR_CATEGORIES:
        if cat not in summary_df.columns:
            summary_df[cat] = 0

    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    width = 0.25 
    x = np.arange(len(bin_labels)) 
    current_x = x - width / 2 

    for shot_type in COLOR_CATEGORIES:
        color = colors_list.get(shot_type, 'gray')
        if shot_type in summary_df.columns:
            ax_bar.bar(current_x, summary_df[shot_type], width, label=shot_type, color=color, alpha=0.8)
        current_x += width 

    ax_bar.set_xlabel('Angle intervals', fontsize=12)
    ax_bar.set_ylabel('Count', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path_bar = os.path.join(save_dir_bar, 'grouped_angle_ball_type_nolable.png')
    plt.savefig(save_path_bar, bbox_inches='tight')
    plt.show()


def plot_badminton_up_down_analysis(
    df_raw, 
    z_axis_max_limit=None, 
    column_map=None,
    zone_map=None,
    shottype_map=None,
    filter_ball_types=None,
    save_dir_3d='./img/3dplot',
    save_dir_bar='./img/angle_grouped'
):
    """
    Analyze badminton data and plot 3D scatter plot and grouped bar chart
    
    Args:
        df_raw (pd.DataFrame): 包含原始數據的 DataFrame
        z_axis_max_limit (int, optional): 3D圖 Z軸的最大值。若未提供，將依數據自動調整。
        column_map (dict, optional): 欄位對應表
        zone_map (dict, optional): 場區對應表
        shottype_map (dict, optional): 球種對應表
        filter_ball_types (list, optional): 要過濾掉的球種清單
        save_dir_3d (str): 3D 圖表存檔目錄
        save_dir_bar (str): 柱狀圖存檔目錄
    """
    
    # --- 0. 預設參數設定 ---
    if column_map is None:
        column_map = {
            'Angle': 'Partner_Angle',
            'Zone': 'Hitting_Zone', 
            'ShotType_Numeric': 'ball_up_down',
            'BallType': 'ball_type' 
        }

    if filter_ball_types is None:
        filter_ball_types = ['發長球', '發短球', '未分類']

    if zone_map is None:
        zone_map = {
            1: 'Forecourt', 2: 'Forecourt', 7: 'Forecourt',
            5: 'Midcourt', 6: 'Midcourt', 8: 'Midcourt',
            3: 'Backcourt', 4: 'Backcourt', 9: 'Backcourt',
            '1': 'Forecourt', '2': 'Forecourt', '7': 'Forecourt',
            '5': 'Midcourt', '6': 'Midcourt', '8': 'Midcourt',
            '3': 'Backcourt', '4': 'Backcourt', '9': 'Backcourt'
        }

    if shottype_map is None:
        shottype_map = {
            1: 'Defense Shot', # 1 防守 (下到上)
            2: 'Attack Shot', # 2 進攻 (上到下)
            0: 'Other Shot'  # 0 其他
            # 1: '防守球', # 1 防守 (下到上)
            # 2: '進攻球', # 2 進攻 (上到下)
            # 0: '其他球'  # 0 其他
        }

    Y_CATEGORIES = ['Forecourt', 'Midcourt', 'Backcourt']
    COLOR_CATEGORIES = ['Attack Shot', 'Defense Shot', 'Other Shot']
    y_mapping = {cat: i for i, cat in enumerate(Y_CATEGORIES)}
    colors_list = {'Attack Shot': 'red', 'Defense Shot': 'blue', 'Other Shot': 'green'} 

    # --- 1. 設定中文字體 ---
    try:
        font_path = fm.findfont(fm.FontProperties(family='Microsoft JhengHei'))
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        print("Unable to find 'Microsoft JhengHei', Chinese characters may not display correctly.")
    plt.rcParams['axes.unicode_minus'] = False 


    # --- 2. 數據清洗與轉換 ---
    try:
        df = pd.DataFrame()
        df['Angle'] = df_raw[column_map['Angle']]
        df['Zone'] = df_raw[column_map['Zone']].map(zone_map)
        df['ShotType'] = df_raw[column_map['ShotType_Numeric']].map(shottype_map)
        df['BallType'] = df_raw[column_map['BallType']]
        df['Count'] = 1 
        
        initial_count = len(df)
        
        # 核心過濾邏輯
        df = df[df['Angle'] != -404.0]
        df = df[~df['BallType'].isin(filter_ball_types)]
        df.dropna(subset=['Angle', 'Zone', 'ShotType', 'BallType'], inplace=True)
        df = df[(df['Angle'] >= 0) & (df['Angle'] <= 90)]
        
        final_count = len(df)

        print(f"Successfully loaded data, original event count: {initial_count}")
        print(f"Filtered Angle=-404.0 and invalid rows, final valid shot events: {final_count}")
        print(f"Filtered BallType: {filter_ball_types}") 
        
        if final_count == 0:
            print("Warning: The number of valid data after processing is zero. Please check the content and column mapping of the passed DataFrame!")
            return
            
    except KeyError as e:
        print(f"Error: DataFrame does not contain column {e}. Please check the column_map setting.")
        return


    # --- 3. 數據處理 (Aggregation) ---
    df_agg = df.groupby(['Angle', 'Zone', 'ShotType']).agg(
        Count=('Count', 'sum')
    ).reset_index()

    df_agg['Zone_Numeric'] = df_agg['Zone'].map(y_mapping)
    max_count = df_agg['Count'].max()

    print("\n--- Data Distribution Summary ---")
    print("Shot Type Ratio (based on actual count sum):\n", df_agg.groupby('ShotType')['Count'].sum().pipe(lambda x: x / x.sum()).round(2))
    print("Zone Distribution of Defensive Shots (based on actual count sum):\n", df_agg[df_agg['ShotType'] == 'Defense Shot'].groupby('Zone')['Count'].sum().pipe(lambda x: x / x.sum()).round(2))
    print("--------------------\n")

    os.makedirs(save_dir_3d, exist_ok=True)
    os.makedirs(save_dir_bar, exist_ok=True)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for shot_type, group in df_agg.groupby('ShotType'):
        color = colors_list.get(shot_type, 'gray')
        ax.scatter(
            group['Angle'],
            group['Zone_Numeric'],
            group['Count'],
            c=color,
            marker='o',
            alpha=0.8,
            s=50 + (group['Count'] / max_count) * 200, 
            label=shot_type 
        )

    ax.set_xlabel('Angle', fontsize=12, labelpad=10)
    ax.set_xlim(0, 90)
    ax.set_ylabel('Hitting zone', fontsize=12, labelpad=10)
    ax.set_yticks(list(y_mapping.values()))
    ax.set_yticklabels(Y_CATEGORIES)
    ax.set_zlabel('Count', fontsize=12, labelpad=10)
    
    if z_axis_max_limit is not None:
        ax.set_zlim(0, z_axis_max_limit)
    else:
        ax.set_zlim(0, max_count * 1.1) 

    try:
        plt.tight_layout(pad=2.0)
    except UserWarning:
        pass
    zone_col_name = column_map['Zone']
    save_path_3d = os.path.join(save_dir_3d, f'3d_{zone_col_name}_plot_by_up_down.png')
    plt.savefig(save_path_3d, bbox_inches='tight')
    plt.show()

    N_BINS = 5
    bins = np.linspace(0, 90, N_BINS + 1)
    bin_labels = [f'{int(bins[i])}-{int(bins[i+1])}°' for i in range(N_BINS)]

    df_agg['Angle_Group'] = pd.cut(df_agg['Angle'], bins=bins, labels=bin_labels, include_lowest=True)
    summary_df = df_agg.groupby(['Angle_Group', 'ShotType'])['Count'].sum().unstack(fill_value=0)

    for cat in COLOR_CATEGORIES:
        if cat not in summary_df.columns:
            summary_df[cat] = 0

    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

    width = 0.25 
    x = np.arange(len(bin_labels)) 
    current_x = x - width 

    for shot_type in COLOR_CATEGORIES:
        color = colors_list.get(shot_type, 'gray')
        if shot_type in summary_df.columns:
            ax_bar.bar(current_x, summary_df[shot_type], width, label=shot_type, color=color, alpha=0.8)
        current_x += width 

    ax_bar.set_xlabel('Angle intervals', fontsize=12)
    ax_bar.set_ylabel('Count', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path_bar = os.path.join(save_dir_bar, 'grouped_angle_up_down_nolable.png')
    plt.savefig(save_path_bar, bbox_inches='tight')
    plt.show()


def analyze_badminton_rally_stats(
    df_raw, 
    event_mapping=None,
    event_col='event',
    rally_col='rally_id',
    shot_col='shot_num'
):
    """
    Analyze badminton rally/shot statistics by event.
    
    Args:
        df_raw (pd.DataFrame): 包含原始數據的 DataFrame
        event_mapping (dict, optional): 賽事項目對應表。預設為 {1: "男雙", 2: "女雙", 3: "混雙"}
        event_col (str): 賽事項目欄位名稱，預設為 'event'
        rally_col (str): 回合 ID 欄位名稱，預設為 'rally_id'
        shot_col (str): 擊球編號/拍數欄位名稱，預設為 'shot_num'
        
    Return:
        tuple: (summary_df, rally_lengths_df)
            - summary_df: 各項目統計摘要
            - rally_lengths_df: 每個回合的拍數明細
    """
    
    # 0. 預設參數設定
    if event_mapping is None:
        # event_mapping = {1: "男雙", 2: "女雙", 3: "混雙"}
        event_mapping = {1: "MD", 2: "WD", 3: "XD"}
        
    # 檢查必備欄位是否存在
    required_cols = [event_col, rally_col, shot_col]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        print(f"Error: Required columns not found in DataFrame: {missing_cols}")
        return None, None

    # 1. 複製資料避免改動原始 df，並定義項目對應
    df = df_raw.copy()
    df['event_name'] = df[event_col].map(event_mapping)
    
    # 處理未對應到的值（可選）
    df['event_name'] = df['event_name'].fillna('未知項目')

    # 2. 核心邏輯：計算「每一個 Rally」分別有多少拍
    rally_lengths = df.groupby(['event_name', rally_col])[shot_col].nunique().reset_index(name='shots_in_this_rally')

    # 3. 針對「回合長度」進行統計
    summary = rally_lengths.groupby('event_name')['shots_in_this_rally'].agg(
        total_rallies='count',
        total_shots='sum',
        average_shots_per_rally='mean',
        max_shots='max',
        min_shots='min' 
    ).reset_index()

    # 4. 輸出結果與全體統計
    print("=== Badminton Rally/Shot Statistics ===")
    print(summary.round(2).to_string(index=False))

    total_rallies = summary['total_rallies'].sum()
    total_shots = summary['total_shots'].sum()
    
    print("-" * 40)
    print(f"Total Rallies: {int(total_rallies)}")
    print(f"Total Shots: {int(total_shots)}")
    if total_rallies > 0:
        print(f"Average Shots per Rally: {total_shots / total_rallies:.2f}")
    else:
        print("Average Shots per Rally: N/A (No valid rallies)")
        
    return summary, rally_lengths

import pandas as pd

def analyze_badminton_shot_distribution(
    df_raw,
    event_mapping=None,
    ball_order=None,
    cols_order=None,
    event_col='event',
    shot_type_col='ball_type'
):
    """
    分析羽球各項目（男雙、女雙、混雙等）的球種使用數量與百分比分佈
    
    參數:
        df_raw (pd.DataFrame): 包含原始數據的 DataFrame
        event_mapping (dict, optional): 賽事項目對應表。預設為 {1: '男生', 2: '女生', 3: '混雙'}
        ball_order (list, optional): 報表 Y 軸的球種顯示順序
        cols_order (list, optional): 報表 X 軸的項目顯示順序
        event_col (str): 賽事項目欄位名稱，預設為 'event'
        shot_type_col (str): 球種欄位名稱，預設為 'ball_type'
        
    回傳:
        pd.DataFrame: 格式化後的最終報表 [數量 (百分比%)]
    """
    
    # 0. 預設參數設定
    if event_mapping is None:
        event_mapping = {1: 'MD', 2: 'WD', 3: 'XD'}
        
    if ball_order is None:
        ball_order = [
            '切球', '發長球', '發短球', '平球', '挑球', 
            '推撲球', '殺球', '網前小球', '長球'
        ]
        
    if cols_order is None:
        cols_order = ['MD', 'WD', 'XD', 'Total']

    # 檢查必備欄位是否存在
    required_cols = [event_col, shot_type_col]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        print(f"Error: Required columns not found in DataFrame: {missing_cols}")
        return None

    # 1. 複製資料並定義名稱
    df = df_raw.copy()
    df['event_name'] = df[event_col].map(event_mapping)

    # 2. 建立基礎數據表 (Count)
    df_count = pd.crosstab(
        df[shot_type_col], 
        df['event_name'], 
        margins=True, 
        margins_name='Total' 
    )

    # 3. 確保欄位順序 (動態過濾掉資料中沒有的項目)
    available_cols = [c for c in cols_order if c in df_count.columns]
    df_count = df_count[available_cols]

    # 4. 計算百分比表 (Percentage)
    df_pct = df_count.div(df_count.loc['Total'], axis=1) * 100

    # 5. 合併「數量」與「百分比」格式
    df_formatted = pd.DataFrame()
    for col in df_count.columns:
        df_formatted[col] = [f"{int(c)} ({p:.2f}%)" for c, p in zip(df_count[col], df_pct[col])]

    # 補回索引 (球種名稱)
    df_formatted.index = df_count.index

    # 6. 依照指定球種順序排列 (Rows)
    df_final = df_formatted.reindex(ball_order).fillna("0 (0.00%)")

    # 7. 處理最下方的「總共球數」列
    if 'Total' in df_count.index:
        total_row_counts = df_count.loc['Total']
        total_row_display = [f"{int(val)}" for val in total_row_counts] 
        df_total_row = pd.DataFrame([total_row_display], columns=df_count.columns, index=['Total'])
        df_final = pd.concat([df_final, df_total_row])

    # Display translation to English
    translation_map = {
        '切球': 'Drop',
        '發長球': 'Long Serve',
        '發短球': 'Short Serve',
        '平球': 'Drive',
        '挑球': 'Lob',
        '推撲球': 'Push/Rush',
        '殺球': 'Smash',
        '網前小球': 'Net Shot',
        '長球': 'Clear',
        '未分類': 'Unclassified',
        'Total': 'Total'
    }
    df_final.index = df_final.index.map(lambda x: translation_map.get(x, x))

    print("\n--- Final Analysis Report [Count (Percentage)] ---")
    print(df_final.to_string())
    print("-" * 40)
    
    return df_final