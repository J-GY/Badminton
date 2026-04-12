import pandas as pd
import numpy as np

X_BINS = [0, 20, 41, 61]
Y_BINS_BOTTOM = [0, 22, 45, 67] # 下半場 (近)
Y_BINS_TOP = [67, 90, 113, 134] # 上半場 (遠)

"""
           13 
        --------
     14| x  x  x|12
     15| x  x  x|11
     16| x  x  x|10
    - 下半場 (1-9): 
       | 2  7  1|
       | 6  8  5|
       | 4  9  3|
"""
def get_zone_id(x, y):
    """將 (x, y) 座標映射到 1-18 的戰術區域 ID"""
    if pd.isna(x) or pd.isna(y):
        return -1
    # 出界區域
    if y > Y_BINS_TOP[3]:
        return 13 
    elif y > Y_BINS_TOP[2]:
        if x> X_BINS[3]:
            return 12 
        elif x < X_BINS[0]:
            return 14
    elif y > Y_BINS_TOP[1]:
        if x> X_BINS[3]:
            return 11 
        elif x < X_BINS[0]:
            return 15
    elif y > Y_BINS_TOP[0]:
        if x> X_BINS[3]:
            return 10
        elif x < X_BINS[0]:
            return 16
    
    if y > Y_BINS_BOTTOM[2] and y <= Y_BINS_BOTTOM[3]:
        if x > X_BINS[2] and x <= X_BINS[3]:
            return 1
        elif x > X_BINS[1] and x <= X_BINS[2]:
            return 7
        elif x >= X_BINS[0] and x <= X_BINS[1]:
            return 2
    elif y > Y_BINS_BOTTOM[1] and y <= Y_BINS_BOTTOM[2]:
        if x > X_BINS[2] and x <= X_BINS[3]:
            return 5
        elif x > X_BINS[1] and x <= X_BINS[2]:
            return 8
        elif x >= X_BINS[0] and x <= X_BINS[1]:
            return 6
    elif y >= Y_BINS_BOTTOM[0] and y <= Y_BINS_BOTTOM[1]:
        if x > X_BINS[2] and x <= X_BINS[3]:
            return 3
        elif x > X_BINS[1] and x <= X_BINS[2]:
            return 9
        elif x >= X_BINS[0] and x <= X_BINS[1]:
            return 4
    return -1  # 無效區域

def normalize_coordinates(df_in, 
                          net_y=67, 
                          court_width=61, 
                          court_length=134,
                          team=None):
    """
    座標歸一 (Normalization) 函數。
    將所有擊球翻轉成從「下半場 (y < 67)」打到「上半場 (y > 67)」。
    
    輸入參數維持您的定義:
    team: 'AB' 表示要檢查 A/B 隊是否在上半場; 'CD' 表示檢查 C/D 隊。
    """
    df = df_in.copy()
    
    # 預設不翻轉 (False)
    is_top_side = pd.Series(False, index=df.index)
    
    # 1. 依據傳入的 team 參數，判斷該隊伍是否在「上半場」
    # 您的邏輯: 只要該隊伍在 y > 67，就代表需要翻轉 (因為我們要以該隊為視角，將其轉到下半場)
    if team == 'AB':
        if 'player_A_y' in df.columns:
            is_top_side = df['player_A_y'] > net_y
    elif team == 'CD':
        if 'player_C_y' in df.columns:
            is_top_side = df['player_C_y'] > net_y

    # 3. 對位於「上半場」的球 (is_top_side) 進行 180 度翻轉
    
    # 翻轉擊球點
    if 'hit_x' in df.columns:
        df.loc[is_top_side, 'hit_x'] = court_width - df.loc[is_top_side, 'hit_x']
    if 'hit_y' in df.columns:
        df.loc[is_top_side, 'hit_y'] = court_length - df.loc[is_top_side, 'hit_y']
        
    # 翻轉落球點
    if 'return_x' in df.columns:
        df.loc[is_top_side, 'return_x'] = court_width - df.loc[is_top_side, 'return_x']
    if 'return_y' in df.columns:
        df.loc[is_top_side, 'return_y'] = court_length - df.loc[is_top_side, 'return_y']
        
    # 翻轉球員位置 (如果有的話)
    cols_to_flip_x = ['player_A_x', 'player_B_x', 'player_C_x', 'player_D_x']
    cols_to_flip_y = ['player_A_y', 'player_B_y', 'player_C_y', 'player_D_y']
    
    for col in cols_to_flip_x:
        if col in df.columns:
            df.loc[is_top_side, col] = court_width - df.loc[is_top_side, col]
    for col in cols_to_flip_y:
        if col in df.columns:
            df.loc[is_top_side, col] = court_length - df.loc[is_top_side, col]
            
    return df

def df_filter_by_conditions(df, 
                            match_id=None, 
                            set_num=None, 
                            start_rally_id=None, 
                            end_rally_id=None,
                            player_id=None,
                            team=None,
                            shot_num=None,
                            exclude_flaw=True,
                            event_type=None):
    df_filtered = df.copy()
    filter_title_parts = []

    if match_id is not None:
        if 'match_id' not in df_filtered.columns:
            print("警告: 找不到 'match_id' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['match_id'] == match_id]
            filter_title_parts.append(f"Match_{match_id}")

        # 依 Set Num 篩選
    if set_num is not None:
        if 'set_num_y' not in df_filtered.columns:
            print("警告: 找不到 'set_num' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['set_num_y'] == set_num]
            filter_title_parts.append(f"Set_{set_num}")

        # 依 Rally ID 範圍篩選
    if start_rally_id is not None:
        if 'rally_id' not in df_filtered.columns:
            print("警告: 找不到 'rally_id' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['rally_id'] >= start_rally_id]
            filter_title_parts.append(f"Rally_from_{start_rally_id}")

    if end_rally_id is not None:
        if 'rally_id' not in df_filtered.columns:
            print("警告: 找不到 'rally_id' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['rally_id'] <= end_rally_id]
            filter_title_parts.append(f"Rally_to_{end_rally_id}")

    if team is not None:
        filter_title_parts.append(f"Team_{team}")

    if player_id is not None:
        if 'player' not in df_filtered.columns:
            print(f"警告: 找不到與球員 ID '{player_id}' 對應的欄位。")
        else:
            df_filtered = df_filtered[df_filtered['player'] == player_id]
            filter_title_parts.append(f"Player_{player_id}")
            
    if shot_num is not None:
        if 'shot_num' not in df_filtered.columns:
            print("警告: 找不到 'shot_num' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['shot_num'] == shot_num]
            
    if exclude_flaw:
        if 'flaw' not in df_filtered.columns:
            print("警告: 找不到 'flaw' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['flaw'] != 1]
            
    if event_type is not None:
        if 'event' not in df_filtered.columns:
            print("警告: 找不到 'event' 欄位。")
        else:
            df_filtered = df_filtered[df_filtered['event'] == event_type]
            filter_title_parts.append(f"Event_{event_type}")
            
    if not filter_title_parts:
        filter_title_suffix = "Full Dataset"
    else:
        filter_title_suffix = "_".join(filter_title_parts)
    return df_filtered, filter_title_suffix
