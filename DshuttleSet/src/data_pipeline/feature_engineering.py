import pandas as pd
import numpy as np
from typing import Dict, List
import os
# --- 1. global constants ---

# Y axis
AVG_BACKCOURT_Y_MIN = 0
AVG_MIDCOURT_Y_START_HIGH = 22
AVG_FORECOURT_Y_START = 45
AVG_NET_Y_CENTER = 67
AVG_FORECOURT_Y_END = 89
AVG_MIDCOURT_Y_END_LOW = 112
AVG_BACKCOURT_Y_MAX = 134

# X axis
X_LEFT_BOUNDARY = 20
X_CENTER_BOUNDARY = 41
X_MAX_COURT = 61
X_MIN_COURT = 0

# angle bin
ANGLE_BINS = [0, 18, 36, 54, 72, 90]
ANGLE_BIN_LABELS = [0, 1, 2, 3, 4]
ANGLE_ERROR_BIN = 5
ANGLE_ERROR_CODE = -404

# ball type classification map
CLASSIFICATION_MAP = {
    '擋小球': '網前小球', '勾球': '網前小球', '放小球': '網前小球', '小平球': '網前小球', '網前小球': '網前小球',
    '推球': '推撲球', '撲球': '推撲球', '推撲球': '推撲球',
    '挑球': '挑球', '防守回挑': '挑球',
    '防守回抽': '平球', '平球': '平球', '後場抽平球': '平球',
    '切球': '切球', '過度切球': '切球',
    '殺球': '殺球',
    '長球': '長球',
    '發長球': '發長球', '發短球': '發短球'
}

# --- 3. ball type classification ---

def map_ball_type_to_group(df: pd.DataFrame, classification_map: Dict[str, str]) -> pd.DataFrame:
    """
    創建新的 'new_ball_type' 欄位，將原始 ball_type 映射到分類群組。
    未分類的球種將被標記為 '未分類'。
    """
    df['ball_type'] = df['ball_type'].map(classification_map)
    
    # 處理未映射的球種 (即 NaN)
    df['ball_type'] = df['ball_type'].fillna('未分類')

    print("\n--- 球種映射完成 ---")
    print(f"最終分類結果:\n{df['ball_type'].value_counts()}")
    return df

# --- 4. Hitting Zone ---

import pandas as pd
from typing import Dict


def get_court_zone(row: pd.Series) -> str:
    """
    According to the hitting player's coordinates (X, Y), divide the 1-9 regions, 
    and assign specific abnormal region codes (10-16) to coordinates outside the boundary.
    """
    hitting_player_id = row['player']
    player_cols = {'A': ['player_A_x', 'player_A_y'], 'B': ['player_B_x', 'player_B_y'],
                   'C': ['player_C_x', 'player_C_y'], 'D': ['player_D_x', 'player_D_y']}
    
    player_id_map = {row.get(key): key for key in player_cols.keys() if row.get(key) is not None}
    
    if hitting_player_id in player_id_map:
        player_key = player_id_map[hitting_player_id]
        try:
            x_coord = row[player_cols[player_key][0]]
            y_coord = row[player_cols[player_key][1]]
            if pd.isna(x_coord) or pd.isna(y_coord):
                return '-1'
        except KeyError:
             return '-1'
    else:
        return '-1'
    
    y_zone_type = '' 
    
    if y_coord < AVG_BACKCOURT_Y_MIN or y_coord>AVG_BACKCOURT_Y_MAX: # AVG_BACKCOURT_Y_MIN = 0
        return '13' 

    if (y_coord >= AVG_BACKCOURT_Y_MIN and y_coord <= AVG_MIDCOURT_Y_START_HIGH) or \
       (y_coord >= AVG_MIDCOURT_Y_END_LOW and y_coord <= AVG_BACKCOURT_Y_MAX):
        y_zone_type = 'Backcourt'
    elif (y_coord > AVG_MIDCOURT_Y_START_HIGH and y_coord < AVG_FORECOURT_Y_START) or \
         (y_coord > AVG_FORECOURT_Y_END and y_coord < AVG_MIDCOURT_Y_END_LOW):
        y_zone_type = 'Midcourt'
    elif y_coord >= AVG_FORECOURT_Y_START and y_coord <= AVG_FORECOURT_Y_END:
        y_zone_type = 'Forecourt'
    else:
        return '-1' 
    
    if x_coord < X_MIN_COURT: # X_MIN_COURT = 0
        if y_zone_type == 'Forecourt':
            return '10' # Forecourt x<0
        elif y_zone_type == 'Midcourt':
            return '11' # Midcourt x<0
        elif y_zone_type == 'Backcourt':
            return '12' # Backcourt x<0
    
    elif x_coord > X_MAX_COURT: # X_MAX_COURT = 61
        if y_zone_type == 'Forecourt':
            return '16' # Forecourt x>61
        elif y_zone_type == 'Midcourt':
            return '15' # Midcourt x>61
        elif y_zone_type == 'Backcourt':
            return '14' # Backcourt x>61

    x_pos = 0 
    if x_coord <= X_LEFT_BOUNDARY:
        x_pos = 1
    elif x_coord < X_CENTER_BOUNDARY:
        x_pos = 2
    elif x_coord <= X_MAX_COURT:
        x_pos = 3
    else:
        return '-1' 

    zone_number = 0
    
    if y_coord < AVG_NET_Y_CENTER: 
        if y_zone_type == 'Forecourt':
            if x_pos == 1: zone_number = 2
            elif x_pos == 2: zone_number = 7
            elif x_pos == 3: zone_number = 1
        elif y_zone_type == 'Midcourt':
            if x_pos == 1: zone_number = 6
            elif x_pos == 2: zone_number = 8
            elif x_pos == 3: zone_number = 5
        elif y_zone_type == 'Backcourt':
            if x_pos == 1: zone_number = 4
            elif x_pos == 2: zone_number = 9
            elif x_pos == 3: zone_number = 3
            
    else: # 上半場 (Y >= 67)
        if y_zone_type == 'Forecourt':
            if x_pos == 1: zone_number = 1
            elif x_pos == 2: zone_number = 7
            elif x_pos == 3: zone_number = 2
        elif y_zone_type == 'Midcourt':
            if x_pos == 1: zone_number = 5
            elif x_pos == 2: zone_number = 8
            elif x_pos == 3: zone_number = 6
        elif y_zone_type == 'Backcourt':
            if x_pos == 1: zone_number = 3
            elif x_pos == 2: zone_number = 9
            elif x_pos == 3: zone_number = 4
            
    return f'{zone_number}' if zone_number != 0 else '-1'

def process_court_zones(df: pd.DataFrame) -> pd.DataFrame:
    """apply the function of court zone and output the statistics result."""
    df['Hitting_Zone'] = df.apply(get_court_zone, axis=1)
    
    error_count = (df['Hitting_Zone'] == '-1').sum()
    if error_count > 0:
        print(f"\n  {error_count} records of null or unclassifiable errors (all marked as -1)!")

    print("\n--- 1-16 blocks result---")
    print(df['Hitting_Zone'].value_counts())
    
    return df

# --- 5. Partner Angle ---

def cal_angle(row: pd.Series) -> float:
    """calculate the angle between the hitting player and his partner (0°~90°)."""
    hitting_player_id = row['player']

    if hitting_player_id == row['A']:
        hitter_suffix = 'A'
        partner_suffix = 'B'
    elif hitting_player_id == row['B']:
        hitter_suffix = 'B'
        partner_suffix = 'A'
    elif hitting_player_id == row['C']:
        hitter_suffix = 'C'
        partner_suffix = 'D'
    elif hitting_player_id == row['D']:
        hitter_suffix = 'D'
        partner_suffix = 'C'
    else:
        return np.nan 

    try:
        x_H = row[f'player_{hitter_suffix}_x']
        y_H = row[f'player_{hitter_suffix}_y']
        x_P = row[f'player_{partner_suffix}_x']
        y_P = row[f'player_{partner_suffix}_y']
    except KeyError:
        return ANGLE_ERROR_CODE 

    delta_x = x_H - x_P
    delta_y = y_H - y_P

    magnitude = np.sqrt(delta_x**2 + delta_y**2)

    if magnitude == 0:
        return ANGLE_ERROR_CODE
        
    cos_phi = delta_x / magnitude
    phi_rad = np.arccos(np.clip(cos_phi, -1.0, 1.0))
    phi_deg = np.degrees(phi_rad)
    
    if phi_deg > 90:
        final_angle = 180 - phi_deg
    else:
        final_angle = phi_deg

    return round(final_angle, 0)

def process_partner_angle(df: pd.DataFrame) -> pd.DataFrame:
    """calculate the partner angle and group it according to the global constants."""
    df['Partner_Angle'] = df.apply(cal_angle, axis=1)
    
    df['Angle_Bin'] = ANGLE_ERROR_BIN 

    valid_indices = df['Partner_Angle'] != ANGLE_ERROR_CODE
    valid_angles = df.loc[valid_indices, 'Partner_Angle']

    df.loc[valid_indices, 'Angle_Bin'] = pd.cut(
        valid_angles,
        bins=ANGLE_BINS,
        labels=ANGLE_BIN_LABELS,
        right=True,
        include_lowest=True,
        ordered=False
    ).astype(int) 

    print("\n--- Partner Angle Bin result ---")
    print(df['Angle_Bin'].value_counts().sort_index())
    
    return df

# --- 6. Ball Height Direction ---

def up_down(row: pd.Series) -> int:
    """judge the ball is going up (2), down (1) or flat (0), -1 means the height data is invalid."""
    hit_h = row['hit_height']
    return_h = row['return_height']
    
    if (hit_h == 0 or return_h == 0):
        return -1
        
    if (hit_h < return_h):
        return 2  # up
    elif (hit_h > return_h):
        return 1  # down
    else:
        return 0  # flat

def process_ball_height_dir(df: pd.DataFrame) -> pd.DataFrame:
    """apply the function of ball height direction and output the statistics result."""
    df['ball_up_down'] = df.apply(up_down, axis=1)
    
    print("\n--- Ball Height Direction result ---")
    print(df['ball_up_down'].value_counts())
    
    return df

# --- 7. (Main) ---

def clean_return_height(df: pd.DataFrame) -> pd.DataFrame:
    """According to the logic, clean the 'return_height' column in the DataFrame."""
    df_clean = df.copy()

    # --- Stage 1: Mark the last shot of each rally (is_final_shot) ---
    if 'rally_id' in df_clean.columns and 'shot_num' in df_clean.columns:
        max_shot_num_per_rally = df_clean.groupby('rally_id')['shot_num'].max().reset_index()
        max_shot_num_per_rally.columns = ['rally_id', 'max_shot_num']
        df_clean = pd.merge(df_clean, max_shot_num_per_rally, on='rally_id', how='left')
        df_clean['is_final_shot'] = (df_clean['shot_num'] == df_clean['max_shot_num'])
        df_clean = df_clean.drop(columns=['max_shot_num'])
        
        # --- Stage 2: Handle the last shot's return_height = 0 (change to 2) ---
        condition_final_shot_zero = (df_clean['is_final_shot'] == True) & (df_clean['return_height'] == 0)
        df_clean.loc[condition_final_shot_zero, 'return_height'] = 2
        
        # --- Stage 3: Handle the non-last shot's return_height = 0 (use the next shot's hit_height) ---
        df_clean = df_clean.sort_values(by=['rally_id', 'shot_num']).reset_index(drop=True)
        df_clean['next_rally_id'] = df_clean['rally_id'].shift(-1)
        df_clean['next_hit_height'] = df_clean['hit_height'].shift(-1)

        condition_update_next = (df_clean['return_height'] == 0) & \
                                (df_clean['rally_id'] == df_clean['next_rally_id'])
        df_clean.loc[condition_update_next, 'return_height'] = df_clean.loc[condition_update_next, 'next_hit_height']
        df_clean = df_clean.drop(columns=['next_rally_id', 'next_hit_height'])
    
    return df_clean

def caculate_analysis_col(df_analysis,selected_type, shot_to_end):
    cols_to_numeric = ['player', 'score_team', 'shot_num', 'shot_count', 'match_id', 'A', 'B', 'C', 'D', 'set_win']
    for col in cols_to_numeric:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')

    df_analysis = df_analysis.sort_values(by=['match_id', 'set_id', 'rally_id', 'shot_num'])

    df_analysis['match_Winner_mid_cover'] = df_analysis['match_Winner_mid_cover'].astype(str).str.lower().isin(['true', '1'])
    df_analysis['match_Loser_mid_cover'] = df_analysis['match_Loser_mid_cover'].astype(str).str.lower().isin(['true', '1'])

    # 0.0 = team_Winner
    # 1.0 = team_Loser
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
            if pid == row['A'] or pid == row['B']:
                return 1.0 
            elif pid == row['C'] or pid == row['D']:
                return 0.0 
        return np.nan

    df_analysis['player_team'] = df_analysis.apply(get_player_team_logic, axis=1)

    df_analysis['did_opponent_team_win'] = (df_analysis['player_team'] != df_analysis['score_team'])
    df_analysis['shots_to_end'] = df_analysis['shot_count'] - df_analysis['shot_num']
    
    is_lose_type = False
    df_analysis['lose_reason'] = df_analysis['lose_reason'].astype(str).str.strip()
    is_lose_type = df_analysis['lose_reason'].isin(selected_type)

    # Definiton of Defense failure：
    df_analysis['is_defense_failure'] = ( 
        (df_analysis['did_opponent_team_win']) &
        (
            (df_analysis['shots_to_end'].isin(shot_to_end)) &
            (is_lose_type)
        )
    )
    
    df_analysis['player_formation'] = np.where(
        df_analysis['player_team'] == 0.0, 
        df_analysis['match_Winner_formation'], 
        df_analysis['match_Loser_formation']
    )
    
    df_analysis['next_rally_id'] = df_analysis['rally_id'].shift(-1)
    df_analysis['next_winner_mid_cover'] = df_analysis['match_Winner_mid_cover'].shift(-1)
    df_analysis['next_loser_mid_cover'] = df_analysis['match_Loser_mid_cover'].shift(-1)
    
    conditions = [
        (df_analysis['rally_id'] != df_analysis['next_rally_id']), 
        (df_analysis['player_team'] == 0.0), 
        (df_analysis['player_team'] == 1.0)
    ]
    
    choices = [
        np.nan, 
        df_analysis['next_winner_mid_cover'],
        df_analysis['next_loser_mid_cover']
    ]
    
    df_analysis['is_reaction_cover'] = np.select(conditions, choices, default=np.nan)

    return df_analysis

def calculate_voronoi_vectorized(df, weight=True, decay_rate=0.021):
    
    xx_up, yy_up = np.meshgrid(np.arange(0, 61, 1), np.arange(67, 134, 1))
    up_grid_x = xx_up.ravel()
    up_grid_y = yy_up.ravel()
    
    xx_down, yy_down = np.meshgrid(np.arange(0, 61, 1), np.arange(0, 67, 1))
    down_grid_x = xx_down.ravel()
    down_grid_y = yy_down.ravel()
    
    df['match_Winner_AAI'] = np.nan
    df['match_Loser_AAI'] = np.nan

    def calc_aai_batch(x1, y1, x2, y2, hx, hy, grid_x, grid_y):
        dist1 = (x1[:, None] - grid_x)**2 + (y1[:, None] - grid_y)**2
        dist2 = (x2[:, None] - grid_x)**2 + (y2[:, None] - grid_y)**2
        
        mask_1 = dist1 < dist2
        
        if weight:
            dist_to_hitter = np.sqrt((hx[:, None] - grid_x)**2 + (hy[:, None] - grid_y)**2)
            weight_grid = np.exp(-decay_rate * dist_to_hitter)
            
            load_1 = (mask_1 * weight_grid).sum(axis=1)
            load_2 = ((~mask_1) * weight_grid).sum(axis=1)
        else:
            load_1 = mask_1.sum(axis=1)
            load_2 = (~mask_1).sum(axis=1)
            
        total_load = load_1 + load_2
        total_load_safe = np.where(total_load == 0, 1, total_load)
        
        aai = np.abs(load_1 - load_2) / total_load_safe
        
        return np.where(total_load == 0, np.nan, aai)

    ab_is_up = (df['player_A_y'] > 67).fillna(False)
    ab_is_down = (~ab_is_up) & df['player_A_y'].notna()

    ab_hits = (df['player_team'] == 0.0) 
    cd_hits = (df['player_team'] == 1.0) 

    mask_1 = ab_hits & ab_is_up
    if mask_1.any():
        df_sub = df[mask_1]
        df.loc[mask_1, 'match_Loser_AAI'] = calc_aai_batch(
            df_sub['player_C_x'].values, df_sub['player_C_y'].values,
            df_sub['player_D_x'].values, df_sub['player_D_y'].values,
            df_sub['hit_x'].values, df_sub['hit_y'].values,  
            down_grid_x, down_grid_y                    
        )

    mask_2 = ab_hits & ab_is_down
    if mask_2.any():
        df_sub = df[mask_2]
        df.loc[mask_2, 'match_Loser_AAI'] = calc_aai_batch(
            df_sub['player_C_x'].values, df_sub['player_C_y'].values,
            df_sub['player_D_x'].values, df_sub['player_D_y'].values,
            df_sub['hit_x'].values, df_sub['hit_y'].values,
            up_grid_x, up_grid_y
        )

    mask_3 = cd_hits & ab_is_up
    if mask_3.any():
        df_sub = df[mask_3]
        df.loc[mask_3, 'match_Winner_AAI'] = calc_aai_batch(
            df_sub['player_A_x'].values, df_sub['player_A_y'].values,
            df_sub['player_B_x'].values, df_sub['player_B_y'].values,
            df_sub['hit_x'].values, df_sub['hit_y'].values, 
            up_grid_x, up_grid_y                        
        )

    mask_4 = cd_hits & ab_is_down
    if mask_4.any():
        df_sub = df[mask_4]
        df.loc[mask_4, 'match_Winner_AAI'] = calc_aai_batch(
            df_sub['player_A_x'].values, df_sub['player_A_y'].values,
            df_sub['player_B_x'].values, df_sub['player_B_y'].values,
            df_sub['hit_x'].values, df_sub['hit_y'].values,
            down_grid_x, down_grid_y
        )

    return df

def main_data_pipeline(df):
    """
    ADD feature and do some change for ball type
    """
        
    # 1.5. Clean return height
    print("\n--- Clean return height ---")
    df = clean_return_height(df)

    # 2. balltype classification
    df = map_ball_type_to_group(df, CLASSIFICATION_MAP)
    
    # 3. grid the hitting zone
    df = process_court_zones(df)
    
    # 4. calculate partner angle
    df = process_partner_angle(df)

    # 5. add center coverage and formation
    print("\n--- add center coverage and formation ---")
    df = add_tactical_columns(df)
    df = caculate_analysis_col(df, selected_type=['對手落地致勝', '未過網', '掛網', '出界'], shot_to_end=[1,2,3])
    df = calculate_voronoi_vectorized(df, weight=True, decay_rate=0.021)

    # 6. determine the direction of ball height
    df = process_ball_height_dir(df)
    
    print("\n--- Final data validation (first few rows) ---")
    print(df[['shot_id', 'player', 'ball_type', 'Hitting_Zone', 'Partner_Angle', 'Angle_Bin', 'ball_up_down']].head())
    
    return df



def add_tactical_columns(df_in, 
                         y_diff_threshold=10, 
                         net_y=67,
                         cover_type='circle', 
                         formation_method='angle'): 
    SQUARE_SIZE = 20
    RADIUS = 15  
    MID_X = 30.5
    DISTANCE_TO_CENTER = 12.5
    MID_Y_BOTTOM = 47 - DISTANCE_TO_CENTER 
    MID_Y_TOP = 87 + DISTANCE_TO_CENTER    
    MID_X_RANGE = (MID_X - SQUARE_SIZE/2, MID_X + SQUARE_SIZE/2)
    MID_Y_BOTTOM_RANGE = (MID_Y_BOTTOM - SQUARE_SIZE/2, MID_Y_BOTTOM + SQUARE_SIZE/2)
    MID_Y_TOP_RANGE = (MID_Y_TOP - SQUARE_SIZE/2, MID_Y_TOP + SQUARE_SIZE/2)
    
    df = df_in.copy()
    required_cols = [
        'player_A_y', 'player_B_y', 'player_C_y', 'player_D_y',
        'player_A_x', 'player_B_x', 'player_C_x', 'player_D_x'
    ]
    if not all(col in df.columns for col in required_cols):
        print("Miss player location.")
        return None

    if formation_method == 'y_diff':
        y_diff_AB = abs(df['player_A_y'] - df['player_B_y'])
        df['match_Winner_formation'] = np.where(y_diff_AB > y_diff_threshold, 'Attack', 'Defense')
        y_diff_CD = abs(df['player_C_y'] - df['player_D_y'])
        df['match_Loser_formation'] = np.where(y_diff_CD > y_diff_threshold, 'Attack', 'Defense')
    elif formation_method == 'angle':
        dy_AB = abs(df['player_A_y'] - df['player_B_y'])
        dx_AB = abs(df['player_A_x'] - df['player_B_x'])
        angle_AB = np.degrees(np.arctan2(dy_AB, dx_AB))
        conditions_AB = [(angle_AB < 54), (angle_AB >= 54)]
        choices = ['Defense', 'Attack']
        df['match_Winner_formation'] = np.select(conditions_AB, choices, default='Other')
        dy_CD = abs(df['player_C_y'] - df['player_D_y'])
        dx_CD = abs(df['player_C_x'] - df['player_D_x'])
        angle_CD = np.degrees(np.arctan2(dy_CD, dx_CD))
        conditions_CD = [(angle_CD < 54), (angle_CD >= 54)]
        df['match_Loser_formation'] = np.select(conditions_CD, choices, default='Other')
    else:
        raise ValueError("formation_method must be 'y_diff' or 'angle'")

    def check_in_zone(x_col, y_col, center_y, y_range):
        if cover_type == 'square':
            in_x = (df[x_col] >= MID_X_RANGE[0]) & (df[x_col] <= MID_X_RANGE[1])
            in_y = (df[y_col] >= y_range[0]) & (df[y_col] <= y_range[1])
            return in_x & in_y
        elif cover_type == 'circle':
            dist_sq = (df[x_col] - MID_X)**2 + (df[y_col] - center_y)**2
            return dist_sq <= (RADIUS**2)
        else:
            raise ValueError("cover_type must be 'square' or 'circle'")

    team_AB_is_bottom = (df['player_A_y'] < net_y)
    A_in_bottom = check_in_zone('player_A_x', 'player_A_y', MID_Y_BOTTOM, MID_Y_BOTTOM_RANGE)
    B_in_bottom = check_in_zone('player_B_x', 'player_B_y', MID_Y_BOTTOM, MID_Y_BOTTOM_RANGE)
    A_in_top = check_in_zone('player_A_x', 'player_A_y', MID_Y_TOP, MID_Y_TOP_RANGE)
    B_in_top = check_in_zone('player_B_x', 'player_B_y', MID_Y_TOP, MID_Y_TOP_RANGE)
    df['match_Winner_mid_cover'] = np.where(team_AB_is_bottom, (A_in_bottom | B_in_bottom), (A_in_top | B_in_top))
    
    team_CD_is_bottom = (df['player_C_y'] < net_y)
    C_in_bottom = check_in_zone('player_C_x', 'player_C_y', MID_Y_BOTTOM, MID_Y_BOTTOM_RANGE)
    D_in_bottom = check_in_zone('player_D_x', 'player_D_y', MID_Y_BOTTOM, MID_Y_BOTTOM_RANGE)
    C_in_top = check_in_zone('player_C_x', 'player_C_y', MID_Y_TOP, MID_Y_TOP_RANGE)
    D_in_top = check_in_zone('player_D_x', 'player_D_y', MID_Y_TOP, MID_Y_TOP_RANGE)
    df['match_Loser_mid_cover'] = np.where(team_CD_is_bottom, (C_in_bottom | D_in_bottom), (C_in_top | D_in_top))
    
    if 'rally_id' in df.columns and 'shot_id' in df.columns:
        df['shot_count'] = df.groupby('rally_id')['shot_id'].transform('count')
    
    return df