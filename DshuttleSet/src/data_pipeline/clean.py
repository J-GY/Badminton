import pandas as pd
import numpy as np

def remove_outlier_rallies(df_main: pd.DataFrame, df_outliers: pd.DataFrame) -> pd.DataFrame:
    """
    According to the rally_id in the exception dataset, remove all rows containing these rally_id from the main dataset.

    Args:
        df_main: The main DataFrame to be cleaned (usually the data that has undergone preliminary cleaning).
        df_outliers: The DataFrame containing exception data (used to extract the list of rally_ids to be removed).

    Returns:
        pd.DataFrame: The final clean DataFrame after removing the entire rally of exception data.
    """
    
    outlier_rally_ids = df_outliers['rally_id'].unique()
    
    condition_to_keep = ~df_main['rally_id'].isin(outlier_rally_ids)
    
    df_final_clean = df_main[condition_to_keep].copy()
    
    rows_removed_by_rally = len(df_main) - len(df_final_clean)
    total_rally_removed = len(outlier_rally_ids)
    
    print("--- remove result ---")
    print(f"Origin {len(df_main)} rows。")
    print(f"{total_rally_removed} rally of Exception")
    print(f"Remove {rows_removed_by_rally} rows。")
    print(f"Final {len(df_final_clean)} rows。")
    
    return df_final_clean

from typing import Dict, List

def clean_coordinate_outliers(df: pd.DataFrame, 
                              classification_map: Dict[str, str] = None, 
                              x_min: int = -10, x_max: int = 71, 
                              y_min: int = -10, y_max: int = 144,
                              serve_types: List[str] = ['發長球', '發短球']) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Data clean:
    ball_type '未分類'，
    hit_height == 0 。
    error case (Hitting_Zone == '-1', Angle_Bin == 5)。
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    condition_ball_type = (df_clean['ball_type'] == '未分類')

    def is_outlier_x(x): return (x < x_min) | (x > x_max)
    def is_outlier_y(y): return (y < y_min) | (y > y_max)
    
    condition_coord = pd.Series(False, index=df_clean.index)
    player_x_cols = ['player_A_x', 'player_B_x', 'player_C_x', 'player_D_x']
    player_y_cols = ['player_A_y', 'player_B_y', 'player_C_y', 'player_D_y']
    
    for col in player_x_cols:
        if col in df_clean.columns: condition_coord |= df_clean[col].apply(is_outlier_x)
    for col in player_y_cols:
        if col in df_clean.columns: condition_coord |= df_clean[col].apply(is_outlier_y)

    condition_height = (df_clean['hit_height'] == 0)

    condition_error_cases = pd.Series(False, index=df_clean.index)
    if 'Hitting_Zone' in df_clean.columns:
        condition_error_cases |= (df_clean['Hitting_Zone'] == '-1')
    if 'Angle_Bin' in df_clean.columns:
        condition_error_cases |= (df_clean['Angle_Bin'] == 5)

    is_outlier = condition_ball_type | condition_coord | condition_height | condition_error_cases
    
    df_retained = df_clean[~is_outlier].copy()
    df_removed = df_clean[is_outlier].copy()
    
    rows_removed = initial_rows - len(df_retained)
    
    print("\n--- Data Clean Result ---")
    print(f"Origin {initial_rows} rows")
    print(f"Remove {rows_removed} rows (coordinate outlier/unclassified/hit_height=0/Error Case)")
    print(f"Final {len(df_retained)} rows")
    
    return df_retained, df_removed

def main_clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean pipeline.
    """
    print("\n[Start Data Clean]...")
    df_retained, df_removed = clean_coordinate_outliers(df)
    df_final = remove_outlier_rallies(df_retained, df_removed)
    print("\n[Data Clean Finished]")
    return df_final
