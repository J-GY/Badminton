import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_badminton_intensity(match_df, shot_df, output_name='intensity_analysis.png'):
    try:
        # duration per set with diff level
        df = match_df.copy()
        df['avg_duration_per_set'] = df['duration'] / df['set_count']
        
        event_map = {1: 'Men (MD)', 2: 'Women (WD)', 3: 'Mixed (XD)'}
        df['event'] = pd.to_numeric(df['event'], errors='coerce')
        df['Match Type'] = df['event'].map(event_map)
        df = df.dropna(subset=['Match Type'])

        def merge_levels(level):
            level = str(level)
            return '300 & 500' if level in ['300', '500'] else level

        df['merge_level'] = df['level'].apply(merge_levels)
        level_order = ['300 & 500', '750', '1000', 'Others']
        df['merge_level'] = pd.Categorical(df['merge_level'], categories=level_order, ordered=True)

        # shot number per SET with diff level (修改這裡)
        shot_copy = shot_df.copy()
        shot_copy['merge_level'] = shot_copy['level'].apply(merge_levels)
        
        # 1. 算出各層級的總擊球數
        total_shots_by_level = shot_copy.groupby('merge_level').size()
        
        # 2. 算出各層級的「總局數」，而不是總場次 (將 set_count 加總)
        total_sets_by_level = df.groupby('merge_level', observed=False)['set_count'].sum()
        
        # 3. 相除得到平均每局擊球數
        avg_shots_per_set = (total_shots_by_level / total_sets_by_level).reindex(level_order)

        # plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        sns.set_style("whitegrid") 

        # x axis
        sns.stripplot(
            data=df, 
            x='merge_level', 
            y='avg_duration_per_set', 
            hue='Match Type', 
            palette='Set1', 
            jitter=0.25,    
            alpha=0.7,       # 點的顏色透明度
            size=9,        # 點的大小
            order=level_order, 
            ax=ax1
        )
        
        # Left y
        ax1.set_ylabel('Avg Minutes per Set', fontsize=13) #fontweight='bold'
        ax1.set_xlabel('Tournament Level (Super Series)', fontsize=13)

        # right y
        ax2 = ax1.twinx()
        
        # avg shot per SET (修改變數名稱與標籤)
        ax2.plot(
            level_order, 
            avg_shots_per_set.values, 
            color='#777777',
            marker='d',          
            markersize=7,        
            linestyle='--',      
            linewidth=2,         
            label='Avg Shots per Set'
        )

        # right y
        ax2.set_ylabel('Average Shots per Set', fontsize=13) #fontweight='bold'
        ax2.grid(False) 

        # merge
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', title='Intensity Metrics')

        plt.title('Badminton Level: Duration vs. Average Shot', fontsize=16, pad=20)
        
        save_path = f"./img/matches/{output_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved plot to: {save_path}")
        
        plt.show()
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")