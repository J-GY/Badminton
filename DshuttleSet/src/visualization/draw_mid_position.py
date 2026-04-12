import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 1. 核心幾何參數 (完全根據您提供的數值)
# ==========================================
SQUARE_SIZE = 20
RADIUS = 15  # 圓形半徑 (Spatial Units，對應 1.5M)

MID_X = 30.5
DISTANCE_TO_CENTER = 12.5

# 下半場基準線 (Bottom Court) - Y=47 是短發球線，往下偏移
MID_Y_BOTTOM = 47 - DISTANCE_TO_CENTER # 34.5

# 上半場基準線 (Top Court) - Y=87 是短發球線，往上偏移
MID_Y_TOP = 87 + DISTANCE_TO_CENTER    # 99.5


# ==========================================
# 2. 綠色球場背景與白線繪製函數
# ==========================================
def draw_court_background_green(ax, full_wid=61.0, half_len=67.0):
    """
    繪製標準羽球半場 (全綠底 + 白線風格) + 座標標記。
    並縮緊邊界以消除多餘空白。
    """
    COURT_COLOR = '#90C080'      # 與熱力圖一致的綠色底
    LINE_COLOR = 'white'         # 白色界線
    lw = 2                       # 線條粗細

    # 1. 設定背景顏色
    ax.set_facecolor(COURT_COLOR)

    # 2. 繪製標準白線
    # 外框 (雙打邊界)
    rect = patches.Rectangle((0, 0), full_wid, half_len, linewidth=lw, edgecolor=LINE_COLOR, facecolor='none', zorder=1)
    ax.add_patch(rect)
    
    # 單打邊線 (左右縮進 4.6)
    margin = 4.6
    ax.plot([margin, margin], [0, half_len], color=LINE_COLOR, linewidth=lw, zorder=1)
    ax.plot([full_wid-margin, full_wid-margin], [0, half_len], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    # 雙打後發球線 (距底線 7.6)
    long_service_dist = 7.6
    ax.plot([0, full_wid], [long_service_dist, long_service_dist], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    # 短發球線 (距網子 19.8 -> 換算座標 y = 67 - 19.8 = 47.2)
    # 代碼中使用 47 作為基準
    short_service_y = 47 # 簡化為 47
    ax.plot([0, full_wid], [short_service_y, short_service_y], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    # 中線 (從底線畫到短發球線)
    mid_x = full_wid / 2
    ax.plot([mid_x, mid_x], [0, short_service_y], color=LINE_COLOR, linewidth=lw, zorder=1)

    # 3. 新增座標標記 (0,0), (61,0), (0,67)
    text_style = {
        'color': 'white', 
        'fontsize': 10, 
        'fontweight': 'bold',
        'zorder': 5
    }
    offset = 1.0 
    ax.text(0 - offset, 0 - offset, '(0,0)', ha='right', va='top', **text_style)
    ax.text(full_wid + offset, 0 - offset, '(61,0)', ha='left', va='top', **text_style)
    ax.text(0 - offset, half_len + offset, '(0,67)', ha='right', va='bottom', **text_style)


# ==========================================
# 3. 繪製中位覆蓋示意圖的主函數
# ==========================================
def draw_coverage_schematic(filename='cce_schematic_radius.pdf'):
    """
    根據參數繪製圓形中位覆蓋示意圖 (CCE)，風格與熱力圖統一。
    不加大標題，並收緊畫布範圍。
    """
    # 建立畫布，設定適合單個半場的比例 (6:8)
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 1. 畫出球場底色、白線和座標標記
    # 下半場全寬 61，半長 67
    draw_court_background_green(ax, full_wid=61.0, half_len=67.0)
    
    # ==========================
    # 2. 繪製戰略覆蓋區域 (圓形判定)
    # ==========================
    # 設定補位區域的幾何中心點 (X_c, Y_c)
    # 因為只畫單個半場，所以 Y_c 使用 MID_Y_BOTTOM (34.5)
    X_c = MID_X
    Y_c = MID_Y_BOTTOM
    
    # 畫出中心防守樞紐點 (Defensive Pivot Point)
    ax.scatter(X_c, Y_c, color='#FF4444', s=100, marker='.', edgecolor='white', linewidth=1.5, zorder=10)
    
    # 建立圓形補位區域 ($\mathcal{Z}_{cover}$) patches
    # 半徑使用您提供的 RADIUS = 15 (對應物理距離 1.5M)
    circle_zone = patches.Circle(
        (X_c, Y_c), 
        RADIUS, 
        edgecolor="#FF2C2C", # 使用熱力圖中的 Offensive 橘紅色
        facecolor='#FF2C2C', 
        alpha=0.5,          # 半透明，以便透出底線
        linewidth=2, 
        zorder=5             # zorder=5 確保蓋在白線上
    )
    ax.add_patch(circle_zone)
    
    # ==========================
    # 3. 調整畫布視野與風格 (與熱力圖一致)
    # ==========================
    # 消除球場上方空白：將 Y 軸邊界從 (-15, 149) 收緊到 (-2, 70)
    # X 軸也收緊，保留空間給座標文字即可
    ax.set_xlim(-8, 69) 
    ax.set_ylim(-2, 70) 
    
    # 設定等比例
    ax.set_aspect('equal')
    
    # 移除 matplotlib 自帶的座標軸線和刻度
    ax.axis('off')
    
    # 填滿整張圖片的外框背景為綠色，確保編譯 LaTeX 時無白邊
    fig.patch.set_facecolor('#90C080')
    
    plt.tight_layout()
    
    # 存檔為高畫質向量 PDF，這在 Overleaf 裡不管怎麼放大都不會模糊
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#90C080')
    plt.show() 
    print(f"補位示意圖已成功儲存: {filename}")


if __name__ == "__main__":
    # 執行繪圖並存檔
    draw_coverage_schematic('cce_schematic_radius.pdf')