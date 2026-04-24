import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

SQUARE_SIZE = 20
RADIUS = 15  
MID_X = 30.5
DISTANCE_TO_CENTER = 12.5
MID_Y_BOTTOM = 47 - DISTANCE_TO_CENTER # 34.5
MID_Y_TOP = 87 + DISTANCE_TO_CENTER    # 99.5

def draw_court_background_green(ax, full_wid=61.0, half_len=67.0):
    COURT_COLOR = '#90C080'
    LINE_COLOR = 'white'
    lw = 2

    ax.set_facecolor(COURT_COLOR)

    rect = patches.Rectangle((0, 0), full_wid, half_len, linewidth=lw, edgecolor=LINE_COLOR, facecolor='none', zorder=1)
    ax.add_patch(rect)
    
    margin = 4.6
    ax.plot([margin, margin], [0, half_len], color=LINE_COLOR, linewidth=lw, zorder=1)
    ax.plot([full_wid-margin, full_wid-margin], [0, half_len], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    long_service_dist = 7.6
    ax.plot([0, full_wid], [long_service_dist, long_service_dist], color=LINE_COLOR, linewidth=lw, zorder=1)

    short_service_y = 47 
    ax.plot([0, full_wid], [short_service_y, short_service_y], color=LINE_COLOR, linewidth=lw, zorder=1)
    
    mid_x = full_wid / 2
    ax.plot([mid_x, mid_x], [0, short_service_y], color=LINE_COLOR, linewidth=lw, zorder=1)

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

def draw_coverage_schematic(filename='cce_schematic_radius.pdf'):
    fig, ax = plt.subplots(figsize=(6, 8))

    draw_court_background_green(ax, full_wid=61.0, half_len=67.0)
    
    X_c = MID_X
    Y_c = MID_Y_BOTTOM
    
    ax.scatter(X_c, Y_c, color='#FF4444', s=100, marker='.', edgecolor='white', linewidth=1.5, zorder=10)
    
    circle_zone = patches.Circle(
        (X_c, Y_c), 
        RADIUS, 
        edgecolor="#FF2C2C",
        facecolor='#FF2C2C', 
        alpha=0.5,     
        linewidth=2, 
        zorder=5            
    )
    ax.add_patch(circle_zone)
    
    ax.set_xlim(-8, 69) 
    ax.set_ylim(-2, 70) 
    
    ax.set_aspect('equal')
    
    ax.axis('off')
    
    fig.patch.set_facecolor('#90C080')
    
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#90C080')
    plt.show() 

if __name__ == "__main__":
    draw_coverage_schematic('cce_schematic_radius.pdf')