import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_environment_diagram():
    # Create outputs/plots directory if it doesn't exist
    os.makedirs('outputs/plots', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    grid_size = 8
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Your actual environment locations
    locations = {
        (0, 0): {'color': 'green', 'label': '🏡 Home', 'emoji': '🏡'},
        (7, 0): {'color': 'blue', 'label': '💼 Work', 'emoji': '💼'},
        (7, 7): {'color': 'gold', 'label': '⭐ Goal', 'emoji': '⭐'},
        (1, 1): {'color': 'lightblue', 'label': '🏠 Childcare', 'emoji': '🏠'},
        (1, 2): {'color': 'lightblue', 'label': '🏠 Childcare', 'emoji': '🏠'},
        (6, 1): {'color': 'lightgreen', 'label': '📚 Education', 'emoji': '📚'},
        (6, 2): {'color': 'lightgreen', 'label': '📚 Education', 'emoji': '📚'},
        (1, 6): {'color': 'yellow', 'label': '💰 Financial', 'emoji': '💰'},
        (2, 6): {'color': 'yellow', 'label': '💰 Financial', 'emoji': '💰'},
        (6, 6): {'color': 'pink', 'label': '🏥 Healthcare', 'emoji': '🏥'},
        (5, 6): {'color': 'pink', 'label': '🏥 Healthcare', 'emoji': '🏥'},
        (3, 3): {'color': 'purple', 'label': '👥 Community', 'emoji': '👥'},
        (4, 4): {'color': 'lightgreen', 'label': '💬 Counseling', 'emoji': '💬'},
    }
    
    for (x, y), info in locations.items():
        rect = patches.Rectangle((x, y), 1, 1, linewidth=2, 
                               edgecolor='black', facecolor=info['color'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.7, info['emoji'], fontsize=20, ha='center', va='center')
        ax.text(x + 0.5, y + 0.3, info['label'], fontsize=8, ha='center', va='center', weight='bold')
    
    ax.plot(0.5, 0.5, 'ro', markersize=15, label='Agent Start (👩)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title('SMEEF Environment - Single Mother Empowerment\nGrid Layout with Services', 
                 fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Grid X Coordinate')
    ax.set_ylabel('Grid Y Coordinate')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/environment_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Environment diagram saved to outputs/plots/environment_diagram.png")

if __name__ == "__main__":
    create_environment_diagram()