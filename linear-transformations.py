import numpy as np
import matplotlib.pyplot as plt


def visualize_transformation(matrix, title):
    """
    Show what a 2x2 matrix does to a unit square.
    Blue = before, Red = after.
    """
    # Unit square corners: (0,0), (1,0), (1,1), (0,1), (0,0)
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])  # shape (2, 5)

    # Apply transformation: matrix @ square
    # (2,2) @ (2,5) = (2,5)
    transformed = matrix @ square

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Before
    ax1.plot(square[0], square[1], 'b-o', linewidth=2, label='Original')
    ax1.fill(square[0], square[1], alpha=0.2, color='blue')
    ax1.set_xlim(-3, 3);
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.set_aspect('equal');
    ax1.set_title('Before');
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # After
    ax2.plot(square[0], square[1], 'b--', linewidth=1, alpha=0.4, label='Original')
    ax2.plot(transformed[0], transformed[1], 'r-o', linewidth=2, label='Transformed')
    ax2.fill(transformed[0], transformed[1], alpha=0.2, color='red')
    ax2.set_xlim(-3, 3);
    ax2.set_ylim(-3, 3)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_aspect('equal');
    ax2.set_title(f'After: {title}');
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Matrix: {matrix.tolist()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'transform_{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


# Try all three transformations
theta = np.pi / 4  # 45 degrees

rotation = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

scaling = np.array([
    [2.0, 0.0],
    [0.0, 0.5]
])

shear = np.array([
    [1.0, 1.0],
    [0.0, 1.0]
])

visualize_transformation(rotation, "Rotation 45°")
visualize_transformation(scaling, "Scaling 2x, 0.5y")
visualize_transformation(shear, "Shear")