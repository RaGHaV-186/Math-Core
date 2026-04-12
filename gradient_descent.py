import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Our loss function: f(x) = (x - 3)^2"""
    return (x - 3) ** 2


def df(x):
    """Derivative: f'(x) = 2(x - 3)"""
    return 2 * (x - 3)


def gradient_descent_1d(starting_x, learning_rate, n_steps):
    x = starting_x
    history_x = [x]
    history_loss = [f(x)]

    for step in range(n_steps):
        gradient = df(x)
        x = x - learning_rate * gradient  # THE core update rule

        history_x.append(x)
        history_loss.append(f(x))

        if step % 10 == 0:
            print(f"Step {step:3d} | x = {x:.4f} | loss = {f(x):.4f} | gradient = {gradient:.4f}")

    return history_x, history_loss


# Run it
print("=== Gradient Descent on f(x) = (x-3)^2 ===")
history_x, history_loss = gradient_descent_1d(
    starting_x=0.0,
    learning_rate=0.1,
    n_steps=50
)

print(f"\nFinal x: {history_x[-1]:.6f} (true minimum: 3.0)")


def plot_1d_descent(history_x, history_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Path on the function curve
    x_range = np.linspace(-1, 7, 300)
    ax1.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x) = (x-3)²')

    # Plot the descent path as dots
    ax1.scatter(history_x, history_loss,
                c=range(len(history_x)), cmap='Reds',
                s=50, zorder=5, label='GD steps')

    # Arrows showing movement direction
    for i in range(min(8, len(history_x) - 1)):
        ax1.annotate('',
                     xy=(history_x[i + 1], history_loss[i + 1]),
                     xytext=(history_x[i], history_loss[i]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax1.scatter([3], [0], c='green', s=200, marker='*',
                zorder=10, label='True minimum (x=3)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Gradient Descent Path on Loss Surface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss over iterations
    ax2.plot(history_loss, 'r-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss f(x)')
    ax2.set_title('Loss vs Iteration (Convergence Curve)')
    ax2.set_yscale('log')  # Log scale shows convergence clearly
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='green', linestyle='--', label='Minimum loss = 0')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('gradient_descent_1d.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_1d_descent(history_x, history_loss)


def f_2d(x, y):
    """Bowl in 2D: f(x,y) = (x-1)^2 + (y-2)^2"""
    return (x - 1) ** 2 + (y - 2) ** 2


def grad_f_2d(x, y):
    """Gradient: [∂f/∂x, ∂f/∂y] = [2(x-1), 2(y-2)]"""
    df_dx = 2 * (x - 1)
    df_dy = 2 * (y - 2)
    return df_dx, df_dy


def gradient_descent_2d(start_x, start_y, learning_rate, n_steps):
    x, y = start_x, start_y
    path = [(x, y)]
    losses = [f_2d(x, y)]

    for step in range(n_steps):
        gx, gy = grad_f_2d(x, y)
        x = x - learning_rate * gx
        y = y - learning_rate * gy
        path.append((x, y))
        losses.append(f_2d(x, y))

    return path, losses


def plot_2d_descent(path, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot
    x_range = np.linspace(-2, 4, 100)
    y_range = np.linspace(-1, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f_2d(X, Y)

    contour = ax1.contourf(X, Y, Z, levels=20, cmap='Blues_r', alpha=0.8)
    ax1.contour(X, Y, Z, levels=20, colors='white', alpha=0.4, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='Loss value')

    # Plot path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax1.plot(path_x, path_y, 'r-o', markersize=5, linewidth=2,
             label='GD path', zorder=5)
    ax1.scatter([path_x[0]], [path_y[0]], c='yellow', s=200,
                marker='o', zorder=6, label='Start')
    ax1.scatter([1], [2], c='lime', s=200, marker='*',
                zorder=6, label='Minimum (1,2)')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('2D Gradient Descent — Contour View')
    ax1.legend()

    # Loss curve
    ax2.plot(losses, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Convergence')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_descent_2d.png', dpi=150, bbox_inches='tight')
    plt.show()


# Run 2D version
path_2d, losses_2d = gradient_descent_2d(
    start_x=-1.0, start_y=4.5,
    learning_rate=0.1,
    n_steps=40
)
plot_2d_descent(path_2d, losses_2d)
print(f"Final position: ({path_2d[-1][0]:.4f}, {path_2d[-1][1]:.4f})")
print(f"True minimum: (1.0, 2.0)")


def compare_learning_rates():
    """This experiment teaches more than any textbook."""
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    labels = ['0.01 (too slow)', '0.1 (good)', '0.5 (ok)',
              '0.9 (borderline)', '1.1 (diverges!)']

    plt.figure(figsize=(12, 5))

    for lr, color, label in zip(learning_rates, colors, labels):
        _, losses = gradient_descent_1d(0.0, lr, 50)
        # Clip for visualization (diverging LR explodes to millions)
        losses_clipped = [min(l, 50) for l in losses]
        plt.plot(losses_clipped, color=color, linewidth=2, label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Loss (clipped at 50)')
    plt.title('Effect of Learning Rate on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 52)
    plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


compare_learning_rates()