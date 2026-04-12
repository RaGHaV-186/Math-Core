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