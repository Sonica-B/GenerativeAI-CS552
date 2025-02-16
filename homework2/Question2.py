import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os


def compute_elbo(x, theta, phi, num_samples=10000):
    """Compute ELBO for given parameters using Monte Carlo sampling."""
    # Sample z from q(z|x) = N(φ, 1)
    z = phi + torch.randn(num_samples)

    # Compute all terms including normalization constants
    log_p_x_given_z = -0.5 * ((x - theta * z).pow(2) + np.log(2 * np.pi))
    log_p_z = -0.5 * (z.pow(2) + np.log(2 * np.pi))
    log_q_z_given_x = -0.5 * ((z - phi).pow(2) + np.log(2 * np.pi))

    # Compute ELBO with better numerical stability
    elbo = (log_p_x_given_z + log_p_z - log_q_z_given_x).mean()

    return elbo.item()


def compute_true_log_likelihood(x, theta, num_points=10000):
    """Compute true log likelihood using numerical integration."""
    z = torch.linspace(-10, 10, num_points)
    dz = z[1] - z[0]

    # Include normalization constants
    log_p_x_given_z = -0.5 * ((x - theta * z).pow(2) + np.log(2 * np.pi))
    log_p_z = -0.5 * (z.pow(2) + np.log(2 * np.pi))

    # Use log-sum-exp trick for numerical stability
    log_joint = log_p_x_given_z + log_p_z
    max_log_joint = torch.max(log_joint)
    log_likelihood = max_log_joint + torch.log(
        torch.sum(torch.exp(log_joint - max_log_joint)) * dz
    )

    return log_likelihood.item()


# Set style parameters
plt.style.use('sns')
plt.figure(figsize=(12, 8))

# Generate data
x = torch.tensor(1.0)
theta_range = torch.linspace(-10, 10, 500)  # Increased resolution
phi_values = np.linspace(-3, 3, 7)  # Specific phi values

# Create custom colormap from cyan to yellow
colors = [(0, 1, 1), (0.3, 1, 0.7), (0.6, 1, 0.4), (0.9, 1, 0)]
cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(phi_values))

# Plot ELBO curves
for i, phi in enumerate(phi_values):
    color = cmap(i / (len(phi_values) - 1))
    elbos = [compute_elbo(x, theta, phi) for theta in theta_range]
    plt.plot(theta_range, elbos, color=color, alpha=0.8,
             label=f'ELBO (φ={phi:.1f})')

# Plot true log-likelihood
true_ll = [compute_true_log_likelihood(x, theta) for theta in theta_range]
plt.plot(theta_range, true_ll, 'k-', linewidth=2.5, label='True Log-Likelihood')

# Customize plot
plt.xlabel('θ', fontsize=12)
plt.ylabel('log P(x)', fontsize=12)
plt.title('True Log-Likelihood vs ELBO', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(-5, 0)
plt.xlim(-10, 10)

# Adjust layout and save
plt.tight_layout()
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'elbo_visualization.png'),
            dpi=300, bbox_inches='tight')
plt.close()