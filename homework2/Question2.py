import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def compute_elbo(x, theta, phi, num_samples=5000):
    """Compute ELBO for given parameters using Monte Carlo sampling."""
    # Sample z from q(z|x) = N(φ, 1)
    z = phi + torch.randn(num_samples)

    # Compute log p(x|z) = log N(θz, 1)
    log_p_x_given_z = -0.5 * (x - theta * z).pow(2)

    # Compute log p(z) = log N(0, 1)
    log_p_z = -0.5 * z.pow(2)

    # Compute log q(z|x) = log N(φ, 1)
    log_q_z_given_x = -0.5 * (z - phi).pow(2)

    # ELBO = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    elbo = (log_p_x_given_z + log_p_z - log_q_z_given_x).mean()

    return elbo.item()


def compute_true_log_likelihood(x, theta, num_points=10000):
    """Compute true log likelihood using numerical integration."""
    z = torch.linspace(-10, 10, num_points)
    dz = z[1] - z[0]

    log_p_x_given_z = -0.5 * (x - theta * z).pow(2)
    log_p_z = -0.5 * z.pow(2)

    log_joint = log_p_x_given_z + log_p_z
    max_log_joint = torch.max(log_joint)
    log_likelihood = max_log_joint + torch.log(
        torch.sum(torch.exp(log_joint - max_log_joint)) * dz
    )

    return log_likelihood.item()


# Create custom colormap from cyan to yellow
colors = [(0, 1, 1), (0, 1, 0.5), (0.5, 1, 0), (1, 1, 0)]  # Cyan to yellow
n_bins = 100
cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

# Set up the plot
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)

x = torch.tensor(1.0)
theta_range = torch.linspace(-10, 10, 300)  # Increased resolution
phi_values = np.linspace(-3, 3, 12)  # More phi values for smoother transition

# Plot ELBO curves
for i, phi in enumerate(phi_values):
    color = cmap(i / (len(phi_values) - 1))
    elbos = [compute_elbo(x, theta, phi) for theta in theta_range]
    plt.plot(theta_range, elbos, color=color, alpha=0.8,
             label=f'ELBO (φ={phi:.1f})' if i % 2 == 0 else "")

# Plot true log-likelihood
true_ll = [compute_true_log_likelihood(x, theta) for theta in theta_range]
plt.plot(theta_range, true_ll, 'k-', linewidth=2, label='True Log-Likelihood')

plt.xlabel('θ')
plt.ylabel('log P(x)')
plt.title('True Log-Likelihood vs ELBO')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(-5, 0)
plt.xlim(-10, 10)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Create output directory if it doesn't exist
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(output_dir, 'elbo_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()