import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_elbo(x, theta, phi, num_samples=1000):
    """Compute ELBO for given parameters using Monte Carlo sampling."""
    # Sample z from q(z|x) = N(φ, 1)
    z = phi + torch.randn(num_samples)

    # Compute log p(x|z) = log N(θz, 1)
    log_p_x_given_z = -0.5 * (x - theta * z).pow(2) - 0.5 * np.log(2 * np.pi)

    # Compute log p(z) = log N(0, 1)
    log_p_z = -0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi)

    # Compute log q(z|x) = log N(φ, 1)
    log_q_z_given_x = -0.5 * (z - phi).pow(2) - 0.5 * np.log(2 * np.pi)

    # ELBO = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]
    elbo = (log_p_x_given_z + log_p_z - log_q_z_given_x).mean()

    return elbo.item()


def compute_true_log_likelihood(x, theta, num_points=10000):
    """Compute true log likelihood using numerical integration."""
    # Create integration grid
    z = torch.linspace(-10, 10, num_points)
    dz = z[1] - z[0]

    # Compute log p(x|z)p(z)
    log_p_x_given_z = -0.5 * (x - theta * z).pow(2)  # log N(θz, 1)
    log_p_z = -0.5 * z.pow(2)  # log N(0, 1)

    # Use log-sum-exp trick for numerical stability
    log_joint = log_p_x_given_z + log_p_z
    max_log_joint = torch.max(log_joint)
    log_likelihood = max_log_joint + torch.log(
        torch.sum(torch.exp(log_joint - max_log_joint)) * dz
    ) - np.log(2 * np.pi)

    return log_likelihood.item()


# Set up the plot
x = torch.tensor(1.0)  # Observed data point
theta_range = torch.linspace(-10, 10, 200)
phi_values = [-3, -2, -1, 0, 1, 2, 3]  # Different encoder means to try
colors = plt.cm.viridis(np.linspace(0, 1, len(phi_values)))

plt.figure(figsize=(10, 6))

# Plot ELBO for different phi values
for phi, color in zip(phi_values, colors):
    elbos = [compute_elbo(x, theta, phi) for theta in theta_range]
    plt.plot(theta_range, elbos, color=color, alpha=0.7,
             label=f'ELBO (φ={phi})')

# Plot true log-likelihood
true_ll = [compute_true_log_likelihood(x, theta) for theta in theta_range]
plt.plot(theta_range, true_ll, 'k-', linewidth=2, label='True Log-Likelihood')

plt.xlabel('θ')
plt.ylabel('log P(x)')
plt.title('True Log-Likelihood vs ELBO for Different Encoder Parameters (φ)')
plt.legend()
plt.grid(True)
plt.ylim(-5, 0)
plt.show()