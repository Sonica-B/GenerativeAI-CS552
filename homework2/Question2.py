import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define a VAE with scalar parameters θ (decoder) and ϕ (encoder)
class ScalarVAE(nn.Module):
    def __init__(self):
        super(ScalarVAE, self).__init__()
        self.phi = nn.Parameter(torch.randn(1))  # Encoder mean parameter
        self.theta = nn.Parameter(torch.randn(1))  # Decoder parameter
        self.fixed_logvar = torch.zeros(1)  # Fixed encoder variance (σ² = 1)

    def encode(self, x):
        return self.phi, self.fixed_logvar  # q(z|x) = N(ϕ, 1)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.theta * z  # p(x|z) = N(θz, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# True log-likelihood via brute-force integration
def true_log_likelihood(x, theta, z_samples=10000):
    z = torch.linspace(-5, 5, z_samples)  # Numerical integration over z
    mu_x = theta * z
    log_p_x_given_z = -0.5 * (x - mu_x).pow(2)  # log N(x; θz, 1)
    log_p_z = -0.5 * z.pow(2)  # log N(z; 0, 1)
    log_p_x = torch.logsumexp(log_p_x_given_z + log_p_z, dim=0) - np.log(z_samples)
    return log_p_x.item()


# Training setup
model = ScalarVAE()
optimizer = optim.Adam(model.parameters(), lr=0.05)
x = torch.tensor([1.0])  # Single 1D data point

# Trackers
true_lls, elbos, thetas, phis = [], [], [], []

# Training loop (maximize ELBO)
for step in range(1000):
    optimizer.zero_grad()
    recon_x, mu, logvar = model(x)

    # ELBO = Reconstruction - KL
    recon_loss = 0.5 * (x - recon_x).pow(2).sum()  # -log p(x|z) ∝ MSE
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum()  # KL(q||p)
    elbo = -(recon_loss + kl)  # Maximize ELBO = minimize loss
    (-elbo).backward()  # Gradient ascent on ELBO
    optimizer.step()

    # Track metrics
    theta = model.theta.item()
    phi = model.phi.item()
    true_ll = true_log_likelihood(x, torch.tensor(theta))
    true_lls.append(true_ll)
    elbos.append(elbo.item())
    thetas.append(theta)
    phis.append(phi)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(thetas, true_lls, label="True Log-Likelihood", color="black", linewidth=2)
plt.plot(thetas, elbos, label="ELBO", color="blue", alpha=0.8)
plt.xlabel("Decoder Parameter (θ)")
plt.ylabel("Value")
plt.legend()
plt.title("True Log-Likelihood vs ELBO During Training")
plt.show()