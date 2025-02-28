import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Define the true data distribution - mixture of Gaussians
def sample_data(n_samples):
    # Create a mixture of 3 Gaussians
    components = [
        {'mean': -2.0, 'std': 0.5, 'weight': 0.3},
        {'mean': 0.0, 'std': 0.4, 'weight': 0.4},
        {'mean': 2.0, 'std': 0.6, 'weight': 0.3}
    ]

    samples = []
    for _ in range(n_samples):
        # Randomly select a component based on weights
        probs = [comp['weight'] for comp in components]
        idx = np.random.choice(len(components), p=probs)

        # Sample from the selected Gaussian
        mean = components[idx]['mean']
        std = components[idx]['std']
        sample = np.random.normal(mean, std)
        samples.append(sample)

    return np.array(samples)


# Generator Network - takes uniform noise and outputs samples
class Generator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)


# Discriminator Network - takes samples and outputs probability
class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Training function
def train_gan():
    # Create output directory
    output_dir = "Outputs/"
    os.makedirs(output_dir, exist_ok=True)

    # Training parameters
    batch_size = 256
    n_epochs = 1000
    num_snapshots = 10
    snapshot_epochs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000]  # Including first and last
    plot_range = (-5, 5)
    n_bins = 50

    # Initialize networks
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers with better parameters for GAN stability
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # List to store snapshot data
    snapshots = []

    # Training loop
    for epoch in range(n_epochs + 1):  # +1 to include epoch 1000
        # Train Discriminator
        for _ in range(3):  # Train discriminator more steps than generator for stability
            # Generate real data batch
            real_data = torch.FloatTensor(sample_data(batch_size)).view(-1, 1)
            real_labels = torch.ones(batch_size, 1)

            # Generate fake data batch
            z = torch.FloatTensor(np.random.uniform(0, 1, size=(batch_size, 1)))
            fake_data = generator(z).detach()  # Detach to avoid training generator
            fake_labels = torch.zeros(batch_size, 1)

            # Reset gradients
            d_optimizer.zero_grad()

            # Train on real data
            real_outputs = discriminator(real_data)
            real_loss = criterion(real_outputs, real_labels)

            # Train on fake data
            fake_outputs = discriminator(fake_data)
            fake_loss = criterion(fake_outputs, fake_labels)

            # Combine losses and update
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        # Generate fake data
        z = torch.FloatTensor(np.random.uniform(0, 1, size=(batch_size, 1)))
        fake_data = generator(z)

        # Get discriminator outputs
        outputs = discriminator(fake_data)

        # Generator loss (want discriminator to think outputs are real)
        g_loss = criterion(outputs, torch.ones(batch_size, 1))
        g_loss.backward()
        g_optimizer.step()

        # Save snapshot if it's in our list of snapshot epochs
        if epoch in snapshot_epochs:
            generator.eval()
            discriminator.eval()

            # Generate samples for histogram
            with torch.no_grad():
                z = torch.FloatTensor(np.random.uniform(0, 1, size=(2000, 1)))
                gen_samples = generator(z).numpy().flatten()

            # Create a grid of points for the discriminator curve
            x_grid = np.linspace(plot_range[0], plot_range[1], 1000)
            x_grid_tensor = torch.FloatTensor(x_grid).view(-1, 1)
            with torch.no_grad():
                d_outputs = discriminator(x_grid_tensor).numpy().flatten()

            # Create plot for this snapshot
            plt.figure(figsize=(10, 6))

            # Plot real data distribution (blue histogram)
            real_data = sample_data(2000)
            plt.hist(real_data, bins=n_bins, range=plot_range, alpha=0.5, color='blue', density=True,
                     label='True data distribution')

            # Plot generated data distribution (red histogram)
            plt.hist(gen_samples, bins=n_bins, range=plot_range, alpha=0.5, color='red', density=True,
                     label='Generated data distribution')

            # Plot discriminator output (orange curve)
            plt.plot(x_grid, d_outputs, color='orange', lw=2, label='Discriminator output D(G(z))')

            plt.xlim(plot_range)
            plt.ylim(0, 1.05)
            plt.title(f"GAN Training Progress - Epoch {epoch}")
            plt.xlabel("x")
            plt.ylabel("Density / D(x)")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.savefig(os.path.join(output_dir,f"{epoch:04d}.png"))
            plt.close()

            # Store this snapshot
            snapshots.append({
                'epoch': epoch,
                'gen_samples': gen_samples,
                'x_grid': x_grid,
                'd_outputs': d_outputs
            })

            # Return to training mode
            generator.train()
            discriminator.train()

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    print("Training complete. All snapshots saved.")
    return snapshots


if __name__ == "__main__":
    train_gan()