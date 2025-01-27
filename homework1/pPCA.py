import numpy as np
import matplotlib.pyplot as plt


def train_ppca(X, d):
    """
    Train p-PCA model with latent dimension d.
    Returns W, mu, sigma2
    """
    n, m = X.shape
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Compute sample covariance and eigendecomposition
    S = (X_centered.T @ X_centered) / n
    eigenvals, eigenvecs = np.linalg.eigh(S)

    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # MLE estimates
    sigma2 = np.mean(eigenvals[d:])
    W = eigenvecs[:, :d] @ np.sqrt(np.diag(eigenvals[:d] - sigma2))

    return W, mu, sigma2


def compute_latent(x, W, mu, sigma2):
    """Compute E[z|x] using np.linalg.solve"""
    M = W.T @ W + sigma2 * np.eye(W.shape[1])
    return np.linalg.solve(M, W.T @ (x - mu))


def reconstruct_image(z, W, mu):
    """Compute E[x|z]"""
    return W @ z + mu


def plot_scatter(z, title="Latent Space Visualization"):
    """Create scatter plot of 2D latent vectors"""
    plt.figure(figsize=(8, 8))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_face_grid(faces, n_rows, n_cols, title="Face Reconstructions"):
    """Plot grid of face images"""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(faces):
            ax.imshow(faces[idx].reshape(24, 24), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    faces = np.load('eigenfaces.npy')
    X = faces.reshape(len(faces), -1)

    # Part a & b: Train model with d=2 and visualize latent space
    W, mu, sigma2 = train_ppca(X, d=2)
    z_all = np.array([compute_latent(x, W, mu, sigma2) for x in X])
    plot_scatter(z_all)

    # Part c: Reconstruct images with different d values
    for d in [16, 32, 64]:
        W, mu, sigma2 = train_ppca(X, d)
        # Select 25 random images
        indices = np.random.choice(len(X), 25, replace=False)
        reconstructions = []
        for idx in indices:
            z = compute_latent(X[idx], W, mu, sigma2)
            reconstructions.append(reconstruct_image(z, W, mu))
        plot_face_grid(reconstructions, 5, 5, f"Reconstructions (d={d})")

    # Part d: Generate new faces
    d = 32  # Choose appropriate d
    W, mu, sigma2 = train_ppca(X, d)
    z_samples = np.random.normal(0, 1, (100, d))
    generated = [reconstruct_image(z, W, mu) for z in z_samples]
    plot_face_grid(generated, 10, 10, "Generated Faces")

    # Part e: Perturbation analysis
    d = 16
    W, mu, sigma2 = train_ppca(X, d)
    # Select random image
    x_random = X[np.random.randint(len(X))]
    z_base = compute_latent(x_random, W, mu, sigma2)

    # Create perturbations
    dims = np.random.choice(d, 5, replace=False)
    deltas = np.linspace(-0.15, 0.15, 10)
    perturbed = []

    for dim in dims:
        for delta in deltas:
            z_pert = z_base.copy()
            z_pert[dim] += delta
            perturbed.append(reconstruct_image(z_pert, W, mu))

    plot_face_grid(perturbed, 5, 10, "Perturbation Analysis")


if __name__ == "__main__":
    main()