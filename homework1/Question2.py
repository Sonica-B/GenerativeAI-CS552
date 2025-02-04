import numpy as np
import matplotlib.pyplot as plt
import os

def train_ppca(X, d):

    n, m = X.shape
    mu = np.mean(X, axis=0)  # Compute mean of the data
    X_centered = X - mu  # Center the data


    S = (X_centered.T @ X_centered) / n


    eigenvals, eigenvecs = np.linalg.eigh(S)


    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]


    sigma2 = np.mean(eigenvals[d:])
    W = eigenvecs[:, :d] @ np.sqrt(np.diag(eigenvals[:d] - sigma2))  # Projection matrix

    return W, mu, sigma2

def compute_latent(x, W, mu, sigma2):

    M = W.T @ W + sigma2 * np.eye(W.shape[1])
    z = np.linalg.solve(M, W.T @ (x - mu))
    return z

def reconstruct_image(z, W, mu):
    return W @ z + mu

def plot_scatter(z, title="Latent Space Visualization"):
    plt.figure(figsize=(8, 8))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_face_grid(faces, n_rows, n_cols, title="Face Reconstructions"):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    for idx, ax in enumerate(axes.flat):
        if idx < len(faces):
            ax.imshow(faces[idx].reshape(24, 24), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def question2b(X):
    W, mu, sigma2 = train_ppca(X, d=2)
    z_all = np.array([compute_latent(x, W, mu, sigma2) for x in X])
    plot_scatter(z_all, title="Latent Space Visualization (d=2)")

def reconstruct(X, d):

    indices = np.random.choice(len(X), 25, replace=False)
    X_selected = X[indices]


    W, mu, sigma2 = train_ppca(X, d)

    # Compute latent vectors for the selected images
    z_hat = np.array([compute_latent(x, W, mu, sigma2) for x in X_selected])

    # Reconstruct the images
    x_reconstructed = np.array([reconstruct_image(z, W, mu) for z in z_hat])

    plot_face_grid(X_selected, 5, 5, title=f"Original Images (d={d})")

    plot_face_grid(x_reconstructed, 5, 5, title=f"Reconstructed Images (d={d})")

def generate_faces(X, d, num_images=100):

    W, mu, sigma2 = train_ppca(X, d)

    # Generate new latent vectors
    z_samples = np.random.normal(0, 1, (num_images, d))

    # Reconstruct images from the latent vectors
    generated_faces = np.array([reconstruct_image(z, W, mu) for z in z_samples])

    plot_face_grid(generated_faces, 10, 10, title="Generated Faces")


def perturb_analysis(X, d=16, n_perturbations=7, perturbation_range=5):

    selected_idx = np.random.randint(len(X))
    original_image = X[selected_idx]

    n, m = X.shape
    mu = np.mean(X, axis=0)
    X_centered = X - mu


    S = (X_centered.T @ X_centered) / n
    eigenvals, eigenvecs = np.linalg.eigh(S)

    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Compute MLE estimates
    sigma2 = np.mean(eigenvals[d:])
    W = eigenvecs[:, :d] @ np.diag(np.sqrt(np.maximum(eigenvals[:d] - sigma2, 0)))

    # Compute M matrix and latent representation
    M = W.T @ W + sigma2 * np.eye(d)
    z_hat = np.linalg.solve(M, W.T @ (original_image - mu))

    perturbations = np.linspace(-perturbation_range, perturbation_range, n_perturbations)

    dims_to_perturb = range(5)

    fig, axes = plt.subplots(len(dims_to_perturb), n_perturbations,
                             figsize=(2 * n_perturbations, 2 * len(dims_to_perturb)))

    # Generate perturbed images
    for i, dim in enumerate(dims_to_perturb):
        for j, perturb in enumerate(perturbations):
            # Create perturbed latent vector
            z_perturbed = z_hat.copy()
            z_perturbed[dim] += perturb

            x_reconstructed = W @ z_perturbed + mu

            axes[i, j].imshow(x_reconstructed.reshape(24, 24), cmap='gray')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'δ={perturb:.3f}')
            if j == 0:
                axes[i, j].set_ylabel(f'z_{dim + 1}')

    plt.suptitle('Face Image Perturbation Analysis\nRows: Different dimensions, Columns: Perturbation magnitude')
    plt.tight_layout()

    return fig

if __name__ == "__main__":
    faces = np.load('eigenfaces.npy')
    X = faces.reshape(len(faces), -1)  # Flatten each image into a vector

    # Part b: Train p-PCA with d=2 and visualize latent space
    question2b(X)

    # Part c: Reconstruct images for d=16, 32, 64
    for d in [16, 32, 64]:
        reconstruct(X, d)

    # Part d: Generate new faces
    generate_faces(X, d=32, num_images=100)

    # Part e: Perturbation analysis
    fig = perturb_analysis(X, d=16)
    plt.show()