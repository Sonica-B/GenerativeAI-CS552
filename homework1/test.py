import numpy as np
import pickle
import matplotlib.pyplot as plt
from graph_viz import render_graph


def sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class SocialNetworkModel:
    def __init__(self):
        """Initialize model parameters to encourage dense graphs"""
        self.a = 2.0  # Initialize with larger positive value
        self.b = 1.0  # Positive bias encourages more edges

    def compute_log_likelihood(self, features, edges):
        """Compute log likelihood for a single graph using tril_indices"""
        n = len(features)
        i, j = np.tril_indices(n, k=-1)  # Get indices for lower triangular part

        # Compute dot products for pairs (i,j)
        dot_products = np.sum(features[i] * features[j], axis=1)

        # Compute probabilities
        logits = self.a * dot_products + self.b
        probs = sigmoid(logits)

        # Get edge values for pairs (i,j)
        e_ij = edges[i, j]

        # Compute log likelihood
        ll = np.sum(e_ij * np.log(probs + 1e-10) +
                    (1 - e_ij) * np.log(1 - probs + 1e-10))

        # Compute gradients
        grad_a = np.sum((e_ij - probs) * dot_products)
        grad_b = np.sum(e_ij - probs)

        return ll, grad_a, grad_b

    def train(self, graphs, learning_rate=1e-5, n_iterations=1000):
        """Train model using gradient ascent"""
        history = []

        for iteration in range(n_iterations):
            total_ll = 0
            total_grad_a = 0
            total_grad_b = 0

            # Accumulate gradients from all graphs
            for features, edges in graphs:
                ll, grad_a, grad_b = self.compute_log_likelihood(features, edges)
                total_ll += ll
                total_grad_a += grad_a
                total_grad_b += grad_b

            # Update parameters
            self.a += learning_rate * total_grad_a
            self.b += learning_rate * total_grad_b

            # Store history
            history.append({
                'iteration': iteration + 1,
                'log_likelihood': total_ll,
                'a': self.a,
                'b': self.b
            })

            # Print last 10 iterations
            if iteration >= n_iterations - 10:
                print(f"{iteration + 1} = {total_ll:.10f}")

        # Print final parameters
        print("\nFinal parameter estimates:")
        print(f"a = {self.a:.10f}")
        print(f"b = {self.b:.10f}")

        return history

    def generate_graph(self, n_students=15, n_features=3):
        """Generate a random graph with minimum density threshold"""
        while True:
            # Generate features uniformly from [-1,+1]
            features = np.random.uniform(-1, 1, (n_students, n_features))
            edges = np.zeros((n_students, n_students), dtype=np.uint8)

            # Generate edges using i < j pairs
            i, j = np.tril_indices(n_students, k=-1)
            dot_products = np.sum(features[i] * features[j], axis=1)
            probs = sigmoid(self.a * dot_products + self.b)

            # Sample edges independently
            edge_samples = (np.random.random(len(probs)) < probs).astype(np.uint8)

            # Fill in the edges matrix (both i,j and j,i due to symmetry)
            edges[i, j] = edge_samples
            edges[j, i] = edge_samples

            # Check density
            n_possible = n_students * (n_students - 1) // 2
            density = np.sum(edge_samples) / n_possible
            if density >= 0.3:  # Require at least 30% edges
                return features, edges


def main():
    # Load data
    with open('classroom_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    # Train model
    model = SocialNetworkModel()
    print("Last 10 iterations and likelihoods:")
    history = model.train(graphs)

    # Generate 5 example graphs
    print("\nGenerating 5 random graphs...")
    for i in range(5):
        features, edges = model.generate_graph()
        density = np.sum(edges) / (len(edges) * (len(edges) - 1))
        plt.figure(figsize=(8, 8))
        plt.title(f"Generated Graph {i + 1} (density: {density:.2f})")
        render_graph(features, edges)
        plt.show()


if __name__ == "__main__":
    main()