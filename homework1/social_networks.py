import numpy as np
import pickle
import matplotlib.pyplot as plt
from graph_viz import render_graph

def sigmoid(z):
    """Numerically stable sigmoid function"""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


class SocialNetworkModel:
    def __init__(self):
        """Initialize model parameters randomly"""
        self.a = np.random.normal(0, 0.1)
        self.b = np.random.normal(0, 0.1)

    def compute_log_likelihood_and_gradients(self, X, E):
        """
        Compute log likelihood and its gradients w.r.t. a and b
        Following the MLE derivation from part (a)
        """
        n = X.shape[0]
        grad_a = 0.0
        grad_b = 0.0
        ll = 0.0

        # Get indices for i < j using np.tril_indices as suggested
        indices = np.tril_indices(n, k=-1)

        for idx in range(len(indices[0])):
            i, j = indices[0][idx], indices[1][idx]
            x_dot = np.dot(X[i], X[j])
            z = self.a * x_dot + self.b
            p_ij = sigmoid(z)

            # Compute log-likelihood contribution
            ll += E[i, j] * np.log(p_ij + 1e-10) + \
                  (1 - E[i, j]) * np.log(1 - p_ij + 1e-10)

            # Compute gradient contributions
            diff = E[i, j] - p_ij  # This is (eij - σ)
            grad_a += diff * x_dot
            grad_b += diff

        return ll, grad_a, grad_b

    def train(self, X, E, learning_rate=0.01, n_iterations=1000):
        """Train model using gradient ascent"""
        history = []

        for iter in range(n_iterations):
            # Compute log likelihood and gradients
            ll, grad_a, grad_b = self.compute_log_likelihood_and_gradients(X, E)

            # Update parameters using gradient ascent
            self.a += learning_rate * grad_a
            self.b += learning_rate * grad_b

            # Store history
            history.append({
                'iteration': iter + 1,
                'a': self.a,
                'b': self.b,
                'll': ll
            })

            # Print last 10 iterations as required
            if iter >= n_iterations - 10:
                print(f"Iteration {iter + 1:4d}: a = {self.a:8.4f}, b = {self.b:8.4f}, "
                      f"log-likelihood = {ll:10.4f}")

        return history

    def generate_graph(self, n_students=15, n_features=3):
        """Generate a new graph using learned parameters"""
        # Generate random features uniformly from [-1,+1]
        X = np.random.uniform(-1, 1, (n_students, n_features))

        # Initialize empty adjacency matrix
        E = np.zeros((n_students, n_students))

        # Generate edges
        for i in range(n_students):
            for j in range(i + 1, n_students):
                # Compute edge probability
                z = self.a * np.dot(X[i], X[j]) + self.b
                p_ij = sigmoid(z)

                # Sample edge and ensure symmetry
                edge = np.random.binomial(1, p_ij)
                E[i, j] = E[j, i] = edge

        return X, E


def main():
    # Load data
    with open('classroom_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    # Use first graph for training
    X = graphs[0][0]  # Features from first graph
    E = graphs[0][1]  # Adjacency matrix from first graph

    # Train model
    print("Training model...")
    model = SocialNetworkModel()
    history = model.train(X, E)

    # Print final parameters
    print("\nFinal parameter estimates:")
    print(f"a = {model.a:.6f}")
    print(f"b = {model.b:.6f}")

    # Plot training convergence
    plt.figure(figsize=(10, 5))
    plt.plot([h['iteration'] for h in history],
             [h['ll'] for h in history])
    plt.title('Log-Likelihood During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plt.show()

    # Generate and visualize 5 new graphs
    print("\nGenerating 5 graphs...")
    for i in range(5):
        X_new, E_new = model.generate_graph()
        plt.figure(figsize=(10, 10))
        plt.title(f"Generated Graph {i + 1}")
        render_graph(X_new, E_new)  # Using the provided visualization function


if __name__ == "__main__":
    main()