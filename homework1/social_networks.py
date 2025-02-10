import numpy as np
import pickle
import matplotlib.pyplot as plt
from graph_viz import render_graph


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class SocialNetworkModel:
    def __init__(self):
        self.a = np.random.normal(0, 0.1)  # Small normal initialization
        self.b = np.random.normal(0, 0.1)

    def compute_log_likelihood_and_gradients(self, X, E):

        n = X.shape[0]

        # Get pairs of indices for i < j
        i, j = np.tril_indices(n, k=-1)

        # Compute x(i)ᵀx(j) for all pairs
        dot_products = np.sum(X[i] * X[j], axis=1)

        # Compute σ(ax(i)ᵀx(j) + b) for all pairs
        logits = self.a * dot_products + self.b
        probs = sigmoid(logits)

        # Get edge values for pairs (i,j)
        e_ij = E[i, j]

        # Compute log likelihood
        ll = np.sum(e_ij * np.log(probs + 1e-10) +
                    (1 - e_ij) * np.log(1 - probs + 1e-10))

        # Compute gradients as derived
        grad_a = np.sum((e_ij - probs) * dot_products)
        grad_b = np.sum(e_ij - probs)

        return ll, grad_a, grad_b

    def train(self, graphs, learning_rate=0.006, max_iter=1000):
        history = []
        last_ll = float('-inf')

        for iter in range(max_iter):
            total_ll = 0
            total_grad_a = 0
            total_grad_b = 0

            # Accumulate gradients from all graphs
            for features, edges in graphs:
                ll, grad_a, grad_b = self.compute_log_likelihood_and_gradients(features, edges)
                total_ll += ll
                total_grad_a += grad_a
                total_grad_b += grad_b

            # Update parameters
            self.a += learning_rate * total_grad_a
            self.b += learning_rate * total_grad_b

            # Store history
            history.append({
                'iteration': iter + 1,
                'log_likelihood': total_ll,
                'a': self.a,
                'b': self.b
            })

            if iter >= max_iter - 10:
                print(f"Iter {iter + 1}: a = {self.a:.4f}, b = {self.b:.4f}, "
                      f"log-likelihood = {total_ll:.4f}")

            # Check convergence
            if abs(total_ll - last_ll) < 0.05:
                break

            last_ll = total_ll

        return history

    def generate_graph(self, n_students=15, n_features=3):


        features = np.random.uniform(-1, 1, (n_students, n_features))

        edges = np.zeros((n_students, n_students), dtype=np.uint8)

        # Generate edges for i < j pairs
        for i in range(n_students):
            for j in range(i + 1, n_students):
                # Compute p(eij = 1) = σ(ax(i)ᵀx(j) + b)
                dot_prod = np.dot(features[i], features[j])
                prob = sigmoid(self.a * dot_prod + self.b)

                # Sample edge independently
                edge = np.random.binomial(1, prob)

                # Set both eij and eji since graph is undirected
                edges[i, j] = edges[j, i] = edge

        return features, edges


def main():

    print("Loading data...")
    with open('classroom_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)


    print("\nTraining model...")
    model = SocialNetworkModel()
    history = model.train(graphs)

    print(f"\nFinal parameters:")
    print(f"a = {model.a:.6f}")
    print(f"b = {model.b:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot([h['iteration'] for h in history],
             [h['log_likelihood'] for h in history])
    plt.title('Log-Likelihood During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plt.show()

    print("\nGenerating 5 example graphs...")
    for i in range(5):
        features, edges = model.generate_graph()
        plt.figure(figsize=(8, 8))
        plt.title(f"Generated Graph {i + 1}")
        render_graph(features, edges)
        plt.show()


if __name__ == "__main__":
    main()