import numpy as np
from typing import List, Tuple, Optional, Dict

class GradientDescent:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, eta: float = 0.1, 
                 n_iterations: int = 1000, batch_size: int = 16):
        self.X = X_train
        self.y = y_train
        self.eta = eta
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.theta = np.zeros((1, X_train.shape[1]))
        self.cost_history: List[float] = []
        
    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray] = None) -> float:
        if theta is None:
            theta = self.theta
        m = len(y)
        error = X.dot(theta.T) - y
        return float(np.sum(error ** 2) / (2 * m))
    
    def gradient_batch(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        m = len(y)
        error = X.dot(theta.T) - y
        return (1/m) * error.T.dot(X)
    
    def gradiente(self, modo: str = 'batch') -> Tuple[np.ndarray, List[Dict[str, float]]]:
        self.cost_history = []
        plot_data = []
        
        if modo == 'batch':
            theta, costs = self._batch_gradient_descent()
        elif modo == 'stochastic':
            theta, costs = self._stochastic_gradient_descent()
        elif modo == 'mini-batch':
            theta, costs = self._mini_batch_gradient_descent()
        else:
            raise ValueError(f"Unknown mode: {modo}")
            
        # Convert numpy types to Python native types for JSON serialization
        plot_data = [
            {
                'iteration': int(i),
                'cost': float(cost)
            }
            for i, cost in enumerate(costs)
        ]
            
        return theta, plot_data
    
    def _batch_gradient_descent(self) -> Tuple[np.ndarray, List[float]]:
        costs = []
        for i in range(self.n_iterations):
            gradient = self.gradient_batch(self.X, self.y, self.theta)
            self.theta = self.theta - self.eta * gradient
            cost = self.compute_cost(self.X, self.y)
            costs.append(float(cost))  # Convert to native Python float
        return self.theta, costs
    
    def _stochastic_gradient_descent(self) -> Tuple[np.ndarray, List[float]]:
        m = len(self.y)
        costs = []
        batch_size = 100  # Reduced from full dataset (adjust as needed)
        cost_every_n = 1  # Compute cost every 10 batches
        
        for iteration in range(self.n_iterations):
            # Randomly sample a mini-batch (with replacement for speed)
            indices = np.random.choice(m, size=batch_size, replace=True)
            X_batch = self.X[indices]
            y_batch = self.y[indices]
            
            # Single vectorized gradient update for the entire batch
            gradient = self.gradient_batch(X_batch, y_batch, self.theta)
            self.theta -= self.eta * gradient
            
            # Only compute cost periodically
            if iteration % cost_every_n == 0:
                cost = self.compute_cost(X_batch, y_batch)  # Optional: Use full dataset
                costs.append(float(cost))
        
        return self.theta, costs
    
    def _mini_batch_gradient_descent(self) -> Tuple[np.ndarray, List[float]]:
        m = len(self.y)
        n_batches = max(m // self.batch_size, 1)
        costs = []
        
        for i in range(self.n_iterations):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = self.X[shuffled_indices]
            y_shuffled = self.y[shuffled_indices]
            
            for batch in range(n_batches):
                start = batch * self.batch_size
                end = min(start + self.batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                gradient = self.gradient_batch(X_batch, y_batch, self.theta)
                self.theta = self.theta - self.eta * gradient
            
            cost = self.compute_cost(self.X, self.y)
            costs.append(float(cost))  # Convert to native Python float
        return self.theta, costs