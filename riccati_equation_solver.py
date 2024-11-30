import numpy as np
import scipy.linalg as linalg
from typing import Tuple

class NonCooperativeRiccatiSolver:
    """
    Solver for Riccati equations in non-cooperative game systems
    Handles multi-player differential games with strategic interactions
    """
    
    def __init__(self, num_players: int):
        """
        Initialize the solver for a multi-player game
        
        Parameters:
        -----------
        num_players : int
            Number of players in the non-cooperative game
        """
        self.num_players = num_players
        
    def solve_two_player_differential_game(
        self, 
        A: np.ndarray, 
        B1: np.ndarray, 
        B2: np.ndarray, 
        Q1: np.ndarray, 
        Q2: np.ndarray, 
        R1: np.ndarray, 
        R2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Riccati equations for a two-player non-cooperative differential game
        
        Parameters:
        -----------
        A : numpy.ndarray
            System dynamics matrix
        B1, B2 : numpy.ndarray
            Input matrices for player 1 and player 2
        Q1, Q2 : numpy.ndarray
            State cost matrices for player 1 and player 2
        R1, R2 : numpy.ndarray
            Control cost matrices for player 1 and player 2
        
        Returns:
        --------
        Tuple of solution matrices for player 1 and player 2
        """
        # Solve coupled Riccati equations
        # These equations represent the strategic interactions between players
        
        # Player 1's Riccati equation
        X1 = linalg.solve_discrete_are(
            A.T,  # Transpose of system matrix
            B1,   # Player 1's input matrix
            Q1,   # Player 1's state cost
            R1    # Player 1's control cost
        )
        
        # Player 2's Riccati equation
        X2 = linalg.solve_discrete_are(
            A.T,  # Transpose of system matrix
            B2,   # Player 2's input matrix
            Q2,   # Player 2's state cost
            R2    # Player 2's control cost
        )
        
        return X1, X2
    
    def compute_feedback_strategies(
        self, 
        A: np.ndarray, 
        B1: np.ndarray, 
        B2: np.ndarray, 
        Q1: np.ndarray, 
        Q2: np.ndarray, 
        R1: np.ndarray, 
        R2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feedback strategies for non-cooperative game
        
        Parameters:
        -----------
        A, B1, B2, Q1, Q2, R1, R2 : numpy.ndarray
            System and cost matrices as described in solve_two_player_differential_game
        
        Returns:
        --------
        Tuple of feedback gain matrices for player 1 and player 2
        """
        # Solve Riccati equations
        X1, X2 = self.solve_two_player_differential_game(A, B1, B2, Q1, Q2, R1, R2)
        
        # Compute feedback gains
        # K1 = -(R1 + B1^T * X1 * B1)^-1 * B1^T * X1 * A
        K1 = -np.linalg.inv(R1 + B1.T @ X1 @ B1) @ B1.T @ X1 @ A
        
        # K2 = -(R2 + B2^T * X2 * B2)^-1 * B2^T * X2 * A
        K2 = -np.linalg.inv(R2 + B2.T @ X2 @ B2) @ B2.T @ X2 @ A
        
        return K1, K2
    
    def nash_equilibrium_analysis(
        self, 
        A: np.ndarray, 
        B1: np.ndarray, 
        B2: np.ndarray, 
        Q1: np.ndarray, 
        Q2: np.ndarray, 
        R1: np.ndarray, 
        R2: np.ndarray
    ) -> dict:
        """
        Perform Nash equilibrium analysis for the non-cooperative game
        
        Parameters:
        -----------
        A, B1, B2, Q1, Q2, R1, R2 : numpy.ndarray
            System and cost matrices
        
        Returns:
        --------
        dict
            Analysis results including solution matrices, feedback strategies, etc.
        """
        # Solve Riccati equations
        X1, X2 = self.solve_two_player_differential_game(A, B1, B2, Q1, Q2, R1, R2)
        
        # Compute feedback strategies
        K1, K2 = self.compute_feedback_strategies(A, B1, B2, Q1, Q2, R1, R2)
        
        # Compute performance costs
        # J1 = x^T * X1 * x (cost for player 1)
        # J2 = x^T * X2 * x (cost for player 2)
        
        return {
            'player1_solution_matrix': X1,
            'player2_solution_matrix': X2,
            'player1_feedback_gain': K1,
            'player2_feedback_gain': K2,
        }

def example_conflict_scenario():
    """
    Example of a non-cooperative differential game scenario
    
    Simulates a scenario with two players (e.g., two competing agents)
    """
    # System dynamics matrix
    A = np.array([
        [0.9, 0.1, 0.3, 0.3, 0.5],
        [0.2, 0.8, 0.1, 0.6, 0.7],
        [0.4, 0.2, 0.5, 0.4, 0.6],
        [0.1, 0.5, 0.3, 0.4, 0.4],
        [0.7, 0.8, 0.3, 0.6, 0.5]
    ])
    
    # Input matrices for players
    B1 = np.array([
        [1.0],
        [0.5],
        [0.4],
        [0.9],
        [0.1]
    ])
    B2 = np.array([
        [0.5],
        [1.0],
        [0.7],
        [0.5],
        [0.1]
    ])
    
    # Cost matrices for players
    # Q represents state costs (penalizing state deviation)
    Q1 = np.eye(5)  # Player 1's state cost
    Q2 = np.eye(5)  # Player 2's state cost
    
    # R represents control costs (penalizing control effort)
    R1 = np.array([[1.0]])  # Player 1's control cost
    R2 = np.array([[1.0]])  # Player 2's control cost
    
    # Create solver
    solver = NonCooperativeRiccatiSolver(num_players=2)
    
    # Perform Nash equilibrium analysis
    results = solver.nash_equilibrium_analysis(A, B1, B2, Q1, Q2, R1, R2)
    
    # Print results
    print("Non-Cooperative Game Nash Equilibrium Analysis:")
    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print()
    
    return results

# Run the example when script is executed directly
if __name__ == "__main__":
    example_conflict_scenario()