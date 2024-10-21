import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AVHandoverSystem:
    def __init__(self, T=10, L=2.5, dt=0.1):
        self.T = T  # Handover duration
        self.L = L  # Wheelbase
        self.dt = dt  # Time step
        self.t = 0  # Current time
        
        # State: [x, y, theta, v]
        self.state = np.array([0., 0., 0., 0.])
        
        # MPC parameters
        self.N = 20  # Prediction horizon
        self.Q = np.diag([1., 1., 1., 1.])  # State cost
        self.R = np.diag([1., 1.])  # Control cost
        self.S = np.diag([1., 1.])  # Human-MPC difference cost
        
        # Constraints
        self.delta_max = np.radians(30)
        self.a_max = 2.0
        self.v_max = 20.0

    def alpha(self):
        return min(1.0, self.t / self.T)

    def alpha(self, alpha_rate = 0.5):
        return 1 / (1 + np.exp(-alpha_rate * (self.t - self.T/2)))

    def kinematic_bicycle_model(self, state, u):
        x, y, theta, v = state
        delta, a = u
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = (v / self.L) * np.tan(delta)
        dv = a
        
        return np.array([dx, dy, dtheta, dv])

    def simulate_human_input(self):
        # Simplified human input simulation
        delta_h = 0.01 * np.sin(2 * np.pi * self.t / 5)  # Oscillating steering
        a_h = 0.1 * np.cos(2 * np.pi * self.t / 10)  # Oscillating acceleration
        return np.array([delta_h, a_h])

    def mpc_cost(self, u_seq, x0, x_ref, u_human):
        x = x0.copy()
        cost = 0
        
        for k in range(self.N):
            u_mpc = u_seq[2*k:2*k+2]
            alpha_k = min(1.0, (self.t + k*self.dt) / self.T)
            u = (1 - alpha_k) * u_mpc + alpha_k * u_human
            
            cost += np.dot(x - x_ref, np.dot(self.Q, x - x_ref))
            cost += np.dot(u_mpc, np.dot(self.R, u_mpc))
            cost += np.dot(u_mpc - u_human, np.dot(self.S, u_mpc - u_human))
            
            x += self.kinematic_bicycle_model(x, u) * self.dt
        
        return cost

    def mpc_constraints(self, u_seq):
        return np.concatenate([
            self.delta_max - np.abs(u_seq[::2]),  # Steering constraints
            self.a_max - np.abs(u_seq[1::2])  # Acceleration constraints
        ])

    def step(self, x_ref):
        u_human = self.simulate_human_input()
        
        # Solve MPC problem
        u0 = np.zeros(2 * self.N)
        bounds = [(-self.delta_max, self.delta_max), (-self.a_max, self.a_max)] * self.N
        constraints = {'type': 'ineq', 'fun': self.mpc_constraints}
        
        result = minimize(self.mpc_cost, u0, args=(self.state, x_ref, u_human),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        u_mpc = result.x[:2]
        
        # Combine MPC and human control
        alpha = self.alpha()
        u = (1 - alpha) * u_mpc + alpha * u_human
        
        # Update state
        self.state += self.kinematic_bicycle_model(self.state, u) * self.dt
        self.t += self.dt
        
        return self.state, u, u_mpc, u_human

def run_simulation():
    system = AVHandoverSystem()
    
    # Simple circular reference trajectory
    t_sim = np.arange(0, 20, system.dt)
    x_ref = np.column_stack([
        10 * np.cos(0.1 * t_sim),
        10 * np.sin(0.1 * t_sim),
        0.1 * np.ones_like(t_sim),
        1 * np.ones_like(t_sim)
    ])
    
    states = []
    controls = []
    
    for x_ref_k in x_ref:
        state, u, u_mpc, u_human = system.step(x_ref_k)
        states.append(state)
        controls.append((u, u_mpc, u_human))
    
    states = np.array(states)
    controls = np.array(controls)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(states[:, 0], states[:, 1], label='Vehicle')
    plt.plot(x_ref[:, 0], x_ref[:, 1], '--', label='Reference')
    plt.title('Vehicle Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(t_sim, controls[:, 0, 0], label='Combined')
    plt.plot(t_sim, controls[:, 1, 0], '--', label='MPC')
    plt.plot(t_sim, controls[:, 2, 0], ':', label='Human')
    plt.title('Steering Control')
    plt.xlabel('Time')
    plt.ylabel('Steering angle (rad)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(t_sim, controls[:, 0, 1], label='Combined')
    plt.plot(t_sim, controls[:, 1, 1], '--', label='MPC')
    plt.plot(t_sim, controls[:, 2, 1], ':', label='Human')
    plt.title('Acceleration Control')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(t_sim, states[:, 3])
    plt.title('Vehicle Speed')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()