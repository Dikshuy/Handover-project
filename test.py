import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

class LaneChangeTrackingController:
    def __init__(self, params):
        # Vehicle parameters
        self.v = params.get('velocity', 10)  # Velocity (m/s)
        self.L_f = params.get('front_axle_distance', 1.2)  # Distance to front axle (m)
        self.L_r = params.get('rear_axle_distance', 1.6)  # Distance to rear axle (m)
        
        # Discretization parameters
        self.delta_t = params.get('time_step', 0.1)  # Time step (s)
        self.horizon = params.get('horizon', 100)  # Horizon length
        
        # Shared control parameters
        self.t_shared_start = params.get('shared_start', 3.0)
        self.t_shared_end = params.get('shared_end', 7.0)
        
        # Cost matrices
        self.Q_human_initial = np.diag([1, 1, 1])
        self.Q_av_initial = np.diag([1, 1, 1])
        self.R_human = np.array([[1]])
        self.R_av = np.array([[0.1]])
    
    def compute_linearized_matrices(self, state, input):
        """
        Compute linearized system matrices around the current state
        Dynamically update linearization point based on current state
        """
        theta = state[2]  # Yaw angle
        
        # Dynamic linearization point
        A_c = np.array([
            [0, 0, -self.v * np.sin(theta)],
            [0, 0, self.v * np.cos(theta)],
            [0, 0, 0]
        ])
        
        # Assume a simplified input-state relationship
        B_c = np.array([
            [-self.v * np.sin(theta)],
            [self.v * np.cos(theta)],
            [self.v / (self.L_f + self.L_r)]  # Simplified heading rate control
        ])
        
        # Discretize system
        A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, None, None), self.delta_t)
        
        return A_d, B_d
    
    def compute_control_gain(self, A_d, B_d, Q, R):
        """
        Compute optimal control gain using algebraic Riccati equation
        """
        # Simplified Riccati equation solving
        P = Q
        # for _ in range(10):  # Iterative refinement
        #     P_next = Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        #     if np.linalg.norm(P - P_next) < 1e-6:
        #         break
        #     P = P_next
        P = Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        # Compute gain
        K = np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        return K
    
    def generate_reference_trajectory(self, total_time, lane_width=3.5, forward_speed=5.0):
        """
        Generate a smooth lane change reference trajectory
        """
        dt = self.delta_t
        # lane_change_time = 5
        # straight_time = (total_time - lane_change_time) / 2

        # Straight path (initial)
        # t_straight1 = np.arange(0, straight_time, dt)
        # x_straight1 = forward_speed * t_straight1
        # y_straight1 = np.zeros_like(t_straight1)
        # theta_straight1 = np.zeros_like(t_straight1)

        # # Lane change
        # t_lane_change = np.arange(0, lane_change_time, dt)
        # x_lane_change = forward_speed * t_lane_change + x_straight1[-1]
        # y_lane_change = lane_width / 2 * (1 - np.cos(np.pi * t_lane_change / lane_change_time))
        # dx_dt = np.gradient(x_lane_change, dt)
        # dy_dt = np.gradient(y_lane_change, dt)
        # theta_lane_change = np.arctan2(dy_dt, dx_dt)

        # # Straight path (final)
        # t_straight2 = np.arange(0, straight_time, dt)
        # x_straight2 = forward_speed * t_straight2 + x_lane_change[-1]
        # y_straight2 = np.full_like(t_straight2, lane_width)
        # theta_straight2 = np.zeros_like(t_straight2)

        # # Combine all segments
        # t = np.concatenate([t_straight1, t_straight1[-1] + t_lane_change, t_straight1[-1] + t_lane_change[-1] + t_straight2])
        # x = np.concatenate([x_straight1, x_lane_change, x_straight2])
        # y = np.concatenate([y_straight1, y_lane_change, y_straight2])
        # theta = np.concatenate([theta_straight1, theta_lane_change, theta_straight2])

        t = np.arange(0, total_time, dt)
        x = forward_speed * t
        y = np.zeros_like(t)
        theta = np.zeros_like(t)

        trajectory = np.vstack([x, y, theta]).T
        return t, trajectory
    
    def simulate_trajectory_tracking(self, ref_trajectory):
        """
        Simulate trajectory tracking with adaptive shared control
        """
        # Initialize state
        x = np.array([0, 0, 0])  # Initial state: [x, y, yaw]
        trajectory = [x]
        inputs = []
        
        # Compute decay factors for shared control
        num_shared_steps = int((self.t_shared_end - self.t_shared_start) / self.delta_t)
        decay_factors = np.linspace(0, 1, num_shared_steps)
        
        for k in range(self.horizon):
            t = k * self.delta_t
            
            # Adaptive cost matrices based on control phase
            if t < self.t_shared_start:
                Q_human = np.zeros_like(self.Q_human_initial)
                Q_av = self.Q_av_initial
            elif self.t_shared_start <= t < self.t_shared_end:
                idx = k - int(self.t_shared_start / self.delta_t)
                decay_factor = decay_factors[min(idx, len(decay_factors)-1)]
                Q_human = decay_factor * self.Q_human_initial
                Q_av = (1 - decay_factor) * self.Q_av_initial
            else:
                Q_human = self.Q_human_initial
                Q_av = np.zeros_like(self.Q_av_initial)
            
            # Get reference state
            x_ref = ref_trajectory[k]
            
            # Compute linearized system matrices
            A_d, B_d = self.compute_linearized_matrices(x, inputs[-1] if inputs else np.array([0]))
            
            # Compute error
            error = x - x_ref
            
            # Compute control gains
            K_human = self.compute_control_gain(A_d, B_d, Q_human, self.R_human)
            K_av = self.compute_control_gain(A_d, B_d, Q_av, self.R_av)
            
            # Compute inputs
            u_human = -K_human @ error
            u_av = -K_av @ error
            
            # Combined input (shared control)
            u = u_human + u_av
            
            # # State update with full nonlinear dynamics
            # x_dot = np.array([
            #     self.v * np.cos(x[2]),
            #     self.v * np.sin(x[2]),
            #     u[0]  # Simplified heading rate control
            # ])
            
            # # Euler integration
            # x = x + x_dot * self.delta_t
            print(u, error)
            x = A_d @ x + B_d @ u
            
            trajectory.append(x)
            inputs.append(u)
        
        return np.array(trajectory), np.array(inputs)

# Main simulation
def main():
    # Parameters
    params = {
        'velocity': 10,
        'time_step': 0.1,
        'horizon': 100,
        'shared_start': 3.0,
        'shared_end': 7.0
    }
    
    # Create controller
    controller = LaneChangeTrackingController(params)
    
    # Generate reference trajectory
    total_time = 10.0
    t_ref, ref_trajectory = controller.generate_reference_trajectory(total_time)
    
    # Simulate tracking
    tracked_trajectory, inputs = controller.simulate_trajectory_tracking(ref_trajectory)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Trajectory plot
    plt.subplot(2, 1, 1)
    plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], label="Reference Trajectory", linestyle="--", color="blue")
    plt.plot(tracked_trajectory[:, 0], tracked_trajectory[:, 1], label="Tracked Trajectory", color="green")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Lane Change Trajectory Tracking")
    plt.legend()
    plt.grid(True)
    
    # Tracking error plot
    # plt.subplot(2, 1, 2)
    # error = tracked_trajectory - ref_trajectory[:tracked_trajectory.shape[0]]
    # plt.plot(t_ref[:tracked_trajectory.shape[0]], error[:, 0], label="X Error")
    # plt.plot(t_ref[:tracked_trajectory.shape[0]], error[:, 1], label="Y Error")
    # plt.plot(t_ref[:tracked_trajectory.shape[0]], error[:, 2], label="Yaw Error")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Error")
    # plt.title("Tracking Errors")
    # plt.legend()
    # plt.grid(True)
    
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
# import numpy as np
# from scipy.signal import cont2discrete
# import matplotlib.pyplot as plt

# # System parameters
# v = 10  # Velocity (m/s)
# L_f = 1.2  # Distance to front axle (m)
# L_r = 1.6  # Distance to rear axle (m)
# delta_t = 0.1  # Time step (s)
# N = 100  # Horizon length (10 seconds)

# # Nominal values
# theta_bar = 0
# beta_bar = 0
# delta_bar = 0

# # Partial derivative of beta
# d_beta_d_delta = L_r / ((L_f + L_r) * np.cos(delta_bar) ** 2)

# # Linearized system matrices
# A_c = np.array([
#     [0, 0, -v * np.sin(theta_bar + beta_bar)],
#     [0, 0, v * np.cos(theta_bar + beta_bar)],
#     [0, 0, 0]
# ])

# B_c = np.array([
#     [-v * np.sin(theta_bar + beta_bar) * d_beta_d_delta],
#     [v * np.cos(theta_bar + beta_bar) * d_beta_d_delta],
#     [v * np.cos(beta_bar) / L_r * d_beta_d_delta]
# ])

# # Discretize system
# A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, None, None), delta_t)

# # Define the lane changing trajectory function
# def lane_changing_trajectory(total_time, dt, lane_width=3.5, forward_speed=5.0):
#     lane_change_time = 5
#     straight_time = (total_time - lane_change_time) / 2  # Time for straight paths

#     # Straight path (initial)
#     t_straight1 = np.arange(0, straight_time, dt)
#     x_straight1 = forward_speed * t_straight1
#     y_straight1 = np.zeros_like(t_straight1)
#     theta_straight1 = np.zeros_like(t_straight1)  # Heading is constant

#     # Lane change
#     t_lane_change = np.arange(0, lane_change_time, dt)
#     x_lane_change = forward_speed * t_lane_change + x_straight1[-1]  # Continue from last x
#     y_lane_change = lane_width / 2 * (1 - np.cos(np.pi * t_lane_change / lane_change_time))  # Smooth lateral shift
#     dx_dt = np.gradient(x_lane_change, dt)
#     dy_dt = np.gradient(y_lane_change, dt)
#     theta_lane_change = np.arctan2(dy_dt, dx_dt)

#     # Straight path (final)
#     t_straight2 = np.arange(0, straight_time, dt)
#     x_straight2 = forward_speed * t_straight2 + x_lane_change[-1]  # Continue from last x
#     y_straight2 = np.full_like(t_straight2, lane_width)  # Stay in the new lane
#     theta_straight2 = np.zeros_like(t_straight2)  # Heading is constant

#     # Combine all segments
#     t = np.concatenate([t_straight1, t_straight1[-1] + t_lane_change, t_straight1[-1] + t_lane_change[-1] + t_straight2])
#     x = np.concatenate([x_straight1, x_lane_change, x_straight2])
#     y = np.concatenate([y_straight1, y_lane_change, y_straight2])
#     theta = np.concatenate([theta_straight1, theta_lane_change, theta_straight2])

#     trajectory = np.vstack([x, y, theta]).T
#     return t, trajectory

# # Generate reference trajectory
# total_time = 10.0  # Total time for the trajectory (10 seconds)
# t_ref, ref_trajectory = lane_changing_trajectory(total_time, delta_t)

# # Initialize state
# x = np.array([0, 0, 0])  # Initial state: [x, y, yaw]
# trajectory = [x]
# u1_inputs, u2_inputs = [], []

# # Define cost matrices
# Q_human = np.diag([1, 1, 1])  # Initial cost for human
# Q_av = np.diag([1, 1, 1])  # Initial cost for AV
# R_human = np.array([[1]])
# R_av = np.array([[1]])

# # Compute linear decay factors
# t_shared_start = 3.0
# t_shared_end = 7.0
# num_shared_steps = int((t_shared_end - t_shared_start) / delta_t)
# decay_factors = np.linspace(0, 1, num_shared_steps)

# # Solve the game over the time horizon
# for k in range(N):
#     t = k * delta_t

#     # Phase 1: Full AV control (0–3 seconds)
#     if t < t_shared_start:
#         Q_human = np.zeros_like(Q_human)
#         Q_av = Q_av

#     # Phase 2: Shared control (3–7 seconds)
#     elif t_shared_start <= t < t_shared_end:
#         idx = k - int(t_shared_start / delta_t)
#         decay_factor = decay_factors[idx]
#         Q_human = decay_factor * Q_human
#         Q_av = (1 - decay_factor) * Q_av

#     # Phase 3: Full human control (7+ seconds)
#     else:
#         Q_human = Q_human
#         Q_av = np.zeros_like(Q_av)

#     # Get reference state at time t
#     x_ref = ref_trajectory[k]

#     # Compute state error
#     error = x - x_ref  # [x, y, yaw] error

#     # Riccati equations for both players
#     P1 = Q_human
#     P2 = Q_av

#     S1 = R_human + B_d.T @ P1 @ B_d
#     S2 = R_av + B_d.T @ P2 @ B_d

#     K1 = np.linalg.inv(S1) @ (B_d.T @ P1 @ A_d)
#     K2 = np.linalg.inv(S2) @ (B_d.T @ P2 @ A_d)

#     # Optimal inputs for trajectory tracking
#     u1 = -K1 @ error
#     u2 = -K2 @ error

#     # Combine inputs for shared control
#     u = u1 + u2

#     # State update
#     x = A_d @ x + B_d @ u

#     # Record trajectory and inputs
#     trajectory.append(x)
#     u1_inputs.append(u1)
#     u2_inputs.append(u2)

# # Convert to arrays for plotting
# trajectory = np.array(trajectory)
# u1_inputs = np.array(u1_inputs).flatten()
# u2_inputs = np.array(u2_inputs).flatten()

# # Plot results
# time = np.arange(0, (N + 1) * delta_t, delta_t)

# plt.figure(figsize=(12, 8))

# plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], label="Reference Trajectory", linestyle="--", color="blue")
# plt.plot(trajectory[:, 0], trajectory[:, 1], label="Tracked Trajectory", color="green")
# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")
# plt.legend()
# plt.title("Lane Change Trajectory Tracking")
# plt.grid()
# # plt.show()

# # plt.figure(figsize=(12, 4))
# # plt.plot(time, u1_inputs, label="Autonomous Vehicle Control Input", color="red")
# # plt.plot(time, u2_inputs, label="Human Driver Control Input", color="purple")
# # plt.xlabel("Time (s)")
# # plt.ylabel("Control Input")
# # plt.legend()
# # plt.title("Control Inputs Over Time")
# # plt.grid()
# # plt.show()

# # # Plot state evolution
# # plt.subplot(3, 1, 1)
# # plt.plot(time, trajectory[:, 0], label="x-position")
# # plt.plot(time, trajectory[:, 1], label="y-position")
# # plt.plot(time, trajectory[:, 2], label="yaw (rad)")
# # plt.xlabel("Time (s)")
# # plt.ylabel("States")
# # plt.title("State Evolution")
# # plt.legend()
# # plt.grid()

# # # Plot control inputs
# # plt.subplot(3, 1, 2)
# # plt.step(time[:-1], u1_inputs, label="Human Input (u1)", where="post")
# # plt.step(time[:-1], u2_inputs, label="AV Input (u2)", where="post")
# # plt.xlabel("Time (s)")
# # plt.ylabel("Control Inputs")
# # plt.title("Control Inputs Over Time")
# # plt.legend()
# # plt.grid()

# # Plot trajectory vs reference trajectory
# # plt.subplot(3, 1, 3)
# # plt.plot(trajectory[:, 0], trajectory[:, 1], label="Vehicle Trajectory")
# # plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], label="Reference Trajectory", linestyle="--")
# # plt.xlabel("X Position")
# # plt.ylabel("Y Position")
# # plt.title("Trajectory Tracking")
# # plt.legend()
# # plt.grid()

# plt.tight_layout()
# plt.show()
