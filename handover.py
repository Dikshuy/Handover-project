import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# Vehicle parameters
l_f = 1.2  # Distance from center of mass to front axle
l_r = 1.6  # Distance from center of mass to rear axle
v = 5.0    # Constant velocity (m/s)

# Improved trajectory generation with heading (theta)
def desired_trajectory(t):
    """
    Generate a more complex trajectory with explicit theta calculation
    Includes: 
    1. Straight line segment
    2. Smooth lane change with curved path
    3. Final straight line segment
    """
    if t < 7:  # Straight trajectory
        return np.array([v * t, 0, 0])  # [x_d, y_d, theta_d]
    elif 7 <= t < 12:  # Curved lane change
        # Use a sinusoidal path for smooth lane change
        lane_change_progress = (t - 7) / 5  # Normalized progress
        x = v * t
        y = 4 * np.sin(np.pi * lane_change_progress / 2)  # Smooth S-curve
        
        # Calculate heading (theta) based on path curvature
        theta = np.arctan2(4 * np.pi/2 * np.cos(np.pi * lane_change_progress / 2), v)
        
        return np.array([x, y, theta])
    else:  # Straight after lane change
        final_y = 4  # Final lane position
        final_theta = 0  # Return to straight heading
        x = v * t
        return np.array([x, final_y, final_theta])

# Improved state-space matrices with more accurate dynamics
def get_matrices(v, theta, beta, delta_f):
    """
    More comprehensive state-space matrix calculation
    Includes explicit dependency on front wheel angle
    """
    # Bicycle model state-space matrix
    A = np.array([
        [0, 0, -v * np.sin(theta)],
        [0, 0, v * np.cos(theta)],
        [0, 0, 0]
    ])
    
    # Improved tire slip angle calculation
    def tire_slip_angle(delta_f):
        return np.arctan((l_r * np.tan(delta_f)) / (l_f + l_r))
    
    # Input matrix with more accurate front wheel angle influence
    B = np.array([
        [-v * np.sin(theta + tire_slip_angle(delta_f))],
        [v * np.cos(theta + tire_slip_angle(delta_f))],
        [v / (l_f + l_r) * np.tan(delta_f)]
    ])
    
    return A, B

# Cost matrices with adaptive weights
def adaptive_cost_matrix(t):
    """
    Dynamically adjust cost matrices during shared control
    """
    if t < t1:
        # AV dominance
        Q_av = np.diag([1, 1, 2])  # Higher emphasis on position tracking
        R_av = np.array([[1]])  # Smooth control input
        return Q_av, R_av
    elif t1 <= t <= t2:
        # Shared control transition
        alpha = (t - t1) / (t2 - t1)
        if alpha == 0:
            alpha = 0.001
        if alpha == 1:
            alpha = 0.999
        Q_av = np.diag([1 * (1-alpha), 1 * (1-alpha), 2 * (1-alpha)])
        Q_human = np.diag([1 * alpha, 1 * alpha, 2 * alpha])
        R_av = np.array([[1 * (1-alpha)]])
        R_human = np.array([[1 * alpha]])
        return Q_av, Q_human, R_av, R_human
    else:
        # Human dominance
        Q_human = np.diag([1, 1, 2])
        R_human = np.array([[1]])
        return Q_human, R_human

# Simulation parameters
t1 = 5   # Shared control begins
t2 = 10  # Full human control begins
t3 = 15  # Simulation end
dt = 0.1  # Time step
time_horizon = int(t3 / dt)

# Initial state
x = np.array([[0], [0], [0]])  # [x, y, theta]
state_history = [x]
control_history = []
tracking_error_history = []

# Control storage
u = np.array([[0]])

# Advanced simulation loop
for t in range(time_horizon):
    current_time = t * dt
    
    # Get desired state
    x_d, y_d, theta_d = desired_trajectory(current_time)
    desired_state = np.array([[x_d], [y_d], [theta_d]])

    # Compute tracking error
    tracking_error = x - desired_state
    tracking_error_history.append(tracking_error)

    # Get updated matrices with current state and control
    A, B = get_matrices(v, x[2, 0], 0, u[0, 0])

    # Adaptive control strategy
    if current_time < t1:
        # AV Full Control
        Q_av, R_av = adaptive_cost_matrix(current_time)
        P_av = solve_discrete_are(A, B, Q_av, R_av)
        K_av = np.linalg.inv(R_av + B.T @ P_av @ B) @ (B.T @ P_av @ A)
        u = -K_av @ tracking_error
    
    elif t1 <= current_time <= t2:
        # Shared Control
        Q_av, Q_human, R_av, R_human = adaptive_cost_matrix(current_time)
        
        P_av = solve_discrete_are(A, B, Q_av, R_av)
        P_human = solve_discrete_are(A, B, Q_human, R_human)

        K_av = np.linalg.inv(R_av + B.T @ P_av @ B) @ (B.T @ P_av @ A)
        K_human = np.linalg.inv(R_human + B.T @ P_human @ B) @ (B.T @ P_human @ A)

        u_av = -K_av @ tracking_error
        u_human = -K_human @ tracking_error
        u = u_av + u_human
    
    elif t2 < current_time <= t3:
        # Human-Dominant Control
        Q_human, R_human = adaptive_cost_matrix(current_time)
        P_human = solve_discrete_are(A, B, Q_human, R_human)
        K_human = np.linalg.inv(R_human + B.T @ P_human @ B) @ (B.T @ P_human @ A)
        u = -K_human @ tracking_error
    
    else:
        # Full manual control
        u = np.array([[0]])
    
    # Update state (simple discrete-time update)
    x = A @ x + B @ u
    
    # Store history
    state_history.append(x)
    control_history.append(u)

# Convert to numpy arrays for analysis
state_history = np.array(state_history)
control_history = np.array(control_history)
tracking_error_history = np.array(tracking_error_history)

# Comprehensive Visualization
time = np.arange(0, t3 + dt, dt)

plt.figure(figsize=(15, 10))

# Position Tracking
plt.subplot(3, 2, 1)
plt.plot(time, state_history[:, 0], label="Actual X")
plt.plot(time, [desired_trajectory(ti)[0] for ti in time], '--', label="Desired X")
plt.title("X Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("X Position")
plt.legend()

# plt.subplot(3, 2, 2)
# plt.plot(time, state_history[:, 1], label="Actual Y")
# plt.plot(time, [desired_trajectory(ti)[1] for ti in time], '--', label="Desired Y")
# plt.title("Y Position Tracking")
# plt.xlabel("Time (s)")
# plt.ylabel("Y Position")
# plt.legend()

plt.subplot(3, 2, 2)
plt.plot(state_history[:, 0], state_history[:, 1], label="Actual Y")
plt.plot([desired_trajectory(ti)[0] for ti in time], [desired_trajectory(ti)[1] for ti in time], '--', label="Desired Y")
plt.title("Y Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Y Position")
plt.legend()

# Heading Tracking
plt.subplot(3, 2, 3)
plt.plot(time, state_history[:, 2], label="Actual θ")
plt.plot(time, [desired_trajectory(ti)[2] for ti in time], '--', label="Desired θ")
plt.title("Heading (θ) Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Heading (radians)")
plt.legend()

# Control Input
plt.subplot(3, 2, 4)
plt.plot(time[:-1], control_history[:, 0, 0], label="Steering Input (δ)")
plt.title("Steering Angle")
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (radians)")
plt.legend()

# Tracking Errors
plt.subplot(3, 2, 5)
plt.plot(time[:-1], [np.linalg.norm(e[:2]) for e in tracking_error_history], label="Position Error")
plt.title("Position Tracking Error")
plt.xlabel("Time (s)")
plt.ylabel("Error Magnitude")
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(time[:-1], np.abs(tracking_error_history[:, 2]), label="Heading Error")
plt.title("Heading Tracking Error")
plt.xlabel("Time (s)")
plt.ylabel("Heading Error (radians)")
plt.legend()

plt.tight_layout()
plt.show()