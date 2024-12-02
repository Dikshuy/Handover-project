import numpy as np
import matplotlib.pyplot as plt

l_f = 1.2
l_r = 1.4
l = l_f + l_r
v = 1.4

Q_av = np.diag([1, 10, 1])
R_av = np.diag([0.1])

Q_human = np.diag([10, 10, 10])
R_human = np.diag([0.1])

dt = 0.1
time = 15
N = 50

t1 = 5  # av control
t2 = 14 # handover time: 10 seconds
t3 = (time-t2)  # human control

def tire_slip_angle(delta_f):
    return np.arctan((l_r * np.tan(delta_f)) / (l_f + l_r))

def get_matrices(v, theta, delta_f):
    beta = tire_slip_angle(delta_f)
    d_beta_d_delta = l_r / ((l_f + l_r) * np.cos(delta_f) ** 2)

    A = np.array([

        [0, 0, -v * np.sin(theta+beta)],
        [0, 0, v * np.cos(theta+beta)],
        [0, 0, 0]
    ])

    B = np.array([
        [-v * np.sin(theta + beta) * d_beta_d_delta],
        [v * np.cos(theta + beta) * d_beta_d_delta],
        [v * np.cos(beta) / l_r * d_beta_d_delta]
    ])
    
    return A, B

def update(state, input):
    x, y, theta = state
    delta = input

    beta = np.arctan(l_r / l * np.tan(delta))

    x = x + v * np.cos(theta + beta) * dt
    y = y + v * np.sin(theta + beta) * dt
    theta = theta + v / l * np.sin(beta) * dt
    state = np.array([x, y, theta])

    return state

def get_trajectory(v, total_time, dt, lane_width=1):
    lane_change_time = 6
    straight_time = (total_time - lane_change_time) / 2

    # Straight path
    t_straight1 = np.arange(0, straight_time, dt)
    x_straight1 = v * t_straight1
    y_straight1 = np.zeros_like(t_straight1)
    theta_straight1 = np.zeros_like(t_straight1)

    # Lane change
    t_lane_change = np.arange(0, lane_change_time, dt)
    x_lane_change = v * t_lane_change + x_straight1[-1]
    y_lane_change = lane_width / 2 * (1 - np.cos(np.pi * t_lane_change / lane_change_time))
    dx_dt = np.gradient(x_lane_change, dt)
    dy_dt = np.gradient(y_lane_change, dt)
    theta_lane_change = np.arctan2(dy_dt, dx_dt)

    # Straight path
    t_straight2 = np.arange(0, straight_time, dt)
    x_straight2 = v * t_straight2 + x_lane_change[-1]
    y_straight2 = np.full_like(t_straight2, lane_width)
    theta_straight2 = np.zeros_like(t_straight2)

    # Combine all segments
    t = np.concatenate([t_straight1, t_straight1[-1] + t_lane_change, t_straight1[-1] + t_lane_change[-1] + t_straight2])
    x = np.concatenate([x_straight1, x_lane_change, x_straight2])
    y = np.concatenate([y_straight1, y_lane_change, y_straight2])
    theta = np.concatenate([theta_straight1, theta_lane_change, theta_straight2])

    trajectory = np.vstack([x, y, theta]).T
    return t, trajectory

def control_input(A, B, Q, R, error):
    P = [None] * (N+1)
    K = [None] * N
    P[N] = Q

    for i in range(N, 0, -1):
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

    for i in range(N):
        K[i] = np.linalg.inv(R + B.T @ P[i+1] @ B) @ (B.T @ P[i+1] @ A)
    
    u[0] = -K[0] @ error

    return u

def av_control(u, simulation_time, desired_trajectory):
    tracking_error_history = []
    state_history = []
    control_history = []

    x = np.array([[0], [0], [0]])

    for t in range(simulation_time):
        state_error = x - desired_trajectory[t].reshape(-1, 1)
        A, B = get_matrices(v, x[2, 0], u[0])

        u_av = control_input(A, B, Q_av, R_av, state_error)
        u = u_av

        # print("control input for AV control:", u[0])

        x = update(x, u[0])
        
        tracking_error_history.append(state_error.flatten())
        state_history.append(x.flatten())
        control_history.append(u[0])

    # Convert to numpy arrays for analysis
    state_history_av = np.array(state_history)
    control_history_av = np.array(control_history)
    tracking_error_history_av = np.array(tracking_error_history)

    return state_history_av, control_history_av, tracking_error_history_av

def shared_control(u, simulation_time, desired_trajectory):
    tracking_error_history = []
    state_history = []
    control_history = []

    x = np.array([[0], [0], [0]])

    for t in range(simulation_time):
        run_time = t * dt
        state_error = x - desired_trajectory[t].reshape(-1, 1)
        A, B = get_matrices(v, x[2, 0], u[0])

        if run_time <= t1:
            u_av = control_input(A, B, Q_av, R_av, state_error)
            u = u_av

        elif t1 < run_time < t2:
            alpha = (run_time - t1) / (t2 - t1)
            Q_av_ = (1 - alpha) * Q_av
            Q_human_ = (alpha) * Q_human
            u_av = control_input(A, B, Q_av_, R_av, state_error)
            u_human = control_input(A, B, Q_human_, R_human, state_error)
            u = u_av + u_human
            # u = (1-alpha) * u_av + alpha * u_human

        elif run_time >= t2:
            u_human = control_input(A, B, Q_human, R_human, state_error)
            u = u_human

        else:
            print("panic!")

        # print("control input for shared control:", u[0])

        x = update(x, u[0])
        
        tracking_error_history.append(state_error.flatten())
        state_history.append(x.flatten())
        control_history.append(u[0])

    # Convert to numpy arrays for analysis
    state_history_shared = np.array(state_history)
    control_history_shared = np.array(control_history)
    tracking_error_history_shared = np.array(tracking_error_history)

    return state_history_shared, control_history_shared, tracking_error_history_shared

def human_control(u, simulation_time, desired_trajectory):
    tracking_error_history = []
    state_history = []
    control_history = []

    x = np.array([[0], [0], [0]])

    for t in range(simulation_time):
        state_error = x - desired_trajectory[t].reshape(-1, 1)
        A, B = get_matrices(v, x[2, 0], u[0])

        u_human = control_input(A, B, Q_human, R_human, state_error)
        u = u_human

        # print("control input for human control:", u[0])

        x = update(x, u[0])
        
        tracking_error_history.append(state_error.flatten())
        state_history.append(x.flatten())
        control_history.append(u[0])

    # Convert to numpy arrays for analysis
    state_history_human = np.array(state_history)
    control_history_human = np.array(control_history)
    tracking_error_history_human = np.array(tracking_error_history)

    return state_history_human, control_history_human, tracking_error_history_human


T, desired_trajectory = get_trajectory(v, time, dt)
simulation_time = len(T)

u = np.zeros(N)

state_history_av, control_history_av, tracking_error_history_av = av_control(u, simulation_time, desired_trajectory)
state_history_shared, control_history_shared, tracking_error_history_shared = shared_control(u, simulation_time, desired_trajectory)
state_history_human, control_history_human, tracking_error_history_human = human_control(u, simulation_time, desired_trajectory)

def calculate_tracking_errors(state_history, desired_trajectory):
    x_error = state_history[:, 0] - desired_trajectory[:, 0]
    y_error = state_history[:, 1] - desired_trajectory[:, 1]
    theta_error = state_history[:, 2] - desired_trajectory[:, 2]

    euclidean_error = np.sqrt(x_error**2 + y_error**2)
    
    return x_error, y_error, theta_error, euclidean_error

x_error_av, y_error_av, theta_error_av, euclidean_error_av = calculate_tracking_errors(state_history_av, desired_trajectory)
x_error_shared, y_error_shared, theta_error_shared, euclidean_error_shared = calculate_tracking_errors(state_history_shared, desired_trajectory)
x_error_human, y_error_human, theta_error_human, euclidean_error_human = calculate_tracking_errors(state_history_human, desired_trajectory)

print("----------------------------------------------------")

def print_error_summary(error_name, errors):
    print(f"{error_name} Error Summary:")
    print(f"  Mean: {np.mean(errors):.4f}")
    print(f"  Max:  {np.max(np.abs(errors)):.4f}")
    print(f"  RMS:  {np.sqrt(np.mean(errors**2)):.4f}\n")

print("Tracking Performance Analysis:")
print("AV Control:")
print_error_summary("X-Position", x_error_av)
print_error_summary("Y-Position", y_error_av)
print_error_summary("Heading", theta_error_av)
print_error_summary("Euclidean", euclidean_error_av)

print("Shared Control:")
print_error_summary("X-Position", x_error_shared)
print_error_summary("Y-Position", y_error_shared)
print_error_summary("Heading", theta_error_shared)
print_error_summary("Euclidean", euclidean_error_shared)

print("Human Control:")
print_error_summary("X-Position", x_error_human)
print_error_summary("Y-Position", y_error_human)
print_error_summary("Heading", theta_error_human)
print_error_summary("Euclidean", euclidean_error_human)

print("----------------------------------------------------")

plt.figure(figsize=(12,10), dpi=100)
# Position tracking
plt.subplot(2, 1, 1)
# Desired trajectory
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], 
         label='Desired Trajectory', 
         color='black', 
         linewidth=3, 
         linestyle='--', 
         alpha=0.7)

# AV control trajectory
plt.plot(state_history_av[:, 0], state_history_av[:, 1], 
         label='AV Control', 
         color='red', 
         marker='o', 
         markersize=4, 
         markeredgecolor='red',
         markerfacecolor='red', 
         linestyle='-', 
         linewidth=2)

# Human control trajectory
plt.plot(state_history_human[:, 0], state_history_human[:, 1], 
         label='Human Control', 
         color='purple', 
         marker='s', 
         markersize=4, 
         markeredgecolor='violet',
         markerfacecolor='purple', 
         linestyle='-', 
         linewidth=2)

# Shared control trajectory
plt.plot(state_history_shared[:, 0], state_history_shared[:, 1], 
         label='Shared Control', 
         color='green', 
         marker='^', 
         markersize=4, 
         markeredgecolor='green',
         markerfacecolor='green', 
         linestyle='-', 
         linewidth=2)

plt.title('Position Tracking Comparison', fontsize=16)
plt.ylim(-0.5, 1.5)
plt.xlabel('X Position [m]', fontsize=12)
plt.ylabel('Y Position [m]', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, color='gray', alpha=0.7)
plt.minorticks_on()

# Control input
plt.subplot(2, 1, 2)
# AV control input
plt.plot(control_history_av, 
         label='AV Control', 
         color='red', 
         marker='o', 
         markersize=4, 
         markeredgecolor='red',
         markerfacecolor='red', 
         linestyle='-', 
         linewidth=2)

# Human control input
plt.plot(control_history_human, 
         label='Human Control', 
         color='purple', 
         marker='s', 
         markersize=4, 
         markeredgecolor='violet',
         markerfacecolor='purple', 
         linestyle='-', 
         linewidth=2)

# Shared control input
plt.plot(control_history_shared, 
         label='Shared Control', 
         color='green', 
         marker='^', 
         markersize=4, 
         markeredgecolor='green',
         markerfacecolor='green', 
         linestyle='-', 
         linewidth=2)

plt.title('Steering Angle Comparison', fontsize=16)
plt.ylim(-0.3, 0.3)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Steering Angle [rad]', fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, color='gray', alpha=0.7)
plt.minorticks_on()
plt.legend(loc='best', fontsize=10)

plt.savefig("controls comparison.png", dpi=300)
plt.tight_layout()
plt.show()



'''
MORE VISUALIZATIONS

# TO DO: in shared control, plot the contribution of AV and human

# Create a more comprehensive visualization
plt.figure(figsize=(20, 20), dpi=100)

# 1. Position Tracking Plot
plt.subplot(3, 2, 1)
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], 
         label='Desired Trajectory', 
         color='black', 
         linewidth=3, 
         linestyle='--', 
         alpha=0.7)
plt.plot(state_history_av[:, 0], state_history_av[:, 1], 
         label='AV Control', 
         color='red', 
         marker='o', 
         markersize=4, 
         markeredgecolor='red',
         markerfacecolor='red', 
         linestyle='-', 
         linewidth=2)
plt.plot(state_history_shared[:, 0], state_history_shared[:, 1], 
         label='Shared Control', 
         color='green', 
         marker='^', 
         markersize=4, 
         markeredgecolor='green',
         markerfacecolor='green', 
         linestyle='-', 
         linewidth=2)
plt.plot(state_history_human[:, 0], state_history_human[:, 1], 
         label='Human Control', 
         color='purple', 
         marker='s', 
         markersize=4, 
         markeredgecolor='violet',
         markerfacecolor='purple', 
         linestyle='-', 
         linewidth=2)
plt.title('Position Tracking Comparison', fontsize=12, fontweight='bold')
plt.xlabel('X Position [m]')
plt.ylabel('Y Position [m]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# 2. Control Input
plt.subplot(3, 2, 2)
plt.plot(control_history_av, 
         label='AV Control', 
         color='red', 
         marker='o', 
         markersize=4, 
         markeredgecolor='red',
         markerfacecolor='red', 
         linestyle='-', 
         linewidth=2)
plt.plot(control_history_shared, 
         label='Shared Control', 
         color='green', 
         marker='^', 
         markersize=4, 
         markeredgecolor='green',
         markerfacecolor='green', 
         linestyle='-', 
         linewidth=2)
plt.plot(control_history_human, 
         label='Human Control', 
         color='purple', 
         marker='s', 
         markersize=4, 
         markeredgecolor='violet',
         markerfacecolor='purple', 
         linestyle='-', 
         linewidth=2)
plt.title('Steering Angle Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Steering Angle [rad]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# 3. X-Position Error
plt.subplot(3, 2, 3)
plt.plot(x_error_av, label='AV Control', color='red', linewidth=2)
plt.plot(x_error_shared, label='Shared Control', color='green', linewidth=2)
plt.plot(x_error_human, label='Human Control', color='purple', linewidth=2)
plt.title('X-Position Tracking Error', fontsize=12, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('X Error [m]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# 4. Y-Position Error
plt.subplot(3, 2, 4)
plt.plot(y_error_av, label='AV Control', color='red', linewidth=2)
plt.plot(y_error_shared, label='Shared Control', color='green', linewidth=2)
plt.plot(y_error_human, label='Human Control', color='purple', linewidth=2)
plt.title('Y-Position Tracking Error', fontsize=12, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Y Error [m]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# 5. Heading (Theta) Error
plt.subplot(3, 2, 5)
plt.plot(theta_error_av, label='AV Control', color='red', linewidth=2)
plt.plot(theta_error_shared, label='Shared Control', color='green', linewidth=2)
plt.plot(theta_error_human, label='Human Control', color='purple', linewidth=2)
plt.title('Heading (Orientation) Error', fontsize=12, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Theta Error [rad]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

# 6. Euclidean Tracking Error
plt.subplot(3, 2, 6)
plt.plot(euclidean_error_av, label='AV Control', color='red', linewidth=2)
plt.plot(euclidean_error_shared, label='Shared Control', color='green', linewidth=2)
plt.plot(euclidean_error_human, label='Human Control', color='purple', linewidth=2)
plt.title('Euclidean Tracking Error', fontsize=12, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Euclidean Error [m]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
'''