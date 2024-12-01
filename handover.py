import numpy as np
import matplotlib.pyplot as plt

l_f = 1.2
l_r = 1.4
l = l_f + l_r
v = 1.4

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
    lane_change_time = 5
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

Q_av = np.diag([100, 100, 0])
R_av = np.diag([0.1])

x = np.array([[0], [0], [0]])
state_history = [x.flatten()]
control_history = []
tracking_error_history = []
dt = 0.1

N = 50
u = np.zeros(N)

time = 20

t, desired_trajectory = get_trajectory(v, time, dt)

simulation_time = int(time / dt)

for t in range(simulation_time):
    state_error = x - desired_trajectory[t].reshape(-1, 1)
    A, B = get_matrices(v, x[2, 0], u[0])

    P_av = [None] * (N+1)
    K_av = [None] * N
    P_av[N] = Q_av

    for i in range(N, 0, -1):
        P_av[i-1] = Q_av + A.T @ P_av[i] @ A - (A.T @ P_av[i] @ B) @ np.linalg.pinv(R_av + B.T @ P_av[i] @ B) @ (B.T @ P_av[i] @ A)

    for i in range(N):
        K_av[i] = np.linalg.inv(R_av + B.T @ P_av[i+1] @ B) @ (B.T @ P_av[i+1] @ A)
        u[i] = -K_av[i] @ state_error

    print("control input:", u[0])

    # Update state (simple discrete-time update)
    x = update(x, u[0])
    
    tracking_error_history.append(state_error.flatten())
    state_history.append(x.flatten())
    control_history.append(u[0])

# Convert to numpy arrays for analysis
state_history = np.array(state_history)
control_history = np.array(control_history)
tracking_error_history = np.array(tracking_error_history)

plt.figure(figsize=(15, 5))

# Position tracking
plt.subplot(1, 3, 1)
plt.plot(state_history[:, 0], state_history[:, 1], label='Actual Trajectory')
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], '--', label='Desired Trajectory')
plt.title('Position Tracking')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Heading angle tracking
plt.subplot(1, 3, 2)
plt.plot(state_history[:, 2], label='Actual Heading')
plt.plot(desired_trajectory[:, 2], '--', label='Desired Heading')
plt.title('Heading Angle Tracking')
plt.xlabel('Time Step')
plt.ylabel('Heading Angle')
plt.legend()

# Control input
plt.subplot(1, 3, 3)
plt.plot(control_history)
plt.title('Control Input')
plt.xlabel('Time Step')
plt.ylabel('Control Input')

plt.tight_layout()
plt.show()