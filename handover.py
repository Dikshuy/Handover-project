import numpy as np
import matplotlib.pyplot as plt

l_f = 1.2
l_r = 1.4
l = l_f + l_r
v = 1.4

Q_av = np.diag([10, 100, 1])
R_av = np.diag([0.1])

Q_human = np.diag([10, 10, 1])
R_human = np.diag([1])

dt = 0.1
time = 30
N = 50

t1 = 7  # av control
t2 = 14 # handover time: 7 seconds
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
    lane_change_time = 7
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


x = np.array([[0], [0], [0]])
state_history = []
control_history = []
tracking_error_history = []

T, desired_trajectory = get_trajectory(v, time, dt)
simulation_time = len(T)

u = np.zeros(N)

# q_sharing = True
q_sharing = False

for t in range(simulation_time):
    run_time = t * dt
    state_error = x - desired_trajectory[t].reshape(-1, 1)
    A, B = get_matrices(v, x[2, 0], u[0])

    if run_time < t1:
        u_av = control_input(A, B, Q_av, R_av, state_error)
        u = u_av

    elif t1 <= run_time < t2:
        alpha = (run_time - t1) / (t2 - t1)
        if q_sharing:
            Q_av_ = (1 - alpha) * Q_av
            Q_human_ = (alpha) * Q_human
            u_av = control_input(A, B, Q_av_, R_av, state_error)
            u_human = control_input(A, B, Q_human_, R_human, state_error)
            u = u_av + u_human
        else:
            u_av = control_input(A, B, Q_av, R_av, state_error)
            u_human = control_input(A, B, Q_human, R_human, state_error)
            u = (1-alpha) * u_av + alpha * u_human

    elif run_time >= t2:
        u_human = control_input(A, B, Q_human, R_human, state_error)
        u = u_human

    else:
        print("panic!")

    print("control input:", u[0])

    x = update(x, u[0])
    
    tracking_error_history.append(state_error.flatten())
    state_history.append(x.flatten())
    control_history.append(u[0])

# Convert to numpy arrays for analysis
state_history = np.array(state_history)
control_history = np.array(control_history)
tracking_error_history = np.array(tracking_error_history)

plt.figure(figsize=(14, 6))

# Position tracking
plt.subplot(1, 3, 1)
plt.plot(state_history[:, 0], state_history[:, 1], label='Actual Trajectory', linewidth=2)
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], '--', label='Desired Trajectory', linewidth=2)
plt.title('Position Tracking', fontsize=14)
plt.ylim(-0.5,1.5)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.minorticks_on()

# Heading angle tracking
plt.subplot(1, 3, 2)
plt.plot(T, state_history[:, 2], label='Actual Heading', linewidth=2)
plt.plot(T, desired_trajectory[:, 2], '--', label='Desired Heading', linewidth=2)
plt.title('Heading Angle Tracking', fontsize=14)
plt.ylim(-0.1, 0.3)
plt.xlabel('Time [s]')
plt.ylabel('Heading Angle [rad]')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.minorticks_on()

# Control input
plt.subplot(1, 3, 3)
plt.plot(control_history, linewidth=2)
plt.title('Control Input', fontsize=14)
plt.ylim(-0.3, 0.3)
plt.xlabel('Time Step')
plt.ylabel('Steering Angle [rad]')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.minorticks_on()

plt.tight_layout()
plt.show()