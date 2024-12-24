import numpy as np
import matplotlib.pyplot as plt
from path import trajectory

l_f = 1.2
l_r = 1.4
length = l_f + l_r
v = 3
width = 1.8

Q_av = np.diag([100, 100, 10])
R_av = np.diag([1])

Q_human = np.diag([100, 100, 1])
R_human = np.diag([1])

dt = 0.1
time = 61
N = 50

t1 = 5   # av control
t2 = 55  # shared control
t3 = (time-t2)  # human control

def get_trajectory(width):
    waypoints = trajectory(width)
    x, y, theta = waypoints['x'], waypoints['y'], waypoints['theta']
    waypoints = np.vstack([x, y, theta]).T
    return waypoints

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
        [-v * np.sin(theta+beta) * d_beta_d_delta],
        [v * np.cos(theta+beta) * d_beta_d_delta],
        [v * np.cos(beta)/l_r * d_beta_d_delta]
    ])
    
    return A, B

def update(state, input):
    x, y, theta = state
    delta = input

    beta = np.arctan(l_r / length * np.tan(delta))

    x = x + v * np.cos(theta + beta) * dt
    y = y + v * np.sin(theta + beta) * dt
    theta = theta + v / length * np.sin(beta) * dt
    state = np.array([x, y, theta])

    return state

def control_input(A, B, Q, R, error):
    P = [None] * (N+1)
    K = [None] * N
    P[N] = Q

    for i in range(N, 0, -1):
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

    for i in range(N):
        K[i] = np.linalg.inv(R + B.T @ P[i+1] @ B) @ (B.T @ P[i+1] @ A)
    
    u = np.zeros(N)
    u[0] = -K[0] @ error

    return u

x = np.array([[0], [0], [0]])
state_history = []
control_history = []
tracking_error_history = []

desired_trajectory = get_trajectory(width)
simulation_time = len(desired_trajectory)
T = np.arange(simulation_time) * dt

u = np.zeros(N)
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
            Q_human_ = alpha * Q_human
            u_av = control_input(A, B, Q_av_, R_av, state_error)
            u_human = control_input(A, B, Q_human_, R_human, state_error)
            u = u_av + u_human
        else:
            u_av = control_input(A, B, Q_av, R_av, state_error)
            u_human = control_input(A, B, Q_human, R_human, state_error)
            u = (1-alpha) * u_av + alpha * u_human

    else:
        u_human = control_input(A, B, Q_human, R_human, state_error)
        u = u_human

    x = update(x, u[0])
    
    tracking_error_history.append(state_error.flatten())
    state_history.append(x.flatten())
    control_history.append(u[0])


state_history = np.array(state_history)
control_history = np.array(control_history)
tracking_error_history = np.array(tracking_error_history)

# Plotting results
plt.figure(figsize=(16, 5))

# Position tracking
plt.subplot(1, 3, 1)
plt.plot(state_history[:, 0], state_history[:, 1], label='Actual Trajectory', linewidth=2)
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], '--', label='Desired Trajectory', linewidth=2)
plt.title('Position Tracking', fontsize=14)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Heading angle tracking
plt.subplot(1, 3, 2)
plt.plot(T, state_history[:, 2], label='Actual Heading', linewidth=2)
plt.plot(T, desired_trajectory[:, 2], '--', label='Desired Heading', linewidth=2)
plt.title('Heading Angle Tracking', fontsize=14)
plt.xlabel('Time [s]')
plt.ylabel('Heading Angle [rad]')
plt.legend()
plt.grid(True)

# Control input
plt.subplot(1, 3, 3)
plt.plot(T, control_history, linewidth=2)
plt.title('Control Input', fontsize=14)
plt.xlabel('Time [s]')
plt.ylabel('Steering Angle [rad]')
plt.grid(True)

plt.tight_layout()
plt.show()