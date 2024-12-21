import numpy as np
import matplotlib.pyplot as plt

# Function to generate waypoints from the track
def path(car_width=1.8):
    sections = [
        {"length": 12, "width": 1.1 * car_width + 0.25},    # Section 1
        {"length": 13.5, "width": (1.1 * car_width + 0.25 + 1) + (1) + (1 + car_width)},  # Section 2
        {"length": 11, "width": 1 + car_width},             # Section 3
        {"length": 12.5, "width": (1.3 * car_width + 0.25 + 1) + (1) + (1 + car_width)},  # Section 4
        {"length": 12, "width": 1.3 * car_width + 0.25},    # Section 5
    ]

    waypoints = []
    current_x = 0
    base_y = -(1.1 * car_width + 0.25) / 2

    # Section 1 (straight)
    length = sections[0]["length"]
    for i in range(100):
        x = current_x + i * length / 100
        waypoints.append([x, base_y + sections[0]["width"] / 2, 0])
    current_x += length

    # Section 2 (transition up)
    length = sections[1]["length"]
    for i in range(100):
        x = current_x + i * length / 100
        waypoints.append([x, base_y + sections[1]["width"] / 2, 0])
    current_x += length

    # Section 3 (narrow section with offset)
    length = sections[2]["length"]
    for i in range(100):
        x = current_x + i * length / 100
        waypoints.append([x, base_y + sections[1]["width"] / 2 - (car_width + 1) / 2, 0])
    current_x += length

    # Section 4 (transition down)
    length = sections[3]["length"]
    for i in range(100):
        x = current_x + i * length / 100
        waypoints.append([x, base_y + sections[3]["width"] / 2, 0])
    current_x += length

    # Section 5 (final straight)
    length = sections[4]["length"]
    for i in range(100):
        x = current_x + i * length / 100
        waypoints.append([x, base_y + sections[4]["width"] / 2, 0])

    return np.array(waypoints)

# Updated trajectory function
def get_trajectory(v, time, dt):
    waypoints = path()
    T = np.linspace(0, time, len(waypoints))
    return T, waypoints

# Main LQR implementation (reusing your code)
def tire_slip_angle(delta_f):
    return np.arctan((1.4 * np.tan(delta_f)) / (1.2 + 1.4))

def get_matrices(v, theta, delta_f):
    beta = tire_slip_angle(delta_f)
    d_beta_d_delta = 1.4 / ((1.2 + 1.4) * np.cos(delta_f) ** 2)

    A = np.array([
        [0, 0, -v * np.sin(theta + beta)],
        [0, 0, v * np.cos(theta + beta)],
        [0, 0, 0]
    ])

    B = np.array([
        [-v * np.sin(theta + beta) * d_beta_d_delta],
        [v * np.cos(theta + beta) * d_beta_d_delta],
        [v * np.cos(beta) / 1.4 * d_beta_d_delta]
    ])

    return A, B

def update(state, input):
    x, y, theta = state
    delta = input

    beta = np.arctan(1.4 / (1.2 + 1.4) * np.tan(delta))

    x = x + 1.4 * np.cos(theta + beta) * 0.1
    y = y + 1.4 * np.sin(theta + beta) * 0.1
    theta = theta + 1.4 / (1.2 + 1.4) * np.sin(beta) * 0.1
    state = np.array([x, y, theta])

    return state

def control_input(A, B, Q, R, error):
    N = 50
    P = [None] * (N + 1)
    K = [None] * N
    P[N] = Q

    for i in range(N, 0, -1):
        P[i - 1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

    K[0] = np.linalg.inv(R + B.T @ P[1] @ B) @ (B.T @ P[1] @ A)

    u = -K[0] @ error

    return u

# Simulation
Q_av = np.diag([10, 100, 1])
R_av = np.diag([0.1])

Q_human = np.diag([10, 10, 1])
R_human = np.diag([1])

x = np.array([[0], [0], [0]])
state_history = []
control_history = []
tracking_error_history = []

T, desired_trajectory = get_trajectory(1.4, 30, 0.1)
simulation_time = len(T)

for t in range(simulation_time):
    run_time = t * 0.1
    state_error = x.flatten() - desired_trajectory[t]
    A, B = get_matrices(1.4, x[2, 0], 0)
    u = control_input(A, B, Q_av, R_av, state_error)

    x = update(x.flatten(), u)

    tracking_error_history.append(state_error)
    state_history.append(x.flatten())
    control_history.append(u)

# Visualization
state_history = np.array(state_history)
plt.figure(figsize=(14, 6))
plt.plot(state_history[:, 0], state_history[:, 1], label="Actual Trajectory")
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], '--', label="Desired Trajectory")
plt.legend()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("LQR Path Tracking")
plt.grid(True)
plt.show()
