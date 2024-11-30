import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, expm

# Discretization helper
def discretize_system(A, B, dt):
    A_d = expm(A * dt)
    n = A.shape[0]
    aug_matrix = np.zeros((2 * n, 2 * n))
    aug_matrix[:n, :n] = A
    aug_matrix[:n, n:] = B
    exp_aug = expm(aug_matrix * dt)
    B_d = exp_aug[:n, n:]
    return A_d, B_d

# Define the bicycle model
def get_bicycle_model():
    v = 5.0  # Speed (m/s)
    lr = 1.5  # Rear axle to CoG distance
    A = np.array([
        [0, 0, -v * np.sin(0)],
        [0, 0,  v * np.cos(0)],
        [0, 0,  0]
    ])
    B = np.array([
        [0],
        [0],
        [v / lr]
    ])
    return A, B

# Solve Riccati equations for shared control
def solve_riccati(A_d, B_d, Q_av, Q_human, R, N, phase, dt):
    P_av = [None] * (N + 1)
    P_human = [None] * (N + 1)

    P_av[N] = Q_av
    P_human[N] = Q_human

    for k in range(N - 1, -1, -1):
        P_next_av = P_av[k + 1]
        P_next_human = P_human[k + 1]

        if phase == "shared":
            Q_av_k = Q_av * (1 - k * dt / 4)  # Linear decay for AV
            Q_human_k = Q_human * (k * dt / 4)  # Linear increase for Human
        else:
            Q_av_k = Q_av
            Q_human_k = Q_human

        G_av = R + B_d.T @ P_next_av @ B_d
        P_av[k] = Q_av_k + A_d.T @ P_next_av @ A_d - A_d.T @ P_next_av @ B_d @ np.linalg.inv(G_av) @ B_d.T @ P_next_av @ A_d

        G_human = R + B_d.T @ P_next_human @ B_d
        P_human[k] = Q_human_k + A_d.T @ P_next_human @ A_d - A_d.T @ P_next_human @ B_d @ np.linalg.inv(G_human) @ B_d.T @ P_next_human @ A_d

    return P_av, P_human

# Visualization function
def visualize(timesteps, control_inputs, states, labels, title):
    plt.figure(figsize=(12, 6))

    # Plot control inputs
    plt.subplot(2, 1, 1)
    for i, control in enumerate(control_inputs):
        plt.plot(timesteps, control, label=labels[i])
    plt.title(f'{title}: Control Inputs')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    plt.legend()
    plt.grid()

    # Plot states (trajectories)
    plt.subplot(2, 1, 2)
    plt.plot(states[:, 0], states[:, 1], label='Trajectory')
    plt.title(f'{title}: Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Main simulation and visualization
def main():
    dt = 0.1
    total_time = 10
    timesteps = np.arange(0, total_time, dt)
    N_mpc = int(3 / dt)
    N_shared = int(4 / dt)
    N_human = int(3 / dt)

    A, B = get_bicycle_model()
    A_d, B_d = discretize_system(A, B, dt)

    Q_av = np.diag([10, 10, 1])
    Q_human = np.diag([1, 1, 0.1])
    R = np.array([[1]])

    states = [np.array([[0], [0], [0]])]
    u_av = []
    u_human = []

    # Phase 1: AV control (0–3 seconds)
    for _ in range(N_mpc):
        P_inf = solve_discrete_are(A_d, B_d, Q_av, R)
        K_av = np.linalg.inv(R + B_d.T @ P_inf @ B_d) @ (B_d.T @ P_inf @ A_d)
        x = states[-1]
        u = -K_av @ x
        u_av.append(u[0, 0])
        u_human.append(0)
        x_next = A_d @ x + B_d @ u
        states.append(x_next)

    # Phase 2: Shared control (3–7 seconds)
    P_av, P_human = solve_riccati(A_d, B_d, Q_av, Q_human, R, N_shared, "shared", dt)
    for k in range(N_shared):
        x = states[-1]
        K_av = np.linalg.inv(R + B_d.T @ P_av[k] @ B_d) @ (B_d.T @ P_av[k] @ A_d)
        K_human = np.linalg.inv(R + B_d.T @ P_human[k] @ B_d) @ (B_d.T @ P_human[k] @ A_d)
        u_av_k = -K_av @ x
        u_human_k = -K_human @ x
        u_av.append(u_av_k[0, 0])
        u_human.append(u_human_k[0, 0])
        x_next = A_d @ x + B_d @ (u_av_k + u_human_k)
        states.append(x_next)

    # Phase 3: Human control (7–10 seconds)
    for _ in range(N_human):
        P_inf_human = solve_discrete_are(A_d, B_d, Q_human, R)
        K_human = np.linalg.inv(R + B_d.T @ P_inf_human @ B_d) @ (B_d.T @ P_inf_human @ A_d)
        x = states[-1]
        u = -K_human @ x
        u_av.append(0)
        u_human.append(u[0, 0])
        x_next = A_d @ x + B_d @ u
        states.append(x_next)

    # Convert to arrays for plotting
    states = np.hstack(states).T
    u_av = np.array(u_av)
    u_human = np.array(u_human)

    # Visualize
    visualize(
        timesteps,
        [u_av, u_human],
        states,
        ['Autonomous Vehicle', 'Human Driver'],
        'Shared Control Visualization'
    )

# Run the simulation
if __name__ == "__main__":
    main()
