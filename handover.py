import numpy as np
import matplotlib.pyplot as plt

l_f = 1.2
l_r = 1.6
v = 5.0

def tire_slip_angle(delta_f):
    return np.arctan((l_r * np.tan(delta_f)) / (l_f + l_r))

def get_matrices(v, theta, beta, delta_f):
    beta = tire_slip_angle(delta_f)
    A = np.array([
        [0, 0, -v * np.sin(theta+beta)],
        [0, 0, v * np.cos(theta+beta)],
        [0, 0, 0]
    ])
    
    d_beta_d_delta = l_r / ((l_f + l_r) * np.cos(delta_f) ** 2)

    B = np.array([
        [-v * np.sin(theta + beta) * d_beta_d_delta],
        [v * np.cos(theta + beta) * d_beta_d_delta],
        [v * np.cos(beta) / l_r * d_beta_d_delta]
    ])
    
    return A, B

Q_av = np.diag([1, 1, 1])
R_av = np.diag([1])

x = np.array([[0], [0], [0]])
state_history = [x]
control_history = []
tracking_error_history = []

N = 50
u = np.zeros((N,1))

dt = 0.1
desired_state = np.array([[25],[25],[np.pi/2]])

for t in range(100):
    print(f'Current State = {x.T}')
    print(f'Desired State = {desired_state.T}')
    state_error = x - desired_state
    print(np.linalg.norm(state_error))
    tracking_error_history.append(state_error)
    # Get updated matrices with current state and control
    A, B = get_matrices(v, x[2, 0], 0, u[N-1][0])

    P_av = [None] * (N+1)

    P_av[N] = Q_av

    for i in range(N, 0, -1):
        P_av[i-1] = Q_av + A.T @ P_av[i] @ A - (A.T @ P_av[i] @ B) @ np.linalg.pinv(R_av + B.T @ P_av[i] @ B) @ (B.T @ P_av[i] @ A) 

    K_av = [None] * N

    for i in range(N):
        # AV Full Control
        K_av[i] = np.linalg.inv(R_av + B.T @ P_av[i+1] @ B) @ (B.T @ P_av[i+1] @ A)
        print(np.shape(-K_av[i] @ state_error), print(-K_av[i] @ state_error))
        u[i] = -K_av[i] @ state_error
        # print(K_av[i], state_error)

    print(np.shape(u))
    u_star = u[N-1]

    print("control input:", u[N-1][0])
    
    # Update state (simple discrete-time update)
    x = A @ x + B @ u_star
    
    # Store history
    state_history.append(x)
    control_history.append(u_star)

# Convert to numpy arrays for analysis
state_history = np.array(state_history)
control_history = np.array(control_history)
tracking_error_history = np.array(tracking_error_history)
