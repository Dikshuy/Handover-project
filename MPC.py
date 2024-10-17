import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


L = 2.5  
N = 5  
dt = 0.1  

# Define state and control variables
x = ca.MX.sym('x')
y = ca.MX.sym('y')
theta = ca.MX.sym('theta')
v = ca.MX.sym('v')
state = ca.vertcat(x, y, theta, v)

a = ca.MX.sym('a') 
delta = ca.MX.sym('delta') 
control = ca.vertcat(a, delta)

def bicycle_model(state, control):
    x, y, theta, v = state[0], state[1], state[2], state[3]
    a, delta = control[0], control[1]

    x_next = x + v * ca.cos(theta) * dt
    y_next = y + v * ca.sin(theta) * dt
    theta_next = theta + (v / L) * ca.tan(delta) * dt
    v_next = v + a * dt

    return ca.vertcat(x_next, y_next, theta_next, v_next)


t_values = np.linspace(0, 20, 50)

def func(t, t0=10, alpha=1):
    y = 1 / (1 + np.exp(-alpha * (t - t0)))
    return y

target_x = t_values
target_y = func(target_x)

Q = np.eye(4)
R = np.eye(2)


def cost_function(state, control, target, control_prev = None):
    state_cost = ca.mtimes((state[:2] - target).T, Q[:2, :2] @ (state[:2] - target))
    control_cost = ca.mtimes(control.T, R @ control)
    if control_prev is not None:
        jerk_penalty = ca.sumsqr(control - control_prev)
    else:
        jerk_penalty = 0
    return state_cost + control_cost + jerk_penalty

opti = ca.Opti()

X = opti.variable(4, N + 1) 
U = opti.variable(2, N)      

target_param = opti.parameter(2, N) 

previous_control = np.zeros(2)

objective = 0

for k in range(N):
    current_state = X[:, k]
    next_state = X[:, k + 1]
    control_input = U[:, k]

    if k == 0:
        control_previous = U[:, k - 1]
    else:
        control_previous = previous_control

    target = target_param[:, k]  # Use the parameterized target
    objective += cost_function(current_state, control_input, target, control_previous)

    opti.subject_to(next_state == bicycle_model(current_state, control_input))

opti.subject_to(opti.bounded(-0.5, U[0, :], 1.0))  # Acceleration limits
opti.subject_to(opti.bounded(-np.pi/4, U[1, :], np.pi/4))  # Steering angle limits

opti.minimize(objective)

opti.solver('ipopt')

plt.figure(figsize=(12, 8))
plt.ion() 

trajectory_ax = plt.subplot(2, 1, 1)
controls_ax = plt.subplot(2, 1, 2)

x_trajectory = []
y_trajectory = []
time_steps = np.arange(len(target_x)) * dt 
acceleration_all = []
steering_angle_all = []

current_state = np.array([0, 0, 0, 0]) 

for t in range(len(target_x) - N):
    opti.set_initial(X[:, 0], current_state)

    target_segment = np.vstack((target_x[t:t + N], target_y[t:t + N]))
    opti.set_value(target_param, target_segment)

    solution = opti.solve()

    optimal_u = solution.value(U[:, 0])

    previous_control = optimal_u        # minimize jerk

    current_state = solution.value(X[:, 1])

    x_trajectory.append(current_state[0])
    y_trajectory.append(current_state[1])
    acceleration_all.append(optimal_u[0])
    steering_angle_all.append(optimal_u[1])

    predicted_x = solution.value(X[0, :N])  
    predicted_y = solution.value(X[1, :N])  

    trajectory_ax.clear()
    trajectory_ax.plot(x_trajectory, y_trajectory, marker='o', label='Actual Trajectory')
    trajectory_ax.plot(target_x, target_y, 'r--', label='Desired Sine Wave Trajectory')
    trajectory_ax.plot(predicted_x, predicted_y, 'b-', label='Predicted Trajectory', alpha=0.6)
    trajectory_ax.set_title('Vehicle Trajectory')
    trajectory_ax.set_xlabel('X Position (m)')
    trajectory_ax.set_ylabel('Y Position (m)')
    trajectory_ax.legend()
    trajectory_ax.grid()

    controls_ax.clear()
    controls_ax.plot(time_steps[:len(acceleration_all)], acceleration_all, label='Acceleration (m/s²)')
    controls_ax.plot(time_steps[:len(steering_angle_all)], steering_angle_all, label='Steering Angle (rad)')
    controls_ax.set_title('Control Inputs over Time')
    controls_ax.set_xlabel('Time (s)')
    controls_ax.set_ylabel('Control Input')
    controls_ax.legend()
    controls_ax.grid()

    plt.tight_layout()
    plt.pause(0.5)


plt.ioff()
plt.show()
