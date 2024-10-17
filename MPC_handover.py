import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

L = 2.5  # Wheelbase
N = 5    # Prediction horizon
dt = 0.1  # Time step

# Add takeover parameters
TAKEOVER_TRIGGER_TIME = 10  # Time at which takeover is triggered
TAKEOVER_DURATION = 5  # Duration of the takeover process

# Define state and control variables
x = ca.MX.sym('x')
y = ca.MX.sym('y')
theta = ca.MX.sym('theta')
v = ca.MX.sym('v')
state = ca.vertcat(x, y, theta, v)

a_h = ca.MX.sym('a_h')  # Human driver acceleration input
delta_h = ca.MX.sym('delta_h')  # Human driver steering input
a_ad = ca.MX.sym('a_ad')  # AD agent acceleration input
delta_ad = ca.MX.sym('delta_ad')  # AD agent steering input

control_h = ca.vertcat(a_h, delta_h)
control_ad = ca.vertcat(a_ad, delta_ad)

# # Time-varying blending factor (alpha) for shared control
# def alpha(t, t0=10, T=5, alpha_rate=1):
#     return 1 / (1 + np.exp(-alpha_rate * ((t - t0) / T)))

# Modified alpha function to include trigger time
def alpha(t, trigger_time=TAKEOVER_TRIGGER_TIME, duration=TAKEOVER_DURATION):
    if t < trigger_time:
        return 0
    elif t >= trigger_time + duration:
        return 1
    else:
        return (t - trigger_time) / duration

# Blended control inputs
def blended_control(alpha_t, control_h, control_ad):
    return alpha_t * control_h + (1 - alpha_t) * control_ad

# Vehicle dynamics using bicycle model
def bicycle_model(state, control):
    x, y, theta, v = state[0], state[1], state[2], state[3]
    a, delta = control[0], control[1]

    x_next = x + v * ca.cos(theta) * dt
    y_next = y + v * ca.sin(theta) * dt
    theta_next = theta + (v / L) * ca.tan(delta) * dt
    v_next = v + a * dt

    return ca.vertcat(x_next, y_next, theta_next, v_next)

def human_input_estimate(t):
    # Simulating more erratic human behavior
    steering = 0.1 * np.sin(2 * np.pi * t / 3) + 0.05 * np.random.randn()
    acceleration = 0.5 * np.cos(2 * np.pi * t / 5) + 0.1 * np.random.randn()
    return np.array([acceleration, steering])

t_values = np.linspace(0, 20, 50)

def func(t, t0=10, alpha=1):
    y = 1 / (1 + np.exp(-alpha * (t - t0)))
    return y

target_x = t_values
target_y = func(target_x)

# Define cost function (tracking + jerk penalty)
Q = np.eye(4)
R = np.eye(2)

# def cost_function(state, control, target, control_h, control_ad, control_prev=None):
#     state_cost = ca.mtimes((state[:2] - target).T, Q[:2, :2] @ (state[:2] - target))
#     control_cost = ca.mtimes(control.T, R @ control)
#     jerk_penalty = ca.sumsqr(control - control_prev) if control_prev is not None else 0
#     transition_penalty = ca.sumsqr(control_h - control_ad)
#     return state_cost + control_cost + jerk_penalty + transition_penalty

def cost_function(state, control, target, control_h, control_ad, control_prev=None, alpha_t=0):
    state_cost = ca.mtimes((state[:2] - target).T, Q[:2, :2] @ (state[:2] - target))
    control_cost = ca.mtimes(control.T, R @ control)
    jerk_penalty = ca.sumsqr(control - control_prev) if control_prev is not None else 0
    human_alignment_cost = alpha_t * ca.sumsqr(control - control_h)
    ad_alignment_cost = (1 - alpha_t) * ca.sumsqr(control - control_ad)
    return state_cost + control_cost + jerk_penalty + human_alignment_cost + ad_alignment_cost


# Initialize optimization problem
opti = ca.Opti()
X = opti.variable(4, N + 1)  # State trajectory
U_h = opti.variable(2, N)    # Human control inputs
U_ad = opti.variable(2, N)   # AD agent control inputs
U_combined = opti.variable(2, N)   # combined control inputs
target_param = opti.parameter(2, N)  # Target trajectory
alpha_param = opti.parameter(N)

# Initialize variables for plotting
x_trajectory, y_trajectory = [], []
acceleration_all, steering_angle_all = [], []
time_steps = np.arange(len(target_x)) * dt 

current_state = np.array([0, 0, 0, 0])  # Initial state
previous_control = np.zeros(2)  # Store previous control to minimize jerk

objective = 0

# Build the optimization problem
for k in range(N):
    current_state_var = X[:, k]
    next_state_var = X[:, k + 1]

    control_h = U_h[:, k]
    control_ad = U_ad[:, k]
    control_combined = U_combined[:, k]

    # Time-varying alpha for blending
    alpha_t = alpha_param[k]

    # Enforce blending constraint
    opti.subject_to(control_combined == blended_control(alpha_t, control_h, control_ad))

    # Compute the target for the current step
    target = target_param[:, k]

    # Accumulate the cost
    objective += cost_function(current_state_var, control_combined, target, control_h, control_ad, previous_control, alpha_t)

    # Enforce system dynamics
    opti.subject_to(next_state_var == bicycle_model(current_state_var, control_combined))

    # Update the previous control for jerk penalty
    previous_control = control_combined

# Add control constraints
opti.subject_to(opti.bounded(-0.5, U_h[0, :], 1.0))  # Acceleration limits
opti.subject_to(opti.bounded(-np.pi / 4, U_h[1, :], np.pi / 4))  # Steering limits
opti.subject_to(opti.bounded(-0.5, U_ad[0, :], 1.0))  # Acceleration limits
opti.subject_to(opti.bounded(-np.pi / 4, U_ad[1, :], np.pi / 4))  # Steering limits

opti.minimize(objective)  # Set the objective function
opti.solver('ipopt')  # Choose solver

# Plotting setup
plt.figure(figsize=(10, 8))
trajectory_ax = plt.subplot(3, 1, 1)
controls_h_ax = plt.subplot(3, 1, 2)
controls_ad_ax = plt.subplot(3, 1, 3)

takeover_triggered = False

for t in range(len(target_x) - N):
    current_time = t * dt

    # Check if takeover should be triggered
    if current_time >= TAKEOVER_TRIGGER_TIME and not takeover_triggered:
        print(f"Takeover triggered at t = {current_time:.2f}s")
        takeover_triggered = True

    # Set the initial state for optimization
    opti.set_initial(X[:, 0], current_state)

    # Set the target trajectory segment
    target_segment = np.vstack((target_x[t:t + N], target_y[t:t + N]))
    opti.set_value(target_param, target_segment)

    # Compute alpha values for the horizon
    alpha_values = [alpha(current_time + i*dt) for i in range(N)]
    opti.set_value(alpha_param, alpha_values)

    # Estimate human input (this would come from actual sensors in a real system)
    human_input = human_input_estimate(current_time)
    opti.set_initial(U_h, np.tile(human_input, (N, 1)).T)

    # Solve the optimization problem
    solution = opti.solve()

    # Get optimal control inputs
    optimal_u_combined = solution.value(U_combined[:, 0])
    optimal_u_h = solution.value(U_h[:, 0])
    optimal_u_ad = solution.value(U_ad[:, 0])

    # Update the current state
    current_state = solution.value(X[:, 1])

    # Store trajectories and control inputs
    x_trajectory.append(current_state[0])
    y_trajectory.append(current_state[1])
    acceleration_all.append(optimal_u_combined[0])
    steering_angle_all.append(optimal_u_combined[1])

    predicted_x = solution.value(X[0, :N])  
    predicted_y = solution.value(X[1, :N]) 

    # Plot the trajectory and controls
    trajectory_ax.clear()
    trajectory_ax.plot(x_trajectory, y_trajectory, marker='o', label='Actual Trajectory')
    trajectory_ax.plot(target_x, target_y, 'r--', label='Desired Sine Wave Trajectory')
    trajectory_ax.plot(predicted_x, predicted_y, 'b-', label='Predicted Trajectory', alpha=0.6)
    trajectory_ax.set_title('Vehicle Trajectory')
    trajectory_ax.set_xlabel('X Position (m)')
    trajectory_ax.set_ylabel('Y Position (m)')
    trajectory_ax.legend()
    trajectory_ax.grid()

    controls_h_ax.clear()
    controls_h_ax.plot(time_steps[:len(acceleration_all)], acceleration_all, label='Acceleration (m/s²)')
    controls_h_ax.plot(time_steps[:len(steering_angle_all)], steering_angle_all, label='Steering Angle (rad)')
    controls_h_ax.set_title('Control Inputs over Time')
    controls_h_ax.set_xlabel('Time (s)')
    controls_h_ax.set_ylabel('Human Driver Controls')
    controls_h_ax.legend()
    controls_h_ax.grid()

    controls_ad_ax.clear()
    controls_ad_ax.plot(time_steps[:len(acceleration_all)], acceleration_all, label='Acceleration (m/s²)')
    controls_ad_ax.plot(time_steps[:len(steering_angle_all)], steering_angle_all, label='Steering Angle (rad)')
    controls_ad_ax.set_title('Control Inputs over Time')
    controls_ad_ax.set_xlabel('Time (s)')
    controls_ad_ax.set_ylabel('AD Agent Controls')
    controls_ad_ax.legend()
    controls_ad_ax.grid()

    plt.tight_layout()
    plt.pause(0.5)


plt.show()
