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


'''
from typing import Tuple, Optional, List
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


class BicycleMPC:
    """
    Model Predictive Controller for a Bicycle Kinematic Model.
    """

    def __init__(
        self, 
        wheelbase: float = 2.5, 
        velocity: float = 5.0, 
        horizon_length: int = 5, 
        time_step: float = 0.1
    ):
        self.L = wheelbase
        self.v = velocity
        self.N = horizon_length
        self.dt = time_step
        
        self.Q = np.eye(3)
        
        self.R = np.eye(1)

    def create_bicycle_model(self) -> ca.Function:
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        state = ca.vertcat(x, y, theta)
        
        delta = ca.MX.sym('delta')
        control = ca.vertcat(delta)

        x_next = x + self.v * ca.cos(theta) * self.dt
        y_next = y + self.v * ca.sin(theta) * self.dt
        theta_next = theta + (self.v / self.L) * ca.tan(delta) * self.dt
        
        next_state = ca.vertcat(x_next, y_next, theta_next)
        
        return ca.Function('bicycle_model', [state, control], [next_state])

    @staticmethod
    def generate_reference_trajectory(
        t_values: np.ndarray, 
        t0: float = 10.0, 
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_values = 1 / (1 + np.exp(-alpha * (t_values - t0)))
        return t_values, y_values

    def cost_function(
        self, 
        state: ca.MX, 
        control: ca.MX, 
        target: ca.MX, 
        control_prev: Optional[ca.MX] = None
    ) -> ca.MX:
        state_cost = ca.mtimes((state[:2] - target).T, self.Q[:2, :2] @ (state[:2] - target))
        
        control_cost = ca.mtimes(control.T, self.R @ control)
        
        if control_prev is not None:
            jerk_penalty = ca.sumsqr(control - control_prev)
        else:
            jerk_penalty = 0
        
        return ca.sum1(state_cost) + ca.sum1(control_cost) + jerk_penalty

    def run_mpc_simulation(self) -> Tuple[List[float], List[float], List[float]]:
        bicycle_model = self.create_bicycle_model()

        t_values = np.linspace(0, 20, 50)
        target_x, target_y = self.generate_reference_trajectory(t_values)

        opti = ca.Opti()
        X = opti.variable(3, self.N + 1)  # State trajectory
        U = opti.variable(1, self.N)      # Control inputs

        opti.set_initial(X, 0)
        opti.set_initial(U, 0)

        target_param = opti.parameter(2, self.N)

        previous_control = np.zeros(1)

        objective = 0

        for k in range(self.N):
            current_state = X[:, k]
            next_state = X[:, k + 1]
            control_input = U[:, k]

            control_previous = previous_control if k == 0 else U[:, k - 1]
            
            target = target_param[:, k]
            
            objective += self.cost_function(current_state, control_input, target, control_previous)

        # Dynamic constraints
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == bicycle_model(X[:, k], U[:, k]))
        
        # Steering angle constraints
        opti.subject_to(opti.bounded(-np.pi/4, U[0, :], np.pi/4))

        opti.minimize(objective)
        opti.solver('ipopt')

        x_trajectory: List[float] = []
        y_trajectory: List[float] = []
        steering_angle_all: List[float] = []
        current_state = np.array([0, 0, 0])

        time_steps = np.arange(len(target_x)) * self.dt
        for t in range(len(target_x) - self.N):
            opti.set_initial(X[:, 0], current_state)

            target_segment = np.vstack((target_x[t:t + self.N], target_y[t:t + self.N]))
            opti.set_value(target_param, target_segment)

            solution = opti.solve()

            optimal_u = solution.value(U[:, 0])
            previous_control = optimal_u

            current_state = solution.value(X[:, 1])
            
            x_trajectory.append(current_state[0])
            y_trajectory.append(current_state[1])
            steering_angle_all.append(optimal_u)

        return x_trajectory, y_trajectory, steering_angle_all

    def visualize_results(
        self, 
        x_trajectory: List[float], 
        y_trajectory: List[float], 
        steering_angles: List[float]
    ):
        t_values = np.linspace(0, 20, 50)
        target_x, target_y = self.generate_reference_trajectory(t_values)
        time_steps = np.arange(len(target_x)) * self.dt

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_trajectory, y_trajectory, marker='o', label='Actual Trajectory')
        plt.plot(target_x, target_y, 'r--', label='Desired Trajectory')
        plt.title('Vehicle Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_steps[:len(steering_angles)], steering_angles, label='Steering Angle (rad)')
        plt.title('Control Inputs over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (rad)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
def main():
    mpc = BicycleMPC()

    x_traj, y_traj, steering_angles = mpc.run_mpc_simulation()

    mpc.visualize_results(x_traj, y_traj, steering_angles)

if __name__ == "__main__":
    main()
'''