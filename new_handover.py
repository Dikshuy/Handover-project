import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class AVHandoverSystem:
    def __init__(self, T=10, L=2.5, dt=0.1):
        self.T = T  # Handover duration
        self.L = L  # Wheelbase
        self.dt = dt  # Time step
        self.t = 0  # Current time
        
        # State: [x, y, theta, v]
        self.state = np.array([0., 0., 0., 0.])
        
        # MPC parameters
        self.N = 20  # Prediction horizon
        self.Q = np.diag([1., 1., 0.5, 0.5])  # State cost
        self.R = np.diag([0.1, 0.1])  # Control cost
        self.S = np.diag([1., 1.])  # Human-MPC difference cost
        
        # Constraints
        self.delta_max = np.radians(30)
        self.a_max = 2.0
        self.v_max = 20.0

        # Setup CasADI optimizer
        self.setup_optimizer()

    def alpha(self):
        # Smoother transition function using sigmoid
        k = 6  # Steepness of the sigmoid
        return 1 / (1 + np.exp(-k * (self.t / self.T - 0.5)))

    def kinematic_bicycle_model(self, x, y, theta, v, delta, a):
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v / self.L) * ca.tan(delta)
        dv = a
        return [dx, dy, dtheta, dv]

    def simulate_human_input(self):
        # Simplified human input simulation
        delta_h = 0.1 * np.sin(2 * np.pi * self.t / 5)  # Oscillating steering
        a_h = 0.5 * np.cos(2 * np.pi * self.t / 10)  # Oscillating acceleration
        return np.array([delta_h, a_h])

    def setup_optimizer(self):
        # State variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y, theta, v)
        n_states = states.numel()

        # Control variables
        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        controls = ca.vertcat(delta, a)
        n_controls = controls.numel()

        # Human input variables
        delta_h = ca.SX.sym('delta_h')
        a_h = ca.SX.sym('a_h')
        human_controls = ca.vertcat(delta_h, a_h)

        # Discretized system dynamics
        rhs = self.kinematic_bicycle_model(x, y, theta, v, delta, a)
        f = ca.Function('f', [states, controls], [ca.vertcat(*rhs)])

        # Decision variables
        U = ca.SX.sym('U', n_controls, self.N)
        X = ca.SX.sym('X', n_states, self.N+1)
        P = ca.SX.sym('P', n_states + n_states + n_controls)  # Parameters: initial state, reference state, human control

        # Objective function
        obj = 0
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            st_ref = P[n_states:2*n_states]
            human_con = P[2*n_states:]
            alpha_k = 1 / (1 + ca.exp(-6 * ((self.t + k*self.dt) / self.T - 0.5)))
            combined_con = (1 - alpha_k) * con + alpha_k * human_con
            
            obj += ca.mtimes([(st - st_ref).T, self.Q, (st - st_ref)])
            obj += ca.mtimes([con.T, self.R, con])
            obj += ca.mtimes([(con - human_con).T, self.S, (con - human_con)])

        # Constraints
        g = []
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            st_next = X[:, k+1]
            human_con = P[2*n_states:]
            alpha_k = 1 / (1 + ca.exp(-6 * ((self.t + k*self.dt) / self.T - 0.5)))
            combined_con = (1 - alpha_k) * con + alpha_k * human_con
            
            st_next_pred = st + f(st, combined_con) * self.dt
            g.append(st_next - st_next_pred)

        # Bounds and initial guess
        lbx = [-ca.inf] * (n_states * (self.N+1)) + [-self.delta_max, -self.a_max] * self.N
        ubx = [ca.inf] * (n_states * (self.N+1)) + [self.delta_max, self.a_max] * self.N
        lbg = [0] * n_states * self.N
        ubg = [0] * n_states * self.N

        # NLP Problem
        nlp = {'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
               'f': obj,
               'g': ca.vertcat(*g),
               'p': P}

        # Create solver instance
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store CasADI functions
        self.f = f
        self.U = U
        self.X = X
        self.P = P

    def step(self, x_ref):
        u_human = self.simulate_human_input()
        
        # Initial state and parameters
        x0 = self.state
        p = np.concatenate([x0, x_ref, u_human])
        
        # Initial guess
        x_init = np.zeros((4, self.N+1))
        x_init[:, 0] = x0
        u_init = np.zeros((2, self.N))
        nlp_init = np.concatenate([x_init.reshape((-1, 1)), u_init.reshape((-1, 1))])

        # Solve NLP
        sol = self.solver(x0=nlp_init, lbx=self.solver.lbx, ubx=self.solver.ubx,
                          lbg=self.solver.lbg, ubg=self.solver.ubg, p=p)

        # Extract solution
        sol_x = sol['x'].full().flatten()
        x_opt = sol_x[:4*(self.N+1)].reshape((4, -1))
        u_opt = sol_x[4*(self.N+1):].reshape((2, -1))

        # Get optimal control
        u_mpc = u_opt[:, 0]
        
        # Combine MPC and human control
        alpha = self.alpha()
        u = (1 - alpha) * u_mpc + alpha * u_human
        
        # Update state
        self.state += np.array(self.f(self.state, u)).flatten() * self.dt
        self.t += self.dt
        
        return self.state, u, u_mpc, u_human

def run_simulation():
    system = AVHandoverSystem()
    
    # Simple circular reference trajectory
    t_sim = np.arange(0, 20, system.dt)
    x_ref = np.column_stack([
        10 * np.cos(0.1 * t_sim),
        10 * np.sin(0.1 * t_sim),
        0.1 * np.ones_like(t_sim),
        1 * np.ones_like(t_sim)
    ])
    
    states = []
    controls = []
    
    for x_ref_k in x_ref:
        state, u, u_mpc, u_human = system.step(x_ref_k)
        states.append(state)
        controls.append((u, u_mpc, u_human))
    
    states = np.array(states)
    controls = np.array(controls)
    
    # Plotting
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.plot(states[:, 0], states[:, 1], label='Vehicle')
    plt.plot(x_ref[:, 0], x_ref[:, 1], '--', label='Reference')
    plt.title('Vehicle Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(t_sim, controls[:, 0, 0], label='Combined')
    plt.plot(t_sim, controls[:, 1, 0], '--', label='MPC')
    plt.plot(t_sim, controls[:, 2, 0], ':', label='Human')
    plt.title('Steering Control')
    plt.xlabel('Time')
    plt.ylabel('Steering angle (rad)')
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(t_sim, controls[:, 0, 1], label='Combined')
    plt.plot(t_sim, controls[:, 1, 1], '--', label='MPC')
    plt.plot(t_sim, controls[:, 2, 1], ':', label='Human')
    plt.title('Acceleration Control')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot(t_sim, states[:, 3])
    plt.title('Vehicle Speed')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    
    plt.subplot(3, 2, 5)
    alphas = [system.alpha() for _ in t_sim]
    plt.plot(t_sim, alphas)
    plt.title('Control Authority Function (α)')
    plt.xlabel('Time')
    plt.ylabel('α')
    
    plt.subplot(3, 2, 6)
    plt.plot(t_sim, np.linalg.norm(controls[:, 0] - controls[:, 1], axis=1), label='Combined - MPC')
    plt.plot(t_sim, np.linalg.norm(controls[:, 0] - controls[:, 2], axis=1), label='Combined - Human')
    plt.title('Control Input Differences')
    plt.xlabel('Time')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()