import numpy as np
import matplotlib.pyplot as plt

# only steering control as of now
# add updates for acceleration control
# use better controller then simple K controller

class KinematicBicycleModel:
    def __init__(self, L):
        """
        Initialize kinematic bicycle model
        L: wheelbase (distance between front and rear axles)
        """
        self.L = L
        
    def state_space(self, t, state, steering_angle):
        """
        Kinematic bicycle model state space
        state = [x, y, θ]
        where:
        x: global x position
        y: global y position
        θ: heading angle
        steering_angle: front wheel steering angle
        """
        x, y, theta = state
        v = 20.0  # constant velocity for simplicity
        
        # State derivatives
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = (v / self.L) * np.tan(steering_angle)
        
        return [dx, dy, dtheta]

class SharedController:
    def __init__(self, model, safety_params):
        """
        Initialize shared controller
        model: KinematicBicycleModel instance
        safety_params: dictionary containing safety thresholds
        """
        self.model = model
        self.safety_params = safety_params
        
    def calculate_tracking_error(self, state, reference):
        """
        Calculate normalized tracking error metrics
        """
        x, y, theta = state
        x_ref, y_ref, theta_ref = reference
        
        # Calculate cross-track error and heading error
        cross_track = np.sqrt((x - x_ref)**2 + (y - y_ref)**2)
        heading_error = np.abs(theta - theta_ref)
        
        # Normalize errors
        norm_cross_track = min(cross_track / self.safety_params['max_cross_track'], 1.0)
        norm_heading = min(heading_error / self.safety_params['max_heading_error'], 1.0)
        
        # Combined tracking metric (higher means better tracking)
        tracking_metric = 1.0 - (0.7 * norm_cross_track + 0.3 * norm_heading)
        return max(tracking_metric, 0.0)
    
    def calculate_control_safety(self, state, steering_angle):
        """
        Calculate safety metric based on steering angle and rate
        """
        # Normalize steering angle
        norm_steering = abs(steering_angle) / self.safety_params['max_steering']
        
        # Safety metric (higher means safer)
        safety_metric = 1.0 - norm_steering
        return max(safety_metric, 0.0)
    
    def calculate_shapley_values(self, state, reference, av_steering, human_steering, 
                               human_readiness, system_reliability):
        """
        Calculate Shapley values for control allocation
        """
        # Get current performance metrics
        tracking = self.calculate_tracking_error(state, reference)
        
        # Calculate individual and coalition values
        def v_empty(): 
            return 0.0
        
        def v_av(): 
            # AV contribution based on tracking performance and system reliability
            safety_av = self.calculate_control_safety(state, av_steering)
            return tracking * system_reliability * safety_av
        
        def v_human(): 
            # Human contribution based on readiness and control safety
            safety_human = self.calculate_control_safety(state, human_steering)
            return tracking * human_readiness * safety_human
        
        def v_coalition():
            # Coalition value considering synergy
            combined_safety = self.calculate_control_safety(
                state, 
                0.5 * (av_steering + human_steering)  # Simple average for coalition
            )
            synergy_factor = 1.2  # Bonus for cooperation
            return min(
                synergy_factor * max(v_av(), v_human()) * combined_safety,
                1.0  # Normalize to 1.0
            )
        
        # Calculate Shapley values
        shapley_av = (v_av() + (v_coalition() - v_human())) / 2
        shapley_human = (v_human() + (v_coalition() - v_av())) / 2
        
        # Normalize
        total = shapley_av + shapley_human
        if total <= 0:
            return 0.5, 0.5  # Equal distribution if both values are 0
        
        return shapley_av/total, shapley_human/total
    
    def blend_control_inputs(self, state, reference, av_steering, human_steering, 
                           human_readiness, system_reliability):
        """
        Blend steering inputs based on Shapley values and safety constraints
        """
        # Calculate control weights using Shapley values
        av_weight, human_weight = self.calculate_shapley_values(
            state, reference, av_steering, human_steering, 
            human_readiness, system_reliability
        )
        
        # Initial blended input
        blended_steering = av_weight * av_steering + human_weight * human_steering
        
        # Safety checks and adjustments
        tracking_performance = self.calculate_tracking_error(state, reference)
        safety_metric = self.calculate_control_safety(state, blended_steering)
        
        # Increase AV control if tracking or safety is poor
        if tracking_performance < self.safety_params['min_tracking'] or \
           safety_metric < self.safety_params['min_safety']:
            # Gradually increase AV control
            correction_factor = min(
                1.0 - tracking_performance,
                1.0 - safety_metric
            )
            av_weight = min(av_weight + correction_factor * (1.0 - av_weight), 1.0)
            human_weight = 1.0 - av_weight
            blended_steering = av_weight * av_steering + human_weight * human_steering
        
        # Final safety constraint on steering angle
        constrained_steering = np.clip(
            blended_steering,
            -self.safety_params['max_steering'],
            self.safety_params['max_steering']
        )
        
        return constrained_steering, (av_weight, human_weight)


class TransitionScenario:
    def __init__(self):
        # Time parameters
        self.t0 = 3.0    # Start of transition
        self.tT = 7.0    # End of transition
        self.T_total = 10.0  # Total simulation time
        
        # Initialize model and controller
        self.L = 2.7  # wheelbase
        self.model = KinematicBicycleModel(self.L)
        
        self.safety_params = {
            'max_steering': np.pi/6,     # 30 degrees
            'max_cross_track': 2.0,      # meters
            'max_heading_error': np.pi/4, # 45 degrees
            'min_tracking': 0.3,
            'min_safety': 0.3
        }
        
        self.controller = SharedController(self.model, self.safety_params)
        
    def transition_function(self, t):
        """
        Smooth transition function using sigmoid
        Returns (av_authority, human_authority)
        """
        if t < self.t0:
            return 1.0, 0.0
        elif t > self.tT:
            return 0.0, 1.0
        else:
            # Smooth sigmoid transition
            progress = (t - self.t0) / (self.tT - self.t0)
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            return 1.0 - sigmoid, sigmoid
    
    def reference_trajectory(self, t):
        """
        Generate reference trajectory (curved path)
        """
        x_ref = 20 * t
        y_ref = 5 * t  # straight path
        # y_ref = 5 * np.sin(0.2 * t)  # Sinusoidal path
        theta_ref = np.arctan2(y_ref, x_ref)#np.cos(0.2 * t), 4)
        return np.array([x_ref, y_ref, theta_ref])
    
    def simulate(self):
        # Simulation time points
        self.t_eval = np.linspace(0, self.T_total, 200)
        
        # Storage for results
        self.results = {
            'state': [],
            'steering': [],
            'av_weight': [],
            'human_weight': [],
            'tracking_error': [],
            'safety_metric': [],
            'time': []
        }
        
        # Initial state
        init = self.reference_trajectory(0)
        state = np.array([init[0], init[1], init[2]])
        
        # Control inputs
        def av_steering(t, state):
            # AV tries to follow reference trajectory precisely
            ref = self.reference_trajectory(t)
            cross_track = state[1] - ref[1]
            heading_error = state[2] - ref[2]
            return -0.1 * cross_track - 0.3 * heading_error
        
        def human_steering(t, state):
            # Human input (slightly oscillatory with delay in responding)
            ref = self.reference_trajectory(t)
            cross_track = state[1] - ref[1]
            heading_error = state[2] - ref[2]
            delay = 0.5  # simulated human delay
            return -0.08 * cross_track - 0.2 * heading_error + 0.01 * np.sin(2*t - delay)
        
        # Simulation loop
        for i, t in enumerate(self.t_eval[:-1]):
            dt = self.t_eval[i+1] - t
            
            # Get reference and controls
            ref = self.reference_trajectory(t)
            av_steer = av_steering(t, state)
            human_steer = human_steering(t, state)
            
            # Calculate transition weights
            av_auth, human_auth = self.transition_function(t)
            
            # Blend controls
            if t < self.t0:
                blended_steering = av_steer
                av_weight, human_weight = 1.0, 0.0
            elif t > self.tT:
                blended_steering = human_steer
                av_weight, human_weight = 0.0, 1.0
            else:
                blended_steering, (av_weight, human_weight) = self.controller.blend_control_inputs(
                    state, ref, av_steer, human_steer,
                    human_readiness=human_auth,
                    system_reliability=av_auth
                )
            
            # Store results
            self.results['state'].append(state.copy())
            self.results['steering'].append(blended_steering)
            self.results['av_weight'].append(av_weight)
            self.results['human_weight'].append(human_weight)
            self.results['tracking_error'].append(
                self.controller.calculate_tracking_error(state, ref)
            )
            self.results['safety_metric'].append(
                self.controller.calculate_control_safety(state, blended_steering)
            )
            self.results['time'].append(t)
            
            # Integrate dynamics
            # sol = solve_ivp(
            #     self.model.state_space, 
            #     [t, t+dt], 
            #     state, 
            #     args=(blended_steering,),
            #     method='RK45'
            # )
            # state = sol.y[:,-1]
            sol = np.array(self.model.state_space(t, state, blended_steering))
            state = state + sol * dt
        
        # Convert results to numpy arrays
        for key in self.results:
            self.results[key] = np.array(self.results[key])
    
    def plot_results(self):
        # Create subplots
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Trajectory plot
        ax1 = fig.add_subplot(gs[0, :])
        ref_y = [self.reference_trajectory(t)[1] for t in self.results['time']]
        ax1.plot(self.results['time'] * 20, ref_y, 'k--', label='Reference')
        ax1.plot(self.results['state'][:,0], self.results['state'][:,1], 'b-', label='Actual')
        ax1.axvline(x=self.t0*20, color='r', linestyle=':', label='Start Transition')
        ax1.axvline(x=self.tT*20, color='g', linestyle=':', label='End Transition')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Vehicle Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Control weights
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.results['time'], self.results['av_weight'], 'b-', label='AV Weight')
        ax2.plot(self.results['time'], self.results['human_weight'], 'r-', label='Human Weight')
        ax2.axvline(x=self.t0, color='r', linestyle=':')
        ax2.axvline(x=self.tT, color='g', linestyle=':')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Weight')
        ax2.set_title('Control Authority Distribution')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Steering angle
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.results['time'], np.rad2deg(self.results['steering']), 'g-')
        ax3.axvline(x=self.t0, color='r', linestyle=':')
        ax3.axvline(x=self.tT, color='g', linestyle=':')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Steering Angle (deg)')
        ax3.set_title('Blended Steering Input')
        ax3.grid(True)
        
        # 4. Performance metrics
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.results['time'], self.results['tracking_error'], 'b-', label='Tracking')
        ax4.plot(self.results['time'], self.results['safety_metric'], 'r-', label='Safety')
        ax4.axvline(x=self.t0, color='r', linestyle=':')
        ax4.axvline(x=self.tT, color='g', linestyle=':')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Performance Metric')
        ax4.set_title('Performance Metrics')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Heading angle
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(self.results['time'], np.rad2deg(self.results['state'][:,2]), 'g-')
        ax5.axvline(x=self.t0, color='r', linestyle=':')
        ax5.axvline(x=self.tT, color='g', linestyle=':')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Heading Angle (deg)')
        ax5.set_title('Vehicle Heading')
        ax5.grid(True)
        
        plt.tight_layout()
        plt.pause(0.5)
        return fig

# Run simulation and create visualization
scenario = TransitionScenario()
scenario.simulate()
scenario.plot_results()
plt.show()