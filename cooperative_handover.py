import numpy as np
from scipy.integrate import solve_ivp

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

def simulate_scenario():
    """
    Example simulation scenario
    """
    # Initialize model and controller
    L = 2.7  # wheelbase in meters
    model = KinematicBicycleModel(L)
    
    safety_params = {
        'max_steering': np.pi/4,  # 45 degrees
        'max_cross_track': 2.0,   # meters
        'max_heading_error': np.pi/4,  # 45 degrees
        'min_tracking': 0.3,
        'min_safety': 0.3
    }
    
    controller = SharedController(model, safety_params)
    
    # Simulation parameters
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    initial_state = [0, 0, 0]  # [x, y, theta]
    
    # Reference trajectory (simple straight line for example)
    reference = lambda t: [t*20, 0, 0]  # [x_ref, y_ref, theta_ref]
    
    # Example control inputs (could be more complex)
    av_steering = lambda t: 0.0
    human_steering = lambda t: 0.1 * np.sin(t)  # slight weaving
    human_readiness = lambda t: 0.5 + 0.5 * (1 - np.exp(-0.3*t))  # increasing readiness
    system_reliability = lambda t: 0.9  # constant high reliability
    
    return model, controller, t_span, t_eval, initial_state, reference, \
           av_steering, human_steering, human_readiness, system_reliability
