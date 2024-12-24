import numpy as np
import matplotlib.pyplot as plt


def bezier_curve(p0, p1, p2, p3, num_points=50):
    """Generate points along a cubic Bézier curve."""
    t = np.linspace(0, 1, num_points)
    curve_x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + \
              3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    curve_y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + \
              3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
              
    return curve_x, curve_y

def calculate_theta(x, y):
    """Calculate heading angle at each point using local tangent"""
    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)
    return theta

def equally_space_points(x, y, num_points):
    """Resample points to be equally spaced along the curve"""
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    
    total_distance = cumulative_distances[-1]
    equal_distances = np.linspace(0, total_distance, num_points)
    
    new_x = np.interp(equal_distances, cumulative_distances, x)
    new_y = np.interp(equal_distances, cumulative_distances, y)
    
    return new_x, new_y

def trajectory(car_width=1.8):
    """Generates a trajectory for a car to follow through an obstacle avoidance track"""    
    section_2_width = (1.1 * car_width + 0.25 + 1) + (1) + (1 + car_width)
    section_4_width = (1.3 * car_width + 0.25 + 1) + (1) + (1 + car_width)
    
    sections = [
        {"length": 12, "width": 1.1 * car_width + 0.25},    # Section 1
        {"length": 13.5, "width": section_2_width},         # Section 2
        {"length": 11, "width": 1 + car_width},             # Section 3
        {"length": 12.5, "width": section_4_width},         # Section 4
        {"length": 12, "width": 1.3 * car_width + 0.25},    # Section 5
    ]
    
    x_coords_top = []
    y_coords_top = []
    x_coords_bottom = []
    y_coords_bottom = []
    x_coords_middle = []
    y_coords_middle = []
    theta_middle = []
    
    points_per_meter = 4
    
    current_x = 0
    base_y = -(1.1 * car_width + 0.25) / 2
    
    # Section 1 (straight)
    width = sections[0]["width"]
    length = sections[0]["length"]
    num_points = int(length * points_per_meter)
    x = np.linspace(current_x, current_x + length, num_points)
    y = np.full_like(x, base_y + width/2)
    theta = np.zeros_like(x)
    
    x_coords_middle = np.concatenate([x_coords_middle, x])
    y_coords_middle = np.concatenate([y_coords_middle, y])
    theta_middle = np.concatenate([theta_middle, theta])
    
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])

    s1_end_x = current_x + length
    s1_end_y = base_y + width/2
    
    current_x += length
    
    # Section 2 (transition up with Bézier curve)
    length = sections[1]["length"]
    width = sections[1]["width"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])
    
    s3_start_x = current_x + length
    s3_start_y = base_y + width - (car_width + 1)/2

    p0 = (s1_end_x, s1_end_y)
    p3 = (s3_start_x, s3_start_y)
    p1 = (s1_end_x + length/3, s1_end_y)
    p2 = (s3_start_x - length/3, s3_start_y)
    
    curve_x, curve_y = bezier_curve(p0, p1, p2, p3, num_points=100)
    curve_x, curve_y = equally_space_points(curve_x, curve_y, int(length * points_per_meter))
    curve_theta = calculate_theta(curve_x, curve_y)
    
    x_coords_middle = np.concatenate([x_coords_middle, curve_x])
    y_coords_middle = np.concatenate([y_coords_middle, curve_y])
    theta_middle = np.concatenate([theta_middle, curve_theta])
    
    current_x += length
    
    # Section 3 (narrow section with offset)
    length = sections[2]["length"]
    num_points = int(length * points_per_meter)
    x = np.linspace(current_x, current_x + length, num_points)
    y = np.full_like(x, s3_start_y)
    theta = np.zeros_like(x)
    
    x_coords_middle = np.concatenate([x_coords_middle, x])
    y_coords_middle = np.concatenate([y_coords_middle, y])
    theta_middle = np.concatenate([theta_middle, theta])
    
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y + width - (car_width + 1), base_y + width - (car_width + 1)])
    
    s3_end_x = current_x + length
    s3_end_y = s3_start_y
    
    current_x += length
    
    # Section 4 (transition down with Bézier curve)
    length = sections[3]["length"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])

    s5_start_x = current_x + length
    s5_start_y = base_y + sections[4]["width"] / 2

    p0 = (s3_end_x, s3_end_y)
    p3 = (s5_start_x, s5_start_y)
    p1 = (s3_end_x + length/3, s3_end_y)
    p2 = (s5_start_x - length/3, s5_start_y)
    
    curve_x, curve_y = bezier_curve(p0, p1, p2, p3, num_points=100)
    curve_x, curve_y = equally_space_points(curve_x, curve_y, int(length * points_per_meter))
    curve_theta = calculate_theta(curve_x, curve_y)
    
    x_coords_middle = np.concatenate([x_coords_middle, curve_x])
    y_coords_middle = np.concatenate([y_coords_middle, curve_y])
    theta_middle = np.concatenate([theta_middle, curve_theta])
    
    current_x += length
    
    # Section 5 (final straight)
    length = sections[4]["length"]
    width = sections[4]["width"]
    num_points = int(length * points_per_meter)
    x = np.linspace(current_x, current_x + length, num_points)
    y = np.full_like(x, s5_start_y)
    theta = np.zeros_like(x)
    
    x_coords_middle = np.concatenate([x_coords_middle, x])
    y_coords_middle = np.concatenate([y_coords_middle, y])
    theta_middle = np.concatenate([theta_middle, theta])
    
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])

    # fig, ax = plt.subplots(figsize=(16, 8))
    
    # ax.plot(x_coords_top, y_coords_top, 'k-', linewidth=2, label='Track Boundary')
    # ax.plot(x_coords_bottom, y_coords_bottom, 'k-', linewidth=2)
    # ax.plot(x_coords_middle, y_coords_middle, 'r--', linewidth=2, label='Middle Trajectory')
    
    # # Visualize heading angles with arrows
    # n = 8
    # arrow_length = 0.3
    # for i in range(0, len(x_coords_middle), n):
    #     ax.arrow(x_coords_middle[i], y_coords_middle[i],
    #             arrow_length * np.cos(theta_middle[i]),
    #             arrow_length * np.sin(theta_middle[i]),
    #             head_width=0.1, head_length=0.2, fc='blue', ec='blue', alpha=0.5)
    
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.set_xlabel("Track Length (m)")
    # ax.set_ylabel("Track Width (m)")
    # ax.set_title("Obstacle Avoidance Track")
    # ax.set_aspect('equal')
    # ax.legend()
    
    trajectory_coords = {
        'x': x_coords_middle,
        'y': y_coords_middle,
        'theta': theta_middle
    }
    
    return trajectory_coords

car_width = 1.8
waypoints = trajectory(car_width)
# plt.show()