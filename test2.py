import numpy as np
import matplotlib.pyplot as plt

def path_with_trajectory(car_width=1.8):
    fig, ax = plt.subplots(figsize=(16, 8))
    
    section_2_width = (1.1 * car_width + 0.25 + 1) + (1) + (1 + car_width)
    section_4_width = (1.3 * car_width + 0.25 + 1) + (1) + (1 + car_width)
    
    sections = [
        {"length": 12, "width": 1.1 * car_width + 0.25},  # Section 1
        {"length": 13.5, "width": section_2_width},  # Section 2
        {"length": 11, "width": 1 + car_width},  # Section 3
        {"length": 12.5, "width": section_4_width},  # Section 4
        {"length": 12, "width": 1.3 * car_width + 0.25},  # Section 5
    ]
    
    x_coords_top = []
    y_coords_top = []
    x_coords_bottom = []
    y_coords_bottom = []
    x_coords_middle = []
    y_coords_middle = []
    
    current_x = 0
    base_y = -(1.1 * car_width + 0.25) / 2
    
    # Section 1 (straight)
    width = sections[0]["width"]
    length = sections[0]["length"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])
    # Middle points for section 1
    x_coords_middle.extend([current_x, current_x + length + sections[1]["length"] /2])
    y_coords_middle.extend([base_y + width/2, base_y + width/2])
    
    current_x += length
    
    # Section 2 (transition up)
    length = sections[1]["length"]
    width = sections[1]["width"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])
    # Middle points for section 2
    x_coords_middle.extend([current_x + length/2, current_x + length/2])
    y_coords_middle.extend([base_y + width/2, base_y + width/2])
    
    current_x += length
    
    # Section 3 (narrow section with offset)
    length = sections[2]["length"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y + width - (car_width + 1), base_y + width - (car_width + 1)])
    # Middle points for section 3
    x_coords_middle.extend([current_x, current_x + length + sections[3]["length"] / 2])
    middle_y = (base_y + width + (base_y + width - (car_width + 1))) / 2
    y_coords_middle.extend([middle_y, middle_y])
    
    current_x += length
    
    # Section 4 (transition down)
    length = sections[3]["length"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])
    # Middle points for section 4
    x_coords_middle.extend([current_x + length/2, current_x + length/2])
    y_coords_middle.extend([base_y + width/2, base_y + width/2])
    
    current_x += length
    
    # Section 5 (final straight)
    length = sections[4]["length"]
    width = sections[4]["width"]
    x_coords_top.extend([current_x, current_x + length])
    y_coords_top.extend([base_y + width, base_y + width])
    x_coords_bottom.extend([current_x, current_x + length])
    y_coords_bottom.extend([base_y, base_y])
    # Middle points for section 5
    x_coords_middle.extend([current_x, current_x + length])
    y_coords_middle.extend([base_y + width/2, base_y + width/2])
    
    # Plot track boundaries
    ax.plot(x_coords_top, y_coords_top, 'k-', linewidth=2, label='Track Boundary')
    ax.plot(x_coords_bottom, y_coords_bottom, 'k-', linewidth=2)
    
    # Plot middle trajectory
    ax.plot(x_coords_middle, y_coords_middle, 'r--', linewidth=2, label='Middle Trajectory')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Track Length (m)")
    ax.set_ylabel("Track Width (m)")
    ax.set_title("Obstacle Avoidance Track with Middle Trajectory")
    ax.set_aspect('equal')
    ax.legend()
    
    # Return coordinates for further use if needed
    trajectory_coords = {
        'x': x_coords_middle,
        'y': y_coords_middle
    }
    
    return fig, ax, trajectory_coords

# Create and display the plot
car_width = 1.8
fig, ax, trajectory = path_with_trajectory(car_width)
plt.show()