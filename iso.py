import numpy as np
import matplotlib.pyplot as plt

car_width = 1.8

def draw_section(ax, x_start, y_start, length, width, lane_offset=None):
    x_end = x_start + length
    y_top = y_start + width / 2
    y_bottom = y_start - width / 2

    # Plot the section borders
    ax.plot([x_start, x_end], [y_top, y_top], 'k')  # Top border
    ax.plot([x_start, x_end], [y_bottom, y_bottom], 'k')  # Bottom border

    # Draw lane offset if applicable
    if lane_offset:
        lane_center = y_start + lane_offset
        ax.plot([x_start, x_end], [lane_center, lane_center], 'r--')  # Lane offset (dashed line)

    return x_end

fig, ax = plt.subplots(figsize=(12, 6))

sections = [
    {"length": 12, "width": 1.1 * car_width + 0.25, "lane_offset": None},       # Section 1
    {"length": 13.5, "width": 6, "lane_offset": None},                          # Section 2
    {"length": 11, "width": 1 + car_width, "lane_offset": 1},                   # Section 3
    {"length": 12.5, "width": 6, "lane_offset": None},                          # Section 4
    {"length": 12, "width": 1.3 * car_width + 0.25, "lane_offset": None},       # Section 5
]

x_start = 0
y_start = 0

for section in sections:
    x_start = draw_section(ax, x_start, y_start, section["length"], section["width"], section["lane_offset"])

# ax.set_aspect('equal')
ax.set_xlabel("Track Length (m)")
ax.set_ylabel("Track Width (m)")
ax.set_title("Obstacle Avoidance Track")
ax.grid(True)

plt.show()
