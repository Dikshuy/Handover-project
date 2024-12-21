import numpy as np
import matplotlib.pyplot as plt

car_width = 1.8

fig, ax = plt.subplots(figsize=(12, 6))

x, y_top, y_bottom = [], [], []

section_2_width = (1.1 * car_width + 0.25 + 1) + (1) + (1 + car_width)
section_4_width = (1.3 * car_width + 0.25 + 1) + (1) + (1 + car_width)

sections = [
    {"length": 12, "width": 1.1 * car_width + 0.25},                       # Section 1
    {"length": 13.5, "width": section_2_width},                            # Section 2
    {"length": 11, "width": 1 + car_width},                                # Section 3
    {"length": 12.5, "width": section_4_width},                            # Section 4
    {"length": 12, "width": 1.3 * car_width + 0.25},                       # Section 5
]

x_start = 0
y_start = 0

# for section in sections:
#     x_start = draw_section(ax, x_start, y_start, section["length"], section["width"], section["lane_offset"])

x_end_1 = x_start + sections[0]["length"]
y_end_1_top = y_start + sections[0]["width"] / 2
y_end_1_bottom = y_start - sections[0]["width"] / 2

x_end_2 = x_end_1 + sections[1]["length"]
y_end_2_top = y_start + sections[1]["width"] - sections[0]["width"] / 2
y_end_2_bottom = y_end_1_bottom

x_end_3 = x_end_2 + sections[2]["length"]
y_end_3_top = y_end_2_top
y_end_3_bottom = y_end_2_top - sections[2]["width"]

x_end_4 = x_end_3 + sections[3]["length"]
y_end_4_top = y_end_3_top
y_end_4_bottom = y_end_3_top - sections[3]["width"]

x_end_5 = x_end_4 + sections[4]["length"]
y_end_5_top = y_end_4_top - sections[4]["width"]
y_end_5_bottom = y_end_4_top

x.extend([x_start, x_end_1, x_end_2, x_end_3, x_end_4, x_end_5])
y_top.extend([y_start, y_end_1_top, y_end_2_top, y_end_3_top, y_end_4_top, y_end_5_top])
y_bottom.extend([y_start, y_end_1_bottom, y_end_2_bottom, y_end_3_bottom, y_end_4_bottom, y_end_5_bottom])

ax.plot(x, y_top, 'k')
ax.plot(x, y_bottom, 'k')

ax.set_xlabel("Track Length (m)")
ax.set_ylabel("Track Width (m)")
ax.set_title("Obstacle Avoidance Track")
ax.grid(True)

plt.show()
