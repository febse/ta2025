import matplotlib.pyplot as plt
import numpy as np

# Define triangle vertices
# A at origin, B along x-axis, C at angle theta
A = np.array([0, 0])
B = np.array([4, 0])  # side c (length 4)
theta = np.pi / 4  # 45 degrees
C = np.array([3 * np.cos(theta), 3 * np.sin(theta)])  # side b (length 3)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Draw triangle
triangle = plt.Polygon([A, B, C], fill=False, edgecolor='blue', linewidth=2)
ax.add_patch(triangle)

# Plot vertices
ax.plot(*A, 'ro', markersize=8)
ax.plot(*B, 'ro', markersize=8)
ax.plot(*C, 'ro', markersize=8)

# Label vertices
ax.text(A[0]-0.3, A[1]-0.3, 'A', fontsize=14, fontweight='bold')
ax.text(B[0]+0.2, B[1]-0.3, 'B', fontsize=14, fontweight='bold')
ax.text(C[0]-0.3, C[1]+0.3, 'C', fontsize=14, fontweight='bold')

# Label sides
# Side a (opposite to A, between B and C)
mid_a = (B + C) / 2
ax.text(mid_a[0]+0.2, mid_a[1]+0.2, 'a', fontsize=14, color='blue', fontweight='bold')

# Side b (opposite to B, between A and C)
mid_b = (A + C) / 2
ax.text(mid_b[0]-0.4, mid_b[1], 'b', fontsize=14, color='blue', fontweight='bold')

# Side c (opposite to C, between A and B)
mid_c = (A + B) / 2
ax.text(mid_c[0], mid_c[1]-0.4, 'c', fontsize=14, color='blue', fontweight='bold')

# Draw and label angle theta at vertex A
angle_arc = plt.Circle(C, 0.2, fill=False, color='red', linewidth=1.5)
ax.add_patch(angle_arc)
theta_arc = np.linspace(0, theta, 50)

# ax.plot(arc_x, arc_y, 'r-', linewidth=2)
ax.text(C[0] - 0.1, C[1] - 0.4, r'$\theta$', fontsize=14, color='red', fontweight='bold')

# Set axis properties
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(r'Triangle with sides a, b, c and angle $\theta$ between a and b', fontsize=14)

plt.tight_layout()
# plt.show()
