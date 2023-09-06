import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cone parameters
opening_angle = 60  # in degrees
inclination = 80  # in degrees

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate points for the cone
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, inclination * np.pi / 180, 50)
U, V = np.meshgrid(u, v)
R = V / (inclination * np.pi / 180)
X = R * np.cos(U)  # Adjusted X-coordinate
Y = R * np.sin(U)  # Adjusted Y-coordinate (pointing upwards)
Z = V * np.tan(opening_angle * np.pi / 360)

# Plot the cone
ax.plot_surface(X, Y, Z, color='b', alpha=0.5)

# Generate points for the plane at Y = 0
plane_x = np.linspace(-1.0, 1.0, 50)
plane_z = np.linspace(0, 1, 50)
plane_X, plane_Z = np.meshgrid(plane_x, plane_z)
plane_Y = np.zeros_like(plane_X)

# Plot the plane at Y = 0
ax.plot_surface(plane_X, plane_Y, plane_Z, color='g', alpha=0.5)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot title
ax.set_title('3D Cone with Plane at Y = 0')

# Set equal aspect ratio for proper visualization
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.show()
