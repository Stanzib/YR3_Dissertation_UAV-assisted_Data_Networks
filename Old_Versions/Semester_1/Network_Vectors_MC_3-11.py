import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp
import math
from mpl_toolkits.mplot3d import axes3d


#defining host
host_pos = [20, 15]

# defining the paremiters of the antenna
#coordiantes
a1_pos = [15, 15]

a2_pos =[20,30]

a3_pos = [50,50]

#reuse distance
reuse_distance = 15

def generate_circle_points(h, k, r, num_points=100):
    """
    Generate points on a circle
    (h, k) is the centre of the cirlce
    r is the radius
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = h + r * np.cos(theta)
    y = k + r * np.sin(theta)
    return x, y

# map of antenna and reuse distance

x_a1_cirlce, y_a1_circle = generate_circle_points(a1_pos[0], a1_pos[1], reuse_distance)
x_a2_cirlce, y_a2_circle = generate_circle_points(a2_pos[0], a2_pos[1], reuse_distance)
x_a3_cirlce, y_a3_circle = generate_circle_points(a3_pos[0], a3_pos[1], reuse_distance)

plt.figure(figsize=(8,8))
plt.plot(x_a1_cirlce, y_a1_circle , markersize=8, label='A1 reuse')
plt.plot(a1_pos[0], a1_pos[1], 'ro', markersize=8, label='Antenna 1')
plt.plot(x_a2_cirlce, y_a2_circle , markersize=8, label='A2 reuse')
plt.plot(a2_pos[0], a2_pos[1], 'ro', markersize=8, label='Antenna 2')
plt.plot(x_a3_cirlce, y_a3_circle , markersize=8, label='A3 reuse')
plt.plot(a3_pos[0], a3_pos[1], 'ro', markersize=8, label='Antenna 3')
plt.legend()
plt.title("map")

plt.plot(host_pos[0], host_pos[1], 'ro', markersize=8, label='Host')

#plt.show()


CONST_pathloss_exp = 2.6
CONST_wavelength = 0.0757
CONST_power = 100

def path_loss(d, n, wavelength):
    """"
    Calculate path loss using the formula:
    Pl = 10 * log10(16 * π² * dⁿ / λ²)

    Path loss in a linear scale:
    Pl = (16 * π² * dⁿ) / λ²
    
    Parameters:
    d: distance
    n: path loss exponent
    wavelength: wavelength
    
    Returns:
    Path loss (linear)
    """
    if d <= 0:
        d = 0.0001
    if wavelength <= 0:
        wavelength = 0.0001

    return (16 * math.pi**2 * d**n) / wavelength**2

def fading(d, wavelength):
    """
    Parameters:
    d: distance
    wavelength: wavelength

    Returns fading multiplier
    """


def power(x_antenna, y_antenna, power_transmitted, x_host, y_host):

    distance = np.sqrt((x_host - x_antenna)**2 + (y_host - y_antenna)**2)

    return power_transmitted * path_loss(distance, CONST_pathloss_exp, CONST_wavelength)


# Power recived (numerator)

power_s = []  

for i in range(40):
    row = []  # Create a new row for each i
    for j in range(40):
        result = power(a1_pos[0], a1_pos[1], CONST_power, i, j)
        row.append(result)  # Append to the current row
    power_s.append(row)  # Append the completed row to u


# Power of interference (denominator)

power_i = []  

for i in range(40):
    row = []  # Create a new row for each i
    for j in range(40):
        result = power(a2_pos[0], a2_pos[1], CONST_power, i, j)
        result += power(a3_pos[0], a3_pos[1], CONST_power, i, j)
        row.append(result)  # Append to the current row
    power_i.append(row)  # Append the completed row to u



    
# displaying power function of a1 @ y= 15 
'''
x_power_plot = np.linspace(0,40,40) - a1_pos[1]

plt.figure(figsize=(8,8))
plt.plot(x_power_plot, u, label = 'power over distance')
plt.title('power recived from Antenna 1 along y=15')
plt.xlabel('distance from antenna')
plt.ylabel('power')
plt.show()
'''


# Convert to NumPy array for plotting
power_s_array = np.array(power_s)

# Create coordinate arrays
X, Y = np.meshgrid(range(40), range(40))

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
surf = ax.plot_surface(X, Y, power_s_array, 
                      cmap='viridis',      # Color map
                      edgecolor='black',   # Edge color
                      linewidth=0.5,       # Edge line width
                      alpha=0.8,           # Transparency
                      rstride=2,           # Reduce sampling for better performance
                      cstride=2)

# Add contour projections
ax.contour(X, Y, power_s_array, zdir='z', offset=np.min(power_s_array), cmap='coolwarm')
ax.contour(X, Y, power_s_array, zdir='x', offset=0, cmap='coolwarm')
ax.contour(X, Y, power_s_array, zdir='y', offset=39, cmap='coolwarm')

# Add labels and title
ax.set(xlabel='X Position', 
       ylabel='Y Position', 
       zlabel='Received Power',
       title='3D Power Distribution from Antenna')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Power')

plt.show()
