import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp
import math


#defining host
x_host = 20
y_host = 15

# defining the paremiters of the antenna
#coordiantes
x_a1 = 15
y_a1 = 15

x_a2 = 30
y_a2 = 20

x_a3 = 50
y_a3 = 50

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

x_a1_cirlce, y_a1_circle = generate_circle_points(x_a1, y_a1, reuse_distance)
x_a2_cirlce, y_a2_circle = generate_circle_points(x_a2, y_a2, reuse_distance)
x_a3_cirlce, y_a3_circle = generate_circle_points(x_a3, y_a3, reuse_distance)

#plt.figure(figsize=(8,8))
#plt.plot(x_a1_cirlce, y_a1_circle , markersize=8, label='A1 reuse')
#plt.plot(x_a1, y_a1, 'ro', markersize=8, label='Antenna 1')
#plt.plot(x_a2_cirlce, y_a2_circle , markersize=8, label='A2 reuse')
#plt.plot(x_a2, y_a2, 'ro', markersize=8, label='Antenna 2')
#plt.plot(x_a3_cirlce, y_a3_circle , markersize=8, label='A3 reuse')
#plt.plot(x_a3, y_a3, 'ro', markersize=8, label='Antenna 3')
#plt.legend()
#plt.title("map")

#plt.plot(x_host, y_host, 'ro', markersize=8, label='Host')

#plt.show()


CONST_pathloss_exp = 2.6
CONST_wavelength = 100

def path_loss(d, n, wavelength):
    """"
    Calculate path loss using the formula:
    Pl = 10 * log10(16 * π² * dⁿ / λ²)
    
    Parameters:
    d: distance
    n: path loss exponent
    wavelength: wavelength
    
    Returns:
    Path loss in dB
    """
    return 10 * math.log10((16 * math.pi**2 * d**n) / wavelength**2)

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


#print(power(x_a1, y_a1, 100, x_host, y_host))

u = []

print("hello world")

for i in range(0, 40):
    if i != x_a1:
        result = power(x_a1, y_a1, 100, i, y_a1)
        u.append(result)
    else:
        u.append(0)

x_power_plot = np.linspace(0,40,40)

plt.figure(figsize=(8,8))
plt.plot(x_power_plot, u, label = 'power over distance')
plt.show()


