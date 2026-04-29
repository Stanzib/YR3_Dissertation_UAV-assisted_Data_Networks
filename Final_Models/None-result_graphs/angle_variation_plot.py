import numpy as np
import matplotlib.pyplot as plt

K_min = 1.0   # at 0° — tune this
K_max = 100.0  # at 90° — tune this

a = K_min
b = np.log(K_max / K_min) / (np.pi / 2)

theta = np.linspace(0, np.pi / 2, 500)
K = a * np.exp(b * theta)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
ax.plot(theta, K, color='purple', linewidth=2)

# Mark key angles
for angle, label in [(0, '0'), (0.698132, '40'), (1.0472, '60'),
                     (1.39626, '80'), (np.pi/2, '90')]:
    rv = a * np.exp(b * angle)
    ax.plot(angle, rv, 'o', color='coral', markersize=6)
    ax.annotate(f'K={rv:.2f}', xy=(angle, rv), fontsize=8,
                xytext=(angle + 0.05, rv * 1.05))

ax.set_thetamin(0)
ax.set_thetamax(90)           # restrict view to first quadrant
ax.set_title(r'Relationship of $k$ as $\theta$ increases', va='bottom', fontsize=13)
ax.set_xlabel(r'$k$')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(rf'C:\Users\stanl\OneDrive - The University of Manchester\uni\Y3\Individual Project\Results\angle_var_graph.png')
plt.show()