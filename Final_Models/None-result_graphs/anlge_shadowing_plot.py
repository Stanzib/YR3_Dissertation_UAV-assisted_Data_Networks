import numpy as np
import matplotlib.pyplot as plt


a = 11
b = 3

theta = np.linspace(0, np.pi / 2, 500)
K = a - ((a-b)* theta) / (np.pi/2)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
ax.plot(theta, K, color='purple', linewidth=2)

# Mark key angles
for angle, label in [(0, '0'), (0.698132, '40'), (1.0472, '60'),
                     (1.39626, '80'), (np.pi/2, '90')]:
    rv = a - (((a-b)* angle) / (np.pi/2))
    ax.plot(angle, rv, 'o', color='coral', markersize=6)
    ax.annotate(f'K={rv:.2f}', xy=(angle, rv), fontsize=8,
                xytext=(angle + 0.05, rv * 1.05))

ax.set_thetamin(0)
ax.set_thetamax(90)           # restrict view to first quadrant
ax.set_title(r'Relationship of $\sigma_{dB}$ as $\theta$ increases', va='bottom', fontsize=13)
ax.set_xlabel(r'$\sigma_{dB}$')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()