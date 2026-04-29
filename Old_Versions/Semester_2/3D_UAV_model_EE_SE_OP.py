import numpy as np
import matplotlib.pyplot as plt

# ── Positions ─────────────────────────────────────────────────────────────────
BS_Pos = [10, 10, 0]       # Serving BS (x, y, z)
Host_centre = [10, 10, 0]  # Centre of host cell

# ── Channel Parameters ────────────────────────────────────────────────────────
alpha_urban  = 3.8    # BS → UAV: dense urban, high path loss
alpha_a2g    = 2.0    # UAV → Host: air-to-ground, near free space
K_factor     = 5.0    # Rician K for UAV→Host (LOS dominant)
R_max        = 500    # Cell radius (metres)
UAV_altitude = 50     # UAV height (metres)

# ── Power Parameters ─────────────────────────────────────────────────────────
P_BS         = 10.0   # BS transmit power (Watts)
P_UAV_tx     = 2.0    # UAV transmit power (Watts)
P_UAV_flight = 150.0  # UAV mechanical flight power (Watts)
Noise_power  = 1e-3   # Thermal noise (Watts)

# ── Thresholds ────────────────────────────────────────────────────────────────
gamma_th     = 1.0    # SINR threshold (= 2^R - 1, R=1 bit/s/Hz)

# ── Simulation ────────────────────────────────────────────────────────────────
num_iterations = 10_000
num_UAVs_max   = 50
relay_mode     = 'DF'   # 'DF' or 'AF'

# ── Results ───────────────────────────────────────────────────────────────────
Outage_list = []
SE_list     = []
EE_list     = []


def rician_gain(K, size=1):
    s     = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))
    h_r   = s + sigma * np.random.randn(size)
    h_i   =     sigma * np.random.randn(size)
    return h_r**2 + h_i**2


def rayleigh_gain(size=1):
    return np.random.exponential(1.0, size)


def path_loss(d, alpha):
    return d ** (-alpha)


def e2e_sinr_DF(sinr1, sinr2):
    # DF: bottleneck is the weaker hop
    return min(sinr1, sinr2)


def e2e_sinr_AF(sinr1, sinr2):
    # AF: harmonic-like combination
    return (sinr1 * sinr2) / (sinr1 + sinr2 + 1)


# ── Main Loop ─────────────────────────────────────────────────────────────────
for num_uav in range(1, num_UAVs_max + 1):

    print(f"UAVs: {num_uav}")

    # Deploy UAVs uniformly within cell at fixed altitude
    uav_positions = []
    for _ in range(num_uav):
        r     = R_max * np.sqrt(np.random.uniform(0.05, 1))
        theta = 2 * np.pi * np.random.uniform()
        x     = Host_centre[0] + r * np.cos(theta)
        y     = Host_centre[1] + r * np.sin(theta)
        uav_positions.append([x, y, UAV_altitude])

    outage_count = 0
    SE_sum       = 0.0
    EE_sum       = 0.0

    for _ in range(num_iterations):

        # ── Random host position ──────────────────────────────────────────────
        r     = R_max * np.sqrt(np.random.uniform(0.05, 1))
        theta = 2 * np.pi * np.random.uniform()
        hx    = Host_centre[0] + r * np.cos(theta)
        hy    = Host_centre[1] + r * np.sin(theta)
        host  = [hx, hy, 0]

        # ── Find closest UAV to host ──────────────────────────────────────────
        uav_dist_to_host = []
        for uav in uav_positions:
            d = np.sqrt((host[0]-uav[0])**2 +
                        (host[1]-uav[1])**2 +
                        (host[2]-uav[2])**2)
            uav_dist_to_host.append(d)

        best_uav_idx = int(np.argmin(uav_dist_to_host))
        best_uav     = uav_positions[best_uav_idx]
        d_uav_host   = uav_dist_to_host[best_uav_idx]

        # ── Hop 1: BS → serving UAV (urban path loss, Rayleigh) ───────────────
        d_bs_uav = np.sqrt((BS_Pos[0]-best_uav[0])**2 +
                           (BS_Pos[1]-best_uav[1])**2 +
                           (BS_Pos[2]-best_uav[2])**2)

        h1       = rayleigh_gain()[0]
        signal1  = P_BS * h1 * path_loss(d_bs_uav, alpha_urban)

        # Interference on hop 1: other UAVs also receiving from BS (co-channel)
        interf1 = 0.0
        for k, uav in enumerate(uav_positions):
            if k != best_uav_idx:
                d_bs_k   = np.sqrt((BS_Pos[0]-uav[0])**2 +
                                   (BS_Pos[1]-uav[1])**2 +
                                   (BS_Pos[2]-uav[2])**2)
                h_k      = rayleigh_gain()[0]
                interf1 += P_BS * h_k * path_loss(d_bs_k, alpha_urban)

        SINR1 = signal1 / (interf1 + Noise_power)

        # ── Hop 2: serving UAV → Host (A2G, Rician) ──────────────────────────
        h2      = rician_gain(K_factor)[0]
        signal2 = P_UAV_tx * h2 * path_loss(d_uav_host, alpha_a2g)

        # Interference on hop 2: other UAVs transmitting to their own hosts
        interf2 = 0.0
        for k, uav in enumerate(uav_positions):
            if k != best_uav_idx:
                d_k      = np.sqrt((host[0]-uav[0])**2 +
                                   (host[1]-uav[1])**2 +
                                   (host[2]-uav[2])**2)
                h_k      = rician_gain(K_factor)[0]
                interf2 += P_UAV_tx * h_k * path_loss(d_k, alpha_a2g)

        SINR2 = signal2 / (interf2 + Noise_power)

        # ── End-to-end SINR ───────────────────────────────────────────────────
        if relay_mode == 'DF':
            SINR_e2e = e2e_sinr_DF(SINR1, SINR2)
        else:
            SINR_e2e = e2e_sinr_AF(SINR1, SINR2)

        # ── Spectral Efficiency (half-duplex penalty) ─────────────────────────
        SE = 0.5 * np.log2(1 + SINR_e2e)

        # ── Energy Efficiency ─────────────────────────────────────────────────
        P_total = P_BS + num_uav * (P_UAV_tx + P_UAV_flight)
        EE      = SE / P_total   # bits/s/Hz per Watt

        # ── Outage ────────────────────────────────────────────────────────────
        if SINR_e2e < gamma_th:
            outage_count += 1

        SE_sum += SE
        EE_sum += EE

    Outage_list.append(outage_count / num_iterations)
    SE_list.append(SE_sum / num_iterations)
    EE_list.append(EE_sum / num_iterations)


# ── Plotting ──────────────────────────────────────────────────────────────────
uav_range = range(1, num_UAVs_max + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'UAV Relay Performance — Urban (α={alpha_urban}), {relay_mode} Relay', fontsize=13)

axes[0].plot(uav_range, Outage_list, 'b-o', markersize=3)
axes[0].set_xlabel('Number of UAVs'); axes[0].set_ylabel('Outage Probability')
axes[0].set_title('Outage Probability'); axes[0].grid(True)

axes[1].plot(uav_range, SE_list, 'g-o', markersize=3)
axes[1].set_xlabel('Number of UAVs'); axes[1].set_ylabel('Spectral Efficiency (bits/s/Hz)')
axes[1].set_title('Spectral Efficiency'); axes[1].grid(True)

axes[2].plot(uav_range, EE_list, 'r-o', markersize=3)
axes[2].set_xlabel('Number of UAVs'); axes[2].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
axes[2].set_title('Energy Efficiency'); axes[2].grid(True)

plt.tight_layout()
plt.show()