# 3D model a host within an non-opertational Base Stations (BS) range
# in a setting with average conditions
# We increase the amount of UAVs within the non-opertational BS reuse distance
# Uplink
# UAVs relay signal to the closest operational BS
# other 5 neighbouring BS act as interferers
# best case worst case calculated

print("script started: Increasing UAV, Disaster Scenario - Central BS Non-Operational")

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from dataclasses import dataclass, field

# ------- Models Map ----------------------------------------
Dist_between_BS = 2500
serving_x = 5000
serving_y = 5000
BS_Height = 25

# Hexagon: 6 neighbouring BSs (index 0-5), central BS (index 6) is NON-OPERATIONAL
# [dc, 0] = x, [dc, 1] = y, [dc, 2] = z
BS_Pos = [
    [serving_x,                                    serving_y + Dist_between_BS,         BS_Height],  # BS0
    [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,     BS_Height],  # BS1
    [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,     BS_Height],  # BS2
    [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,     BS_Height],  # BS3
    [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,     BS_Height],  # BS4
    [serving_x,                                    serving_y - Dist_between_BS,         BS_Height],  # BS5
    [serving_x,                                    serving_y,                           BS_Height],  # BS6 = CENTRAL (NON-OPERATIONAL)
]

CENTRAL_BS_IDX   = 6          # non-operational — never used
NEIGHBOUR_BS_IDX = list(range(6))  

R_max = Dist_between_BS * 0.5
Dist_to_worst_case = Dist_between_BS - R_max   
Dist_to_Best_case  = Dist_between_BS + R_max   

# ------- Performance Metric variables ------------------------
P_tx_UAV         = 5
P_circuit_UAV    = 10
P_circuit_host   = 5
R_0              = 20
P_UAV_flight     = 150
Outage_threshold = 1
relay_mode       = 'AF'   # 'DF' or 'AF'

# --------- Scenario / channel parameters --------------------
PathLossExponent_urban = 3.25
PathLossExponent_UAV   = 2.5
shadowing_angle_cutoff = 45
UAV_Height = 100
#num_UAV = 100
P_tx_Host = 0.2

# ---- Result lists ------------------------------------------
SINR_list     = []; Capacity_list     = []; BER_list     = []; Outage_list     = []; Energy_Eff_list     = []
SINR_list_wor = []; Capacity_list_wor = []; BER_list_wor = []; Outage_list_wor = []; Energy_Eff_list_wor = []
SINR_list_bes = []; Capacity_list_bes = []; BER_list_bes = []; Outage_list_bes = []; Energy_Eff_list_bes = []
SINR_list_baseline = []


# ------- Helper functions ------------------------------------
@dataclass
class PerfAccumulator:
    SINR_sum:       float = 0.0
    Capacity_sum:   float = 0.0
    BER_sum:        float = 0.0
    outage_sum:     float = 0.0
    Energy_Eff_sum: float = 0.0

    def reset(self):
        self.SINR_sum = self.Capacity_sum = self.BER_sum = self.outage_sum = self.Energy_Eff_sum = 0.0

def rician_gain(K, size=1):
    s     = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))
    h_r   = s + sigma * np.random.randn(size)
    h_i   =     sigma * np.random.randn(size)
    return h_r**2 + h_i**2

def rayleigh_gain(size=1):
    return np.random.exponential(1.0, size)

def path_loss(d, alpha):
    d = max(d, 0.1)
    return d ** (-alpha)

def random_pos(center_x, center_y, is_UAV):
    theta = 2 * np.pi * np.random.uniform()
    r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform(0, 1))
    z     = UAV_Height if is_UAV else 0
    return [center_x + r * np.cos(theta),
            center_y + r * np.sin(theta),
            z]

def Noise():
    k   = 1.38e-23
    T   = 290
    B   = 1
    return k * T * B

def e2e_sinr(sinr1, sinr2, mode='DF'):
    if mode == 'DF':
        return min(sinr1, sinr2)
    else:  # AF
        return (sinr1 * sinr2) / (sinr1 + sinr2 + 1)

def performance_metrics(sinr, is_uav, acc: PerfAccumulator, P_total_UAV=None):
    BER = 0.5 * special.erfc(np.sqrt(max(sinr, 0)))
    if is_uav:
        Capacity   = 0.5 * np.log2(1 + sinr)
        Energy_Eff = Capacity / P_total_UAV
    else:
        Capacity   = np.log2(1 + sinr)
        Energy_Eff = Capacity / (P_circuit_host + P_tx_Host)
    if sinr < Outage_threshold:
        acc.outage_sum += 1
    acc.SINR_sum       += sinr
    acc.Capacity_sum   += Capacity
    acc.Energy_Eff_sum += Energy_Eff
    acc.BER_sum        += BER

def shadowing_gain(angle_deg=None):
    if angle_deg is None:
        sigma_dB = 12
    elif angle_deg > shadowing_angle_cutoff:
        return 1
    else:
        sigma_dB = 12.0 - (8.0 * (angle_deg / 90.0))
        sigma_dB = np.clip(sigma_dB, 4.0, 12.0)
    X_sigma_dB = np.random.normal(0, sigma_dB)
    return min(10 ** (X_sigma_dB / 10), 1)

def elevation_angle(uav_pos, ground_pos):
    dx = uav_pos[0] - ground_pos[0]
    dy = uav_pos[1] - ground_pos[1]
    dz = uav_pos[2] - ground_pos[2]
    horiz = np.sqrt(dx**2 + dy**2)
    return np.degrees(np.arctan2(dz, horiz))

def K_factor_elevation(angle_deg):
    K_min = 1.0
    K_max = 100.0
    b     = np.log(K_max / K_min) / (np.pi / 2)
    return K_min * np.exp(b * angle_deg)

def dist3d(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def dist2d(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def nearest_bs(host_pos, candidate_indices):
    # returns closest BS postition in BS_pos list
    dists = [dist2d(host_pos, BS_Pos[i]) for i in candidate_indices]
    return candidate_indices[int(np.argmin(dists))]

def build_fixed_interf_ring(serving_bs_idx, case='worst'):
    # returns the worst and best case interferer positions depending on the serving BS
    serving_BS = BS_Pos[serving_bs_idx]
    sx, sy = serving_BS[0], serving_BS[1]

    other_bs_indices = [i for i in NEIGHBOUR_BS_IDX if i != serving_bs_idx]

    result = []
    for i in other_bs_indices:
        bx, by = BS_Pos[i][0], BS_Pos[i][1]

        # Unit vector from this BS toward the serving BS
        dx, dy = sx - bx, sy - by
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / dist, dy / dist

        if case == 'worst':
            # Host on the side facing the serving BS — closest approach
            result.append([bx + R_max * ux, by + R_max * uy, 0])
        else:
            # Host on the opposite side — furthest from serving BS
            result.append([bx - R_max * ux, by - R_max * uy, 0])

    return result

# =========== Monte Carlo ==========================================
num_UAVs_total = 150
num_iterations = 100   # increase to 10_000 for final results

acc_gen      = PerfAccumulator()
acc_wor      = PerfAccumulator()
acc_bes      = PerfAccumulator()
acc_baseline = PerfAccumulator()

sinr_baseline_total = 0

for num_UAV in range(1, num_UAVs_total + 1):

    acc_gen.reset(); acc_wor.reset(); acc_bes.reset(); acc_baseline.reset()
    print(f"Starting UAV count: {num_UAV}")

    for j in range(num_iterations):

        # ----- Spawn UAVs around the (dead) central cell centre-----------
        UAV_pos_list = [random_pos(serving_x, serving_y, True) for _ in range(num_UAV)]

        #-- Random host position -----------------------------------
        Host_Pos_cart = random_pos(serving_x, serving_y, False)

        #===== disaster scenario ===========
        # -- Select serving BS: nearest neighbour to host --------
        serving_bs_idx = nearest_bs(Host_Pos_cart, NEIGHBOUR_BS_IDX)
        serving_BS     = BS_Pos[serving_bs_idx]

        # -- Interfering BSs: the other 5 operational neighbours ------------
        interf_bs_indices = [i for i in NEIGHBOUR_BS_IDX if i != serving_bs_idx]  

        # -- One random interfering host per interfering BS ----------------------
        interf_rand_pos_list = [random_pos(BS_Pos[i][0], BS_Pos[i][1], False)
                                for i in interf_bs_indices]



        # -- Best UAV: closest to host --------------------------
        dists    = [dist3d(Host_Pos_cart, u) for u in UAV_pos_list]
        best     = int(np.argmin(dists))
        best_uav = UAV_pos_list[best]
        d_uav_host = dists[best]

        # ---- Fixed interference rings (5 positions each) -----------------------
        worst_case_interf_list = build_fixed_interf_ring(serving_bs_idx, 'worst')
        best_case_interf_list  = build_fixed_interf_ring(serving_bs_idx, 'best')

        # -- Elevation angles ----------------------------------------
        angle_uav_host = elevation_angle(best_uav, Host_Pos_cart)
        angle_uav_bs   = elevation_angle(best_uav, serving_BS)
        K_hop1 = K_factor_elevation(angle_uav_host)
        K_hop2 = K_factor_elevation(angle_uav_bs)

        angles_interf_uav_gen  = [elevation_angle(best_uav, u)  for u in interf_rand_pos_list]
        angles_interf_uav_wor  = [elevation_angle(best_uav, u)  for u in worst_case_interf_list]
        angles_interf_uav_bes  = [elevation_angle(best_uav, u)  for u in best_case_interf_list]

        angles_interf_bs_gen   = [elevation_angle(serving_BS, u) for u in interf_rand_pos_list]
        angles_interf_bs_wor   = [elevation_angle(serving_BS, u) for u in worst_case_interf_list]
        angles_interf_bs_bes   = [elevation_angle(serving_BS, u) for u in best_case_interf_list]

        # --- Fading + shadowing draws ----------------------------------
        fading_h1_gen = [rician_gain(K_factor_elevation(angles_interf_uav_gen[i]))[0]
                         * shadowing_gain(angles_interf_uav_gen[i])
                         for i in range(len(interf_rand_pos_list))]
        fading_h2_gen = [rician_gain(K_factor_elevation(angles_interf_bs_gen[i]))[0]
                         * shadowing_gain(angles_interf_bs_gen[i])
                         for i in range(len(interf_rand_pos_list))]

        fading_h1_wor = [rician_gain(K_factor_elevation(angles_interf_uav_wor[i]))[0]
                         * shadowing_gain(angles_interf_uav_wor[i])
                         for i in range(len(worst_case_interf_list))]
        fading_h2_wor = [rician_gain(K_factor_elevation(angles_interf_bs_wor[i]))[0]
                         * shadowing_gain(angles_interf_bs_wor[i])
                         for i in range(len(worst_case_interf_list))]

        fading_h1_bes = [rician_gain(K_factor_elevation(angles_interf_uav_bes[i]))[0]
                         * shadowing_gain(angles_interf_uav_bes[i])
                         for i in range(len(best_case_interf_list))]
        fading_h2_bes = [rician_gain(K_factor_elevation(angles_interf_bs_bes[i]))[0]
                         * shadowing_gain(angles_interf_bs_bes[i])
                         for i in range(len(best_case_interf_list))]

        fading_desired_h1 = rician_gain(K_hop1)[0] * shadowing_gain(angle_uav_host)
        fading_desired_h2 = rician_gain(K_hop2)[0]

        fading_baseline_desired  = rayleigh_gain() * shadowing_gain(
            elevation_angle(serving_BS, Host_Pos_cart))
        fading_baseline_interf   = [rayleigh_gain() * shadowing_gain(angles_interf_bs_gen[i])
                                    for i in range(len(interf_rand_pos_list))]

        N0 = Noise()

        # =========================================================
        # Hop distances
        d_bs_uav = dist3d(serving_BS, best_uav)

        # -- General case ------------------------------------
        signal1_gen = (P_tx_Host * fading_desired_h1 * path_loss(d_uav_host, PathLossExponent_UAV)) / N0
        interf1_gen = sum(
            (P_tx_Host * fading_h1_gen[i] * path_loss(dist3d(best_uav, u), PathLossExponent_UAV)) / N0
            for i, u in enumerate(interf_rand_pos_list))
        SINR1_gen = signal1_gen / (interf1_gen + 1)

        signal2_gen = (P_tx_UAV * fading_desired_h2 * path_loss(d_bs_uav, PathLossExponent_UAV)) / N0
        interf2_gen = sum(
            (P_tx_Host * fading_h2_gen[i] * path_loss(dist3d(serving_BS, u), PathLossExponent_UAV)) / N0
            for i, u in enumerate(interf_rand_pos_list))
        SINR2_gen = signal2_gen / (interf2_gen + 1)

        # -- Worst case ---------------------------------------------
        signal1_wor = signal1_gen
        interf1_wor = sum(
            (P_tx_Host * fading_h1_wor[i] * path_loss(dist3d(best_uav, u), PathLossExponent_UAV)) / N0
            for i, u in enumerate(worst_case_interf_list))
        SINR1_wor = signal1_wor / (interf1_wor + 1)

        signal2_wor = signal2_gen
        interf2_wor = sum(
            (P_tx_Host * fading_h2_wor[i] * path_loss(dist3d(serving_BS, u), PathLossExponent_UAV)) / N0
            for i, u in enumerate(worst_case_interf_list))
        SINR2_wor = signal2_wor / (interf2_wor + 1)

        # -- Best case ----------------------------------------------
        signal1_bes = signal1_gen
        interf1_bes = sum(
            (P_tx_Host * fading_h1_bes[i] * path_loss(dist3d(best_uav, u), PathLossExponent_UAV)) / N0
            for i, u in enumerate(best_case_interf_list))
        SINR1_bes = signal1_bes / (interf1_bes + 1)

        signal2_bes = signal2_gen
        interf2_bes = sum(
            (P_tx_Host * fading_h2_bes[i] * path_loss(dist3d(serving_BS, u), PathLossExponent_UAV)) / N0
            for i, u in enumerate(best_case_interf_list))
        SINR2_bes = signal2_bes / (interf2_bes + 1)

        # -- Baseline: Host → nearest-neighbour BS directly ----------------------------
        d_bs_host       = dist3d(serving_BS, Host_Pos_cart)
        signal_baseline = (P_tx_Host * fading_baseline_desired * path_loss(d_bs_host, PathLossExponent_urban)) / N0
        interf_baseline = sum(
            (P_tx_Host * fading_baseline_interf[i] * path_loss(dist3d(serving_BS, u), PathLossExponent_urban)) / N0
            for i, u in enumerate(interf_rand_pos_list))

        # -- Power total ----------------------------------
        P_total_UAV = P_tx_UAV + P_tx_Host + P_circuit_host + num_UAV * (P_circuit_UAV + P_UAV_flight)

        # -- End-to-end SINR ------------------------------------
        SINR_e2e_gen      = e2e_sinr(SINR1_gen, SINR2_gen, relay_mode)
        SINR_e2e_wor      = e2e_sinr(SINR1_wor, SINR2_wor, relay_mode)
        SINR_e2e_bes      = e2e_sinr(SINR1_bes, SINR2_bes, relay_mode)
        SINR_e2e_baseline = signal_baseline / (interf_baseline + 1)


        performance_metrics(SINR_e2e_gen,      True,  acc_gen,      P_total_UAV)
        performance_metrics(SINR_e2e_wor,      True,  acc_wor,      P_total_UAV)
        performance_metrics(SINR_e2e_bes,      True,  acc_bes,      P_total_UAV)
        performance_metrics(SINR_e2e_baseline, False, acc_baseline, P_total_UAV)

        sinr_baseline_total += float(np.squeeze(SINR_e2e_baseline))

    # -- Append averages --------------------------
    SINR_list.append(acc_gen.SINR_sum / num_iterations)
    Capacity_list.append(acc_gen.Capacity_sum / num_iterations)
    Outage_list.append(acc_gen.outage_sum / num_iterations)
    BER_list.append(acc_gen.BER_sum / num_iterations)
    Energy_Eff_list.append(acc_gen.Energy_Eff_sum / num_iterations)

    SINR_list_wor.append(acc_wor.SINR_sum / num_iterations)
    Capacity_list_wor.append(acc_wor.Capacity_sum / num_iterations)
    Outage_list_wor.append(acc_wor.outage_sum / num_iterations)
    BER_list_wor.append(acc_wor.BER_sum / num_iterations)
    Energy_Eff_list_wor.append(acc_wor.Energy_Eff_sum / num_iterations)

    SINR_list_bes.append(acc_bes.SINR_sum / num_iterations)
    Capacity_list_bes.append(acc_bes.Capacity_sum / num_iterations)
    Outage_list_bes.append(acc_bes.outage_sum / num_iterations)
    BER_list_bes.append(acc_bes.BER_sum / num_iterations)
    Energy_Eff_list_bes.append(acc_bes.Energy_Eff_sum / num_iterations)

    SINR_list_baseline.append(acc_baseline.SINR_sum / num_iterations)


# =========== Plotting ==========================================
def print_results():
    uav_range = range(1, num_UAVs_total + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Disaster Scenario — UAV Relay Performance vs Number of UAVs\n'
                 '(Central BS Non-Operational; Host Served by Nearest Neighbour BS)',
                 fontsize=13)

    axes[0, 0].plot(uav_range, SINR_list,     'b-o', markersize=3, label='General')
    axes[0, 0].plot(uav_range, SINR_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 0].plot(uav_range, SINR_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 0].set_xlabel('Number of UAVs'); axes[0, 0].set_ylabel('Mean SINR (linear)')
    axes[0, 0].set_title('Mean SINR'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(uav_range, Capacity_list,     'b-o', markersize=3, label='General')
    axes[0, 1].plot(uav_range, Capacity_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 1].plot(uav_range, Capacity_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 1].set_xlabel('Number of UAVs'); axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    axes[0, 1].set_title('Channel Capacity'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[0, 2].plot(uav_range, Outage_list,     'b-o', markersize=3, label='General')
    axes[0, 2].plot(uav_range, Outage_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 2].plot(uav_range, Outage_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 2].set_xlabel('Number of UAVs'); axes[0, 2].set_ylabel('Outage Probability')
    axes[0, 2].set_title('Outage Probability'); axes[0, 2].legend(); axes[0, 2].grid(True)

    axes[1, 0].semilogy(uav_range, BER_list,     'b-o', markersize=3, label='General')
    axes[1, 0].semilogy(uav_range, BER_list_wor, 'r--', markersize=3, label='Worst')
    axes[1, 0].semilogy(uav_range, BER_list_bes, 'g--', markersize=3, label='Best')
    axes[1, 0].set_xlabel('Number of UAVs'); axes[1, 0].set_ylabel('BER')
    axes[1, 0].set_title('BER (log scale)'); axes[1, 0].legend(); axes[1, 0].grid(True, which='both')

    axes[1, 1].plot(uav_range, Energy_Eff_list,     'b-o', markersize=3, label='General')
    axes[1, 1].plot(uav_range, Energy_Eff_list_wor, 'r--', markersize=3, label='Worst')
    axes[1, 1].plot(uav_range, Energy_Eff_list_bes, 'g--', markersize=3, label='Best')
    axes[1, 1].set_xlabel('Number of UAVs'); axes[1, 1].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
    axes[1, 1].set_title('Energy Efficiency'); axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].plot(uav_range, SINR_list_baseline, 'y-o', markersize=3, label='Baseline (no UAV)')
    axes[1, 2].set_xlabel('Number of UAVs'); axes[1, 2].set_ylabel('Mean SINR (linear)')
    axes[1, 2].set_title('Baseline SINR (Host → Nearest Neighbour BS, No Relay)')
    axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

baseline_average = sinr_baseline_total / (num_iterations * num_UAVs_total)
print(f"\nOverall baseline SINR average: {baseline_average:.4f}")
print_results()