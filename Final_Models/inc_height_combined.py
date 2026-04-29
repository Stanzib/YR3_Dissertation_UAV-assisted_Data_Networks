# The combination of all power increase scripts - AI brought them together and subsequently tidied it up

print("script started: Combined Height Model Comparison")

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from dataclasses import dataclass

# ------- Models Map ----------------------------------------
Dist_between_BS = 2500
serving_x = 5000
serving_y = 5000
BS_Height = 25

BS_Pos_urban = [
    [serving_x,                                   serving_y + Dist_between_BS,        BS_Height],
    [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    BS_Height],
    [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    BS_Height],
    [serving_x,                                   serving_y,                          BS_Height],  # serving BS = index 3
    [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    BS_Height],
    [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    BS_Height],
    [serving_x,                                   serving_y - Dist_between_BS,        BS_Height]
]

BS_Pos_disaster = [
    [serving_x,                                   serving_y + Dist_between_BS,        BS_Height],  # BS0
    [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    BS_Height],  # BS1
    [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    BS_Height],  # BS2
    [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    BS_Height],  # BS3
    [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    BS_Height],  # BS4
    [serving_x,                                   serving_y - Dist_between_BS,        BS_Height],  # BS5
    [serving_x,                                   serving_y,                          BS_Height],  # BS6 = dead
]

NEIGHBOUR_BS_IDX = list(range(6))
R_max            = Dist_between_BS * 0.5
R_0              = 20

# ------- Shared performance parameters ----------------------
P_tx_UAV         = 5
P_tx_Host        = 0.2
P_circuit_UAV    = 10
P_circuit_host   = 5
P_UAV_flight     = 150
Outage_threshold = 1
relay_mode       = 'AF'
num_UAV          = 100  # fixed UAV count for height sweep
Coding_gain      = 10

# ------- Per-model channel parameters -----------------------
MODELS = {
    'Ideal': {
        'PathLossExponent_ground': 2.5,
        'PathLossExponent_UAV':    2.0,
        'shadowing_angle_cutoff':  30,
        'baseline_fading':         'rician',
        'sigma_LoS_dB':            3.0,
        'sigma_NLoS_dB':           11.0,
    },
    'Urban': {
        'PathLossExponent_ground': 4.0,
        'PathLossExponent_UAV':    3.0,
        'shadowing_angle_cutoff':  60,
        'baseline_fading':         'rayleigh',
        'sigma_LoS_dB':            5.0,
        'sigma_NLoS_dB':           17.0,
    },
    'Disaster': {
        'PathLossExponent_ground': 3.25,
        'PathLossExponent_UAV':    2.5,
        'shadowing_angle_cutoff':  45,
        'baseline_fading':         'rayleigh',
        'sigma_LoS_dB':            4.0,
        'sigma_NLoS_dB':           14.0,
    },
}

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
    return max(d, 0.1) ** (-alpha)

def Noise():
    return 1.38e-23 * 290 * 1

def e2e_sinr(sinr1, sinr2, mode='AF'):
    if mode == 'DF':
        return min(sinr1, sinr2)
    return (sinr1 * sinr2) / (sinr1 + sinr2 + 1)

def shadowing_gain(angle_deg, cutoff, sigma_los_dB, sigma_nlos_dB):
    sigma_dB = sigma_nlos_dB - (sigma_nlos_dB - sigma_los_dB) * (angle_deg / 90.0)
    sigma_dB = np.clip(sigma_dB, sigma_los_dB, sigma_nlos_dB)
    return 10 ** (np.random.normal(0, sigma_dB) / 10)

def shadowing_gain_ground(sigma_nlos_dB):
    return 10 ** (np.random.normal(0, sigma_nlos_dB) / 10)

def elevation_angle(pos_a, pos_b):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    dz = pos_a[2] - pos_b[2]
    return np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))

def K_factor_elevation(angle_deg):
    return np.exp(np.log(100.0) / (np.pi / 2) * angle_deg)

def dist3d(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def dist2d(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def random_pos_ground(cx, cy):
    theta = 2 * np.pi * np.random.uniform()
    r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform())
    return [cx + r*np.cos(theta), cy + r*np.sin(theta), 0]

def random_pos_uav(cx, cy, height):
    theta = 2 * np.pi * np.random.uniform()
    r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform())
    return [cx + r*np.cos(theta), cy + r*np.sin(theta), height]

def nearest_bs(host_pos, candidate_indices, BS_Pos):
    dists = [dist2d(host_pos, BS_Pos[i]) for i in candidate_indices]
    return candidate_indices[int(np.argmin(dists))]

def build_fixed_interf_ring(serving_bs_idx, case, BS_Pos):
    sx, sy = BS_Pos[serving_bs_idx][0], BS_Pos[serving_bs_idx][1]
    result = []
    for i in [i for i in NEIGHBOUR_BS_IDX if i != serving_bs_idx]:
        bx, by = BS_Pos[i][0], BS_Pos[i][1]
        dx, dy = sx - bx, sy - by
        d      = np.sqrt(dx**2 + dy**2)
        ux, uy = dx/d, dy/d
        if case == 'worst':
            result.append([bx + R_max*ux, by + R_max*uy, 0])
        else:
            result.append([bx - R_max*ux, by - R_max*uy, 0])
    return result

def performance_metrics(sinr, is_uav, acc, P_total_UAV):
    sinr = float(np.squeeze(sinr))
    Coding_gain_linear = 10**(Coding_gain/10)
    BER  = 0.5 * special.erfc(np.sqrt(max(sinr * Coding_gain_linear, 0)))
    if is_uav:
        Cap = 0.5 * np.log2(1 + sinr)
        EE  = Cap / P_total_UAV
    else:
        Cap = 0.5 * np.log2(1 + sinr)  # consistent half-duplex for baseline
        EE  = Cap / (P_circuit_host + P_tx_Host)
    acc.outage_sum     += 1 if sinr < Outage_threshold else 0
    acc.SINR_sum       += sinr
    acc.Capacity_sum   += Cap
    acc.Energy_Eff_sum += EE
    acc.BER_sum        += BER


# =========== Monte Carlo ==========================================
height_total   = 150
num_iterations = 5_000

results = {m: {
    'SINR': [], 'SINR_wor': [], 'SINR_bes': [],
    'Cap':  [], 'Cap_wor':  [], 'Cap_bes':  [],
    'Out':  [], 'Out_wor':  [], 'Out_bes':  [],
    'BER':  [], 'BER_wor':  [], 'BER_bes':  [],
    'EE':   [], 'EE_wor':   [], 'EE_bes':   [],
    'SINR_baseline': [],
    'Cap_baseline':  [],
    'BER_baseline':  [],
    'baseline_total': 0.0,
} for m in MODELS}

for model_name, params in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Running model: {model_name}")

    alpha_g     = params['PathLossExponent_ground']
    alpha_uav   = params['PathLossExponent_UAV']
    sigma_los  = params['sigma_LoS_dB']
    sigma_nlos = params['sigma_NLoS_dB']
    cutoff      = params['shadowing_angle_cutoff']
    bl_fading   = params['baseline_fading']
    is_disaster = (model_name == 'Disaster')
    BS_Pos      = BS_Pos_disaster if is_disaster else BS_Pos_urban
    servicing_BS = None if is_disaster else 3

    if not is_disaster:
        Dist_wor = Dist_between_BS - R_max
        Dist_bes = Dist_between_BS + R_max
        worst_case_interf_list = [
            [serving_x,                                     serving_y + Dist_wor,       0],
            [serving_x - Dist_wor*(np.sqrt(3)/2), serving_y + Dist_wor*0.5,   0],
            [serving_x + Dist_wor*(np.sqrt(3)/2), serving_y + Dist_wor*0.5,   0],
            [serving_x - Dist_wor*(np.sqrt(3)/2), serving_y - Dist_wor*0.5,   0],
            [serving_x + Dist_wor*(np.sqrt(3)/2), serving_y - Dist_wor*0.5,   0],
            [serving_x,                                     serving_y - Dist_wor,       0],
        ]
        best_case_interf_list = [
            [serving_x,                                     serving_y + Dist_bes,       0],
            [serving_x - Dist_bes*(np.sqrt(3)/2), serving_y + Dist_bes*0.5,   0],
            [serving_x + Dist_bes*(np.sqrt(3)/2), serving_y + Dist_bes*0.5,   0],
            [serving_x - Dist_bes*(np.sqrt(3)/2), serving_y - Dist_bes*0.5,   0],
            [serving_x + Dist_bes*(np.sqrt(3)/2), serving_y - Dist_bes*0.5,   0],
            [serving_x,                                     serving_y - Dist_bes,       0],
        ]

    acc_gen = PerfAccumulator(); acc_wor = PerfAccumulator()
    acc_bes = PerfAccumulator(); acc_bl  = PerfAccumulator()

    for UAV_Height in range(1, height_total + 1):
        acc_gen.reset(); acc_wor.reset(); acc_bes.reset(); acc_bl.reset()
        print(f"  {model_name} | Height {UAV_Height}/{height_total}", end='\r')

        for j in range(num_iterations):

            N0          = Noise()
            P_total_UAV = P_tx_UAV + P_tx_Host + P_circuit_host + num_UAV*(P_circuit_UAV + P_UAV_flight)

            # --- Spawn positions ---
            if is_disaster:
                UAV_pos_list  = [random_pos_uav(serving_x, serving_y, UAV_Height) for _ in range(num_UAV)]
                Host_Pos_cart = random_pos_ground(serving_x, serving_y)
                srv_idx       = nearest_bs(Host_Pos_cart, NEIGHBOUR_BS_IDX, BS_Pos)
                serving_BS    = BS_Pos[srv_idx]
                interf_rand_pos_list   = [random_pos_ground(BS_Pos[i][0], BS_Pos[i][1])
                                          for i in NEIGHBOUR_BS_IDX if i != srv_idx]
                worst_case_interf_list = build_fixed_interf_ring(srv_idx, 'worst', BS_Pos)
                best_case_interf_list  = build_fixed_interf_ring(srv_idx, 'best',  BS_Pos)
            else:
                UAV_pos_list  = [random_pos_uav(BS_Pos[servicing_BS][0], BS_Pos[servicing_BS][1], UAV_Height)
                                 for _ in range(num_UAV)]
                Host_Pos_cart = random_pos_ground(BS_Pos[servicing_BS][0], BS_Pos[servicing_BS][1])
                serving_BS    = BS_Pos[servicing_BS]
                interf_rand_pos_list = [random_pos_ground(u[0], u[1])
                                        for k, u in enumerate(BS_Pos) if k != servicing_BS]

            # --- Best UAV ---
            dists      = [dist3d(Host_Pos_cart, u) for u in UAV_pos_list]
            best       = int(np.argmin(dists))
            best_uav   = UAV_pos_list[best]
            d_uav_host = dists[best]
            d_bs_uav   = dist3d(serving_BS, best_uav)

            # --- Elevation angles ---
            ang_h  = elevation_angle(best_uav, Host_Pos_cart)
            ang_bs = elevation_angle(best_uav, serving_BS)

            ang_interf_uav_gen = [elevation_angle(best_uav, u) for u in interf_rand_pos_list]
            ang_interf_uav_wor = [elevation_angle(best_uav, u) for u in worst_case_interf_list]
            ang_interf_uav_bes = [elevation_angle(best_uav, u) for u in best_case_interf_list]
            ang_interf_bs_gen  = [elevation_angle(serving_BS, u) for u in interf_rand_pos_list]
            ang_interf_bs_wor  = [elevation_angle(serving_BS, u) for u in worst_case_interf_list]
            ang_interf_bs_bes  = [elevation_angle(serving_BS, u) for u in best_case_interf_list]

            # --- Fading draws ---
            fh1_gen = [rician_gain(K_factor_elevation(a))[0] * shadowing_gain(a, cutoff, sigma_los, sigma_nlos) for a in ang_interf_uav_gen]
            fh2_gen = [rician_gain(K_factor_elevation(a))[0] * shadowing_gain(a, cutoff, sigma_los, sigma_nlos) for a in ang_interf_bs_gen]
            fh1_wor = [rician_gain(K_factor_elevation(a))[0] * shadowing_gain(a, cutoff, sigma_los, sigma_nlos) for a in ang_interf_uav_wor]
            fh2_wor = [rician_gain(K_factor_elevation(a))[0] * shadowing_gain(a, cutoff, sigma_los, sigma_nlos) for a in ang_interf_bs_wor]
            fh1_bes = [rician_gain(K_factor_elevation(a))[0] * shadowing_gain(a, cutoff, sigma_los, sigma_nlos) for a in ang_interf_uav_bes]
            fh2_bes = [rician_gain(K_factor_elevation(a))[0] * shadowing_gain(a, cutoff, sigma_los, sigma_nlos) for a in ang_interf_bs_bes]

            fd_h1 = rician_gain(K_factor_elevation(ang_h))[0]  * shadowing_gain(ang_h,  cutoff, sigma_los, sigma_nlos)
            fd_h2 = rician_gain(K_factor_elevation(ang_bs))[0]

            if bl_fading == 'rayleigh':
                fd_bl      = float(np.squeeze(rayleigh_gain()))  * shadowing_gain_ground(sigma_nlos)
                fbl_interf = [float(np.squeeze(rayleigh_gain())) * shadowing_gain_ground(sigma_nlos) for a in ang_interf_bs_gen]
            else:
                ang_bl     = elevation_angle(serving_BS, Host_Pos_cart)
                fd_bl      = rician_gain(K_factor_elevation(ang_bl))[0] * shadowing_gain(ang_bl, cutoff, sigma_los, sigma_nlos)
                fbl_interf = [rician_gain(K_factor_elevation(a))[0]     * shadowing_gain(a,      cutoff, sigma_los, sigma_nlos) for a in ang_interf_bs_gen]

            # --- SINR calculations ---
            sig1 = (P_tx_Host * fd_h1 * path_loss(d_uav_host, alpha_uav)) / N0
            sig2 = (P_tx_UAV  * fd_h2 * path_loss(d_bs_uav,   alpha_uav)) / N0

            def interf_sum(fading_list, pos_list, rx_pos, alpha):
                return sum((P_tx_Host * fading_list[i] * path_loss(dist3d(rx_pos, u), alpha)) / N0
                           for i, u in enumerate(pos_list))

            I1_gen = interf_sum(fh1_gen, interf_rand_pos_list,   best_uav,   alpha_uav)
            I2_gen = interf_sum(fh2_gen, interf_rand_pos_list,   serving_BS, alpha_g)
            I1_wor = interf_sum(fh1_wor, worst_case_interf_list, best_uav,   alpha_uav)
            I2_wor = interf_sum(fh2_wor, worst_case_interf_list, serving_BS, alpha_g)
            I1_bes = interf_sum(fh1_bes, best_case_interf_list,  best_uav,   alpha_uav)
            I2_bes = interf_sum(fh2_bes, best_case_interf_list,  serving_BS, alpha_g)

            SINR1_gen = sig1 / (I1_gen + 1);  SINR2_gen = sig2 / (I2_gen + 1)
            SINR1_wor = sig1 / (I1_wor + 1);  SINR2_wor = sig2 / (I2_wor + 1)
            SINR1_bes = sig1 / (I1_bes + 1);  SINR2_bes = sig2 / (I2_bes + 1)

            d_bs_host = dist3d(serving_BS, Host_Pos_cart)
            sig_bl    = (P_tx_Host * fd_bl * path_loss(d_bs_host, alpha_g)) / N0
            I_bl      = interf_sum(fbl_interf, interf_rand_pos_list, serving_BS, alpha_g)

            SINR_gen = e2e_sinr(SINR1_gen, SINR2_gen, relay_mode)
            SINR_wor = e2e_sinr(SINR1_wor, SINR2_wor, relay_mode)
            SINR_bes = e2e_sinr(SINR1_bes, SINR2_bes, relay_mode)
            SINR_bl  = float(np.squeeze(sig_bl / (I_bl + 1)))

            performance_metrics(SINR_gen, True,  acc_gen, P_total_UAV)
            performance_metrics(SINR_wor, True,  acc_wor, P_total_UAV)
            performance_metrics(SINR_bes, True,  acc_bes, P_total_UAV)
            performance_metrics(SINR_bl,  True,  acc_bl,  P_total_UAV)
            results[model_name]['baseline_total'] += SINR_bl

        r = results[model_name]
        r['SINR'].append(acc_gen.SINR_sum / num_iterations)
        r['SINR_wor'].append(acc_wor.SINR_sum / num_iterations)
        r['SINR_bes'].append(acc_bes.SINR_sum / num_iterations)
        r['Cap'].append(acc_gen.Capacity_sum / num_iterations)
        r['Cap_wor'].append(acc_wor.Capacity_sum / num_iterations)
        r['Cap_bes'].append(acc_bes.Capacity_sum / num_iterations)
        r['Out'].append(acc_gen.outage_sum / num_iterations)
        r['Out_wor'].append(acc_wor.outage_sum / num_iterations)
        r['Out_bes'].append(acc_bes.outage_sum / num_iterations)
        r['BER'].append(acc_gen.BER_sum / num_iterations)
        r['BER_wor'].append(acc_wor.BER_sum / num_iterations)
        r['BER_bes'].append(acc_bes.BER_sum / num_iterations)
        r['EE'].append(acc_gen.Energy_Eff_sum / num_iterations)
        r['EE_wor'].append(acc_wor.Energy_Eff_sum / num_iterations)
        r['EE_bes'].append(acc_bes.Energy_Eff_sum / num_iterations)
        r['SINR_baseline'].append(acc_bl.SINR_sum / num_iterations)
        r['Cap_baseline'].append(acc_bl.Capacity_sum / num_iterations)
        r['BER_baseline'].append(acc_bl.BER_sum / num_iterations)

    print(f"\n  {model_name} done.")


# =========== Plotting ==========================================
def print_results(model_name):
    r            = results[model_name]
    height_range = range(1, height_total + 1)
    bl_avg       = float(r['baseline_total'] / (num_iterations * height_total))
    bl_cap_avg   = float(sum(r['Cap_baseline']) / len(r['Cap_baseline']))
    bl_ber_avg   = float(sum(r['BER_baseline']) / len(r['BER_baseline']))
    bl_out_avg   = float(sum(r['Out']) / len(r['Out']))

    eff_SINR = [max(s, bl_avg)     for s in r['SINR']]
    eff_Cap  = [max(c, bl_cap_avg) for c in r['Cap']]
    eff_Out  = [min(o, bl_out_avg) for o in r['Out']]
    eff_BER  = [min(b, bl_ber_avg) for b in r['BER']]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} Model — UAV Relay Performance vs UAV Height', fontsize=13)

    # ── SINR ─────────────────────────────────────────────────────────────────
    axes[0, 0].plot(height_range, r['SINR'],     'b-o', markersize=3, label='General')
    axes[0, 0].plot(height_range, r['SINR_wor'], 'r--', markersize=3, label='Worst')
    axes[0, 0].plot(height_range, r['SINR_bes'], 'g--', markersize=3, label='Best')
    axes[0, 0].set_xlabel('UAV Height (m)'); axes[0, 0].set_ylabel('Mean SINR (linear)')
    axes[0, 0].set_title(f'Increasing UAV Altitude {model_name} Mean SINR'); axes[0, 0].legend(); axes[0, 0].grid(True)

    # ── Capacity ──────────────────────────────────────────────────────────────
    axes[0, 1].axhline(bl_cap_avg, color='y', linestyle='--', linewidth=1.5,
                       label=f'Baseline (avg = {bl_cap_avg:.2f})')
    axes[0, 1].plot(height_range, r['Cap'],        'b-o', markersize=3, label='General')
    axes[0, 1].plot(height_range, r['Cap_wor'],   'r--', markersize=3, label='Worst')
    axes[0, 1].plot(height_range, r['Cap_bes'],   'g--', markersize=3, label='Best')
    axes[0, 1].set_xlabel('UAV Height (m)'); axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    axes[0, 1].set_title(f'Increasing UAV Altitude {model_name} Channel Capacity'); axes[0, 1].legend(); axes[0, 1].grid(True)

    # ── Outage ────────────────────────────────────────────────────────────────
    axes[0, 2].axhline(bl_out_avg, color='y', linestyle='--', linewidth=1.5,
                       label=f'Baseline (avg = {bl_out_avg:.2f})')
    axes[0, 2].plot(height_range, r['Out'],        'b-o', markersize=3, label='General')
    axes[0, 2].plot(height_range, r['Out_wor'],   'r--', markersize=3, label='Worst')
    axes[0, 2].plot(height_range, r['Out_bes'],   'g--', markersize=3, label='Best')
    axes[0, 2].set_xlabel('UAV Height (m)'); axes[0, 2].set_ylabel('Outage Probability')
    axes[0, 2].set_title(f'Increasing UAV Altitude {model_name} Outage Probability'); axes[0, 2].legend(); axes[0, 2].grid(True)

    # ── BER ───────────────────────────────────────────────────────────────────
    axes[1, 0].axhline(bl_ber_avg, color='y', linestyle='--', linewidth=1.5,
                       label=f'Baseline (avg = {bl_ber_avg:.4f})')
    axes[1, 0].semilogy(height_range, r['BER'],       'b-o', markersize=3, label='General')
    axes[1, 0].semilogy(height_range, r['BER_wor'],  'r--', markersize=3, label='Worst')
    axes[1, 0].semilogy(height_range, r['BER_bes'],  'g--', markersize=3, label='Best')
    axes[1, 0].set_xlabel('UAV Height (m)'); axes[1, 0].set_ylabel('BER')
    axes[1, 0].set_title(f'Increasing UAV Altitude {model_name} BER (log scale)'); axes[1, 0].legend(); axes[1, 0].grid(True, which='both')

    # ── Energy Efficiency ─────────────────────────────────────────────────────
    axes[1, 1].plot(height_range, r['EE'],     'b-o', markersize=3, label='General')
    axes[1, 1].plot(height_range, r['EE_wor'], 'r--', markersize=3, label='Worst')
    axes[1, 1].plot(height_range, r['EE_bes'], 'g--', markersize=3, label='Best')
    axes[1, 1].set_xlabel('UAV Height (m)'); axes[1, 1].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
    axes[1, 1].set_title(f'Increasing UAV Altitude {model_name} Energy Efficiency'); axes[1, 1].legend(); axes[1, 1].grid(True)

    # ── Effective SINR handover ───────────────────────────────────────────────
    axes[1, 2].axhline(bl_avg, color='y', linestyle='--', linewidth=1.5,
                       label=f'Baseline (avg = {bl_avg:.2f})')
    axes[1, 2].plot(height_range, eff_SINR, 'b-o', markersize=3,
                    label='Effective SINR (Baseline → General)')
    axes[1, 2].set_xlabel('UAV Height (m)'); axes[1, 2].set_ylabel('Mean SINR (linear)')
    axes[1, 2].set_title(f'Increasing UAV Altitude {model_name} Effective SINR')
    axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(rf'C:\Users\n70943sb\OneDrive - The University of Manchester\uni\Y3\Individual Project\Results\{model_name}_plots_incHeight.png')
    plt.show()


def plot_model_comparison():
    height_range = range(1, height_total + 1)

    styles = {
        'Ideal':    {'color': 'b', 'marker': 'o'},
        'Urban':    {'color': 'r', 'marker': 's'},
        'Disaster': {'color': 'g', 'marker': '^'},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('General SINR Comparison Across UAV Height', fontsize=13)

    for model_name, r in results.items():
        s      = styles[model_name]
        bl_avg = float(r['baseline_total'] / (num_iterations * height_total))
        eff_SINR = [max(val, bl_avg) for val in r['SINR']]

        ax.plot(height_range, r['SINR'], color=s['color'], marker=s['marker'],
                markersize=3, linestyle='-', label=f'{model_name} SINR')

    ax.set_xlabel('UAV Height (m)')
    ax.set_ylabel('Mean SINR (linear)')
    ax.legend()
    ax.grid(True)

    plt.savefig(rf'C:\Users\n70943sb\OneDrive - The University of Manchester\uni\Y3\Individual Project\Results\combined_SINR_plot_incHeight.png')
    plt.tight_layout()
    plt.show()


print_results('Ideal')
print_results('Urban')
print_results('Disaster')
plot_model_comparison()
