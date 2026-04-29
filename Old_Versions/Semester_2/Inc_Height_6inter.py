# 3D model a host within an Base Stations (BS) range
# There is a large urban area
# We increase the amount of power transmitted by host
# UAV power transmittion is uneffected
# uplink
# 6 interfering BS
# best case worst case calculated

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from dataclasses import dataclass, field

print ("start script: Increaseing height, Urban model")
# ------- Models Map ----------------------------------------
Dist_between_BS = 2500
serving_x = 5000
serving_y = 5000
# hexagon with each corner as a BS, Centre as the serving BS. Listed from top left to right.
# [dc, 0] = x, [dc, 1] = y, [dc, 2] = z
BS_Pos = [
          [serving_x,                                   serving_y + Dist_between_BS,        0], # [0, dc] = BS0 coordinates
          [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    0], # [1, dc] = BS1 coordinates
          [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    0],
          [serving_x,                                   serving_y,                          0], # Serving BS (3)
          [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    0],
          [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    0],
          [serving_x,                                   serving_y - Dist_between_BS,        0]
        ]

servicing_BS = 3
R_max = Dist_between_BS * 0.5
Dist_to_worst_case = Dist_between_BS - R_max
Dist_to_Best_case = Dist_between_BS + R_max

# missing serving bs out of this list

best_case_interf_list = [
    [serving_x,                                      serving_y + Dist_to_Best_case,        0],
    [serving_x - Dist_to_Best_case*(np.sqrt(3)/2),  serving_y + Dist_to_Best_case*0.5,    0],
    [serving_x + Dist_to_Best_case*(np.sqrt(3)/2),  serving_y + Dist_to_Best_case*0.5,    0],
    [serving_x - Dist_to_Best_case*(np.sqrt(3)/2),  serving_y - Dist_to_Best_case*0.5,    0],
    [serving_x + Dist_to_Best_case*(np.sqrt(3)/2),  serving_y - Dist_to_Best_case*0.5,    0],
    [serving_x,                                      serving_y - Dist_to_Best_case,        0]
]

worst_case_interf_list = [
    [serving_x,                                       serving_y + Dist_to_worst_case,       0],
    [serving_x - Dist_to_worst_case*(np.sqrt(3)/2),  serving_y + Dist_to_worst_case*0.5,   0],
    [serving_x + Dist_to_worst_case*(np.sqrt(3)/2),  serving_y + Dist_to_worst_case*0.5,   0],
    [serving_x - Dist_to_worst_case*(np.sqrt(3)/2),  serving_y - Dist_to_worst_case*0.5,   0],
    [serving_x + Dist_to_worst_case*(np.sqrt(3)/2),  serving_y - Dist_to_worst_case*0.5,   0],
    [serving_x,                                       serving_y - Dist_to_worst_case,       0]
]

# ------- Performance Metric variables ------------------------
P_tx_UAV = 1 # 2
P_tx_Host = 0.2 # 10
P_circuit_UAV = 10 #!!! need to find reference for these
P_circuit_host = 5
R_0 = 20 # minimum distance a random position can be to the centre its spawned around
PathLossExponent_urban = 3 # typical urban enviroment beta = 2 to 4
PathLossExponent_UAV = 2.5 # free space
P_UAV_flight = 150
Outage_threshold = 1 # 2.5e-5
K_factor = 10 # for ricain fading, 10 is strong LoS, infinate is perfect LoS. 
relay_mode     = 'AF'   # 'DF' = decode and forward or 'AF' = amplify and forward
num_UAV = 100
Max_height = 30

# ____Results list____
SINR_list = []
Capacity_list = []
BER_list = []
Outage_list = []
Energy_Eff_list = []

SINR_list_wor = []
Capacity_list_wor = []
BER_list_wor = []
Outage_list_wor = []
Energy_Eff_list_wor = []

SINR_list_bes = []
Capacity_list_bes = []
BER_list_bes = []
Outage_list_bes = []
Energy_Eff_list_bes = []

SINR_list_baseline = []
height_values = []


# ------- Help Functions ----------------------------------------
@dataclass # means you dont have to __init__
class PerfAccumulator:
    SINR_sum:       float = 0.0
    Capacity_sum:   float = 0.0
    BER_sum:        float = 0.0
    outage_sum:     float = 0.0
    Energy_Eff_sum: float = 0.0

    def reset(self):
        self.SINR_sum       = 0.0
        self.Capacity_sum   = 0.0
        self.BER_sum        = 0.0
        self.outage_sum     = 0.0
        self.Energy_Eff_sum = 0.0

def rician_gain(K, size=1): # A2G fading model
    s     = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))
    h_r   = s + sigma * np.random.randn(size)
    h_i   =     sigma * np.random.randn(size)
    return h_r**2 + h_i**2

def rayleigh_gain(size=1): # Urban fading model
    return np.random.exponential(1.0, size)

def path_loss(d, alpha):
    d = max(d, 0.1)   # avoid division by zero / extreme near-field values
    return d ** (-alpha)

def random_pos(center_x, center_y, is_UAV, height = 0):
    theta = 2 * np.pi * np.random.uniform()
    if is_UAV == False:
        r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform(0, 1))
        return [center_x + r * np.cos(theta),
                center_y + r * np.sin(theta),
                0]
    else:
        r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform(0, 1))
        return [center_x + r * np.cos(theta),
                center_y + r * np.sin(theta),
                height] # how high the UAV hovers
    
def Noise():
    k = 1.38*10**(-23)
    T = 290
    B= 1
    N_0 = k*T*B
    return N_0

def e2e_sinr(sinr1, sinr2, mode='DF'):
    if mode == 'DF':
        return min(sinr1, sinr2)
    else:  # AF
        return (sinr1 * sinr2) / (sinr1 + sinr2 + 1)

def performance_metrics(sinr, is_uav, acc: PerfAccumulator):
    BER = 0.5 * special.erfc(np.sqrt(sinr))

    if is_uav:
        Capacity   = 0.5 * np.log2(1 + sinr)          # half-duplex penalty
        Energy_Eff = Capacity / P_total_UAV
    else:
        Capacity   = np.log2(1 + sinr)
        Energy_Eff =  Capacity / (P_circuit_host + P_tx_Host)

    if sinr < Outage_threshold:
        acc.outage_sum += 1

    acc.SINR_sum       += sinr
    acc.Capacity_sum   += Capacity
    acc.Energy_Eff_sum += Energy_Eff
    acc.BER_sum        += BER

def shadowing_gain():
    rand = np.random.uniform()
    if rand > 0.2:
        return 0.01
    else:
        return 1

def elevation_angle(uav_pos, ground_pos):
    dx = uav_pos[0] - ground_pos[0]
    dy = uav_pos[1] - ground_pos[1]
    dz = uav_pos[2] - ground_pos[2]
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    return np.degrees(np.arctan2(dz, horizontal_dist))  # 0-90 degrees

def K_factor_elevation(angle_deg):
    # K increases with elevation angle - stronger LoS at higher angles
    # Common model: K(theta) = a * exp(b * theta)
    K_min = 1.0   # at 0° — tune this
    K_max = 100.0  # at 90° — tune this
    a = K_min
    b = np.log(K_max / K_min) / (np.pi / 2)
    return a * np.exp(b * angle_deg)    

#not sure about these
def LoS_probability(angle_deg):
    # Probability of LoS increases with elevation angle
    # ITU model for urban environment
    a = 9.61   # urban environment constants
    b = 0.16
    return 1 / (1 + a * np.exp(-b * (angle_deg - a)))

def shadowing_gain_elevation(angle_deg):
    p_los = LoS_probability(angle_deg)
    if np.random.uniform() < p_los:
        return 1.0       # LoS - no shadowing
    else:
        return 0.01      # NLoS - heavy shadowing

# =========== Monte Carlo ==========================================
#____Variables_______
num_height_inc = 200
num_iterations = 1_000
acc_gen = PerfAccumulator()   # general / UAV-relay case
acc_wor = PerfAccumulator()   # worst-case
acc_bes = PerfAccumulator()   # best-case
acc_baseline = PerfAccumulator()

height_increment = Max_height / num_height_inc

# -------- Each UAV -----------------------------------------
for i in range(1, num_height_inc + 1):

    #____Initalising____
    acc_gen.reset()
    acc_wor.reset()
    acc_bes.reset()
    acc_baseline.reset()

    Height_UAV = i * height_increment
    height_values.append(Height_UAV)
    
    print("starting Height: ", Height_UAV, "\n")

    

    # ------- Each Indervidual Host -------------------------
    for j in range(num_iterations):

        #____Reset and regenerate UAV positions each iteration____
        UAV_pos_list = [] # [x][0] gives UAVx x-coordinate, [x][1] gives UAVx y-coordinate, [x][2] gives UAVx z-coordinate
        for _ in range(num_UAV):
            UAV_pos_list.append(random_pos(BS_Pos[servicing_BS][0], BS_Pos[servicing_BS][1], True, Height_UAV))

        
        #___Initalising___
        signal_strength_UAV_Host = []
        interf_BS_Host = []

        #____Random Host Location___
        Host_Pos_cart = random_pos(BS_Pos[servicing_BS][0], BS_Pos[servicing_BS][1], False)

        #____Best UAV algorithem_____
        dists = [np.sqrt((Host_Pos_cart[0]-u[0])**2 +
                         (Host_Pos_cart[1]-u[1])**2 +
                         (Host_Pos_cart[2]-u[2])**2) for u in UAV_pos_list]

        #____UAV closest to midpoint between Host and Serving BS_____
        mid_x = (Host_Pos_cart[0] + BS_Pos[servicing_BS][0]) / 2
        mid_y = (Host_Pos_cart[1] + BS_Pos[servicing_BS][1]) / 2

        dists_to_mid = [np.sqrt((mid_x - u[0])**2 +
                                (mid_y - u[1])**2) for u in UAV_pos_list]

        best = int(np.argmin(dists_to_mid))
        d_uav_host = dists[best]  # keep using original dists for hop 1
        best_uav = UAV_pos_list[best]
        
        #____Interfering UD locations_____
        interf_rand_pos_list = []
        for k, u in enumerate(BS_Pos):
            if k != servicing_BS:
                rand_pos = random_pos(u[0], u[1], False)
                interf_rand_pos_list.append(rand_pos)


        #_______angle of UAV to BS and Host_________
        angle_uav_host = elevation_angle(best_uav, Host_Pos_cart)
        angle_uav_bs = elevation_angle(best_uav, BS_Pos[servicing_BS])
        K_hop1 =  K_factor_elevation(angle_uav_host)
        K_hop2 = K_factor_elevation(angle_uav_bs)

        #=========== Host -> UAV -> BS general case ======================
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_gen = (P_tx_Host * rician_gain(K_hop1)[0] * path_loss(d_uav_host, PathLossExponent_UAV)) / Noise()

        #____ interferance from other hosts_____
        #!!! aproximate K-factor for interferers
        interf1_gen = sum(
            (P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for u in interf_rand_pos_list
        )

        SINR1_gen = signal1_gen / (interf1_gen + 1)

        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        d_bs_uav = np.sqrt((BS_Pos[servicing_BS][0] - best_uav[0])**2 +
                           (BS_Pos[servicing_BS][1] - best_uav[1])**2 +
                           (BS_Pos[servicing_BS][2] - best_uav[2])**2)
        
        signal2_gen = (P_tx_UAV * rician_gain(K_hop2)[0] * path_loss(d_bs_uav, PathLossExponent_UAV)) / Noise()

        #______ interference from other UAV______
        interf2_gen = sum(
            (P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for u in interf_rand_pos_list
        )

        SINR2_gen = signal2_gen / (interf2_gen + 1)

        
        #============== Host -> UAV -> BS Worst case ==================

        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_wor = signal1_gen

        #____ interferance from other hosts_____
        interf1_wor = sum(
            (P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for u in worst_case_interf_list
        )

        SINR1_wor = signal1_wor / (interf1_wor + 1)

        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        signal2_wor = signal2_gen

        #______ interference from hosts______
        interf2_wor = sum(
            (P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for u in worst_case_interf_list
        )

        SINR2_wor = signal2_wor / (interf2_wor + 1)



        #============== Host -> UAV -> BS Best case ===================
        
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_bes = signal1_gen

        #____ interferance from other hosts_____
        interf1_bes = sum(
            (P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for u in best_case_interf_list
        )

        SINR1_bes = signal1_bes / (interf1_bes + 1)

        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        signal2_bes = signal2_gen

        #______ interference from other UAV______
        interf2_bes = sum(
            (P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for u in best_case_interf_list
        )

        SINR2_bes = signal2_bes / (interf2_bes + 1)


    
        #========= Baseline: Host -> BS (general) ====================

        d_bs_host = np.sqrt((BS_Pos[servicing_BS][0] - Host_Pos_cart[0])**2 +
                           (BS_Pos[servicing_BS][1] - Host_Pos_cart[1])**2 +
                           (BS_Pos[servicing_BS][2] - Host_Pos_cart[2])**2)
        
        singal_baseline = (P_tx_Host * rayleigh_gain() * shadowing_gain() * path_loss(d_bs_host, PathLossExponent_urban)) / Noise()

        #______ interference from other Host______
        interf_baseline = sum(
            (P_tx_Host * rayleigh_gain() * shadowing_gain() * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_urban)) / Noise()
            for u in interf_rand_pos_list
        )

        

        #_____ Power consumption _______
        P_total_UAV  = P_tx_UAV + P_tx_Host + P_circuit_host + num_UAV * (P_circuit_UAV + P_UAV_flight)

        #---------- Calculating performance metrics--------------
        SINR_e2e_gen = e2e_sinr(SINR1_gen, SINR2_gen, relay_mode)
        SINR_e2e_wor = e2e_sinr(SINR1_wor, SINR2_wor, relay_mode)
        SINR_e2e_bes = e2e_sinr(SINR1_bes, SINR2_bes, relay_mode)
        SINR_e2e_baseline = singal_baseline / (interf_baseline + 1)
        performance_metrics(SINR_e2e_gen, True, acc_gen)
        performance_metrics(SINR_e2e_wor, True, acc_wor)
        performance_metrics(SINR_e2e_bes, True, acc_bes)
        performance_metrics(SINR_e2e_baseline, False, acc_baseline)
   
    

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

    



def print_results2():
    height_range = height_values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('UAV Relay Performance vs UAV Height (Urban Environment)', fontsize=14)

    # ── SINR ─────────────────────────────────────────────────────────────────
    axes[0, 0].plot(height_range, SINR_list, 'b-o', markersize=3, label='General Case')
    axes[0, 0].plot(height_range, SINR_list_wor, 'r--', markersize=3, label='Worst Case')
    axes[0, 0].plot(height_range, SINR_list_bes, 'g--', markersize=3, label='Best Case')
    axes[0, 0].set_xlabel('UAV Height (m)')
    axes[0, 0].set_ylabel('Mean SINR (Linear Scale)')
    axes[0, 0].set_title('Average SINR vs UAV Height')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # ── Capacity ──────────────────────────────────────────────────────────────
    axes[0, 1].plot(height_range, Capacity_list, 'b-o', markersize=3, label='General')
    axes[0, 1].plot(height_range, Capacity_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 1].plot(height_range, Capacity_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 1].set_xlabel('UAV Height (m)')
    axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    axes[0, 1].set_title('Channel Capacity vs UAV Height')
    axes[0, 1].grid(True)

    # ── Outage ────────────────────────────────────────────────────────────────
    axes[0, 2].plot(height_range, Outage_list, 'b-o', markersize=3, label='General')
    axes[0, 2].plot(height_range, Outage_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 2].plot(height_range, Outage_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 2].set_xlabel('UAV Height (m)')
    axes[0, 2].set_ylabel('Outage Probability')
    axes[0, 2].set_title('Outage Probability vs UAV Height')
    axes[0, 2].grid(True)

    # ── BER ───────────────────────────────────────────────────────────────────
    axes[1, 0].semilogy(height_range, BER_list, 'b-o', markersize=3, label='General')
    axes[1, 0].semilogy(height_range, BER_list_wor, 'r--', markersize=3, label='Worst')
    axes[1, 0].semilogy(height_range, BER_list_bes, 'g--', markersize=3, label='Best')
    axes[1, 0].set_xlabel('UAV Height (m)')
    axes[1, 0].set_ylabel('Bit Error Rate (BER)')
    axes[1, 0].set_title('BER vs UAV Height (Log Scale)')
    axes[1, 0].grid(True, which='both')

    # ── Energy Efficiency ─────────────────────────────────────────────────────
    axes[1, 1].plot(height_range, Energy_Eff_list, 'b-o', markersize=3, label='General')
    axes[1, 1].plot(height_range, Energy_Eff_list_wor, 'r--', markersize=3, label='Worst')
    axes[1, 1].plot(height_range, Energy_Eff_list_bes, 'g--', markersize=3, label='Best')
    axes[1, 1].set_xlabel('UAV Height (m)')
    axes[1, 1].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
    axes[1, 1].set_title('Energy Efficiency vs UAV Height')
    axes[1, 1].grid(True)

    # ── Baseline SINR ─────────────────────────────────────────────────────────
    axes[1, 2].plot(height_range, SINR_list_baseline, 'y-o', markersize=3, label='Direct Link (No UAV)')
    axes[1, 2].set_xlabel('UAV Height (m)')
    axes[1, 2].set_ylabel('Mean SINR (Linear Scale)')
    axes[1, 2].set_title('Baseline SINR vs UAV Height')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()



print_results2()
#print_test_1()
#print_test_2()
