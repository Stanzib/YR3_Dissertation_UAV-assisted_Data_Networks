# 3D model a host within an Base Stations (BS) range
# There is a large urban area
# We increase the amount of UAVs within the BS reuse distance
# Downlink

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# ------- Models Map ----------------------------------------
# [dc, 0] = x, [dc, 1] = y, [dc, 2] = z
BS_Pos = [[0, 200, 0],   # [0, dc] = BS0 coordinates
          [200 , 200, 0], # [1, dc] = BS1 coordinates
          [100, 100, 0],  # BS2 failed
          [0, 0, 0],
          [200, 0, 0]]
R_max = 50
servicing_BS = 2

# ------- Performance Metric variables ------------------------
P_tx_BS = 10
P_tx_UAV = 2
PathLossExponent_urban = 2.5 # typical urban enviroment beta = 2 to 4
PathLossExponent_UAV = 2 # free space
P_UAV_flight = 150
Outage_threshold = 1
K_factor = 5 # for ricain fading. 
relay_mode     = 'AF'   # 'DF' = decode and forward or 'AF' = amplify and forward

# ____Results list____
SINR_list = []
Capacity_list = []
BER_list = []
Outage_list = []
Energy_Eff_list = []

SINR_list_BL = []
Capacity_list_BL = []
BER_list_BL = []
Outage_list_BL = []
Energy_Eff_list_BL = []


# ------- Help Functions ----------------------------------------
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

def random_pos(is_UAV):
    theta = 2 * np.pi * np.random.uniform()
    if is_UAV == False:
        r     = R_max
        return [BS_Pos[2][0] + r * np.cos(theta),
                BS_Pos[2][1] + r * np.sin(theta),
                0]
    else:
        r     = R_max/2
        return [BS_Pos[2][0] + r * np.cos(theta),
                BS_Pos[2][1] + r * np.sin(theta),
                10]
    
def Noise():
    return np.random.uniform(0.001, 1)

def e2e_sinr(sinr1, sinr2, mode='DF'):
    if mode == 'DF':
        return min(sinr1, sinr2)
    else:  # AF
        return (sinr1 * sinr2) / (sinr1 + sinr2 + 1)
    
def performance_metrics(sinr, is_uav):
        global SINR_sum
        global Capacity_sum  
        global BER_sum 
        global outage_sum 
        global Energy_Eff_sum
        global SINR_sum_BL
        global Capacity_sum_BL 
        global BER_sum_BL
        global outage_sum_BL 
        global Energy_Eff_sum_BL

        BER = 0.5 * special.erfc(np.sqrt(sinr))
        
        if is_uav == True:
            Capacity = 0.5 * np.log2(1 + sinr) # half-duplex penalty
            Energy_Eff = Capacity / P_total_UAV
            if sinr < Outage_threshold:
                outage_sum += 1
            SINR_sum += sinr
            Capacity_sum += Capacity
            Energy_Eff_sum += Energy_Eff
            BER_sum += BER
        else:
            Capacity = np.log2(1 + sinr)
            Energy_Eff = Capacity / P_tx_BS
            if sinr < Outage_threshold:
                outage_sum_BL += 1
            SINR_sum_BL += sinr
            Capacity_sum_BL += Capacity
            Energy_Eff_sum_BL += Energy_Eff
            BER_sum_BL += BER

def shadowing_gain():
    rand = np.random.uniform()
    if rand > 0.2:
        return 0.01
    else:
        return 1

    

# =========== Monte Carlo ==========================================
#____Variables_______
UAV_pos_list = [] # [x][0] gives UAVx x-coordinate, [x][1] gives UAVx y-coordinate, [x][2] gives UAVx z-coordinate
num_UAVs_total = 30
num_iterations = 10_000

# -------- Each UAV -----------------------------------------
for num_UAV in range(1, num_UAVs_total + 1):

    #____Initalising____
    SINR_sum = 0
    Capacity_sum = 0 
    BER_sum = 0
    outage_sum = 0
    Energy_Eff_sum = 0

    SINR_sum_BL = 0
    Capacity_sum_BL = 0 
    BER_sum_BL = 0
    outage_sum_BL = 0
    Energy_Eff_sum_BL = 0

    print("starting UAV: ", num_UAV, "\n")

    #____Giving random loaction of UAV_____
    UAV_Pos_cart = random_pos(True)
    UAV_pos_list.append(UAV_Pos_cart)

    # ------- Each Indervidual Host -------------------------
    for j in range(num_iterations):
        
        #___Initalising___
        signal_strength_UAV_Host = []
        interf_BS_Host = []

        #____Random Host Location___
        Host_Pos_cart = random_pos(False)

        #____Closest UAV to Host_____
        dists = [np.sqrt((Host_Pos_cart[0]-u[0])**2 +
                         (Host_Pos_cart[1]-u[1])**2 +
                         (Host_Pos_cart[2]-u[2])**2) for u in UAV_pos_list]
        best  = int(np.argmin(dists))
        d_uav_host = dists[best]
        best_uav   = UAV_pos_list[best]

        # ------ Hop 1: BS to UAV (Rician A2G)--------------
        d_bs_uav = np.sqrt((BS_Pos[servicing_BS][0] - best_uav[0])**2 +
                           (BS_Pos[servicing_BS][1] - best_uav[1])**2 +
                           (BS_Pos[servicing_BS][2] - best_uav[2])**2)
        
        signal1 = P_tx_BS * rician_gain(K_factor)[0] * path_loss(d_bs_uav, PathLossExponent_UAV)

        #______ interference from other BS______
        interf1 = sum(
            P_tx_BS * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)
            for k, u in enumerate(BS_Pos) if k != servicing_BS
        )

        SINR1 = signal1 / (interf1 + Noise())

        # --------- Hop 2: UAV to Host (Rician A2G)-----------------
        signal2 = P_tx_UAV * rician_gain(K_factor)[0] * path_loss(d_uav_host, PathLossExponent_UAV)

        #____ interferance from other UAVs_____
        interf2 = sum(
            P_tx_UAV * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((Host_Pos_cart[0]-u[0])**2 +
                        (Host_Pos_cart[1]-u[1])**2 +
                        (Host_Pos_cart[2]-u[2])**2), PathLossExponent_UAV)
            for k, u in enumerate(UAV_pos_list) if k != best
        )

        SINR2 = signal2 / (interf2 + Noise())

        # ------ Baseline: BS to host (rayleigh)--------------
        d_bs_host = np.sqrt((BS_Pos[servicing_BS][0] - Host_Pos_cart[0])**2 +
                           (BS_Pos[servicing_BS][1] - Host_Pos_cart[1])**2 +
                           (BS_Pos[servicing_BS][2] - Host_Pos_cart[2])**2)
        
        signal3 = P_tx_BS * rayleigh_gain() * shadowing_gain() * path_loss(d_bs_host, PathLossExponent_urban)

        #______ interference from other BS______
        interf3 = sum(
            P_tx_BS * rayleigh_gain() * path_loss(
                np.sqrt((Host_Pos_cart[0]-u[0])**2 +
                        (Host_Pos_cart[1]-u[1])**2 +
                        (Host_Pos_cart[2]-u[2])**2), PathLossExponent_urban)
            for k, u in enumerate(BS_Pos) if k != servicing_BS
        )

    

    

        #---------- Calculating performance metrics--------------
        SINR_e2e = e2e_sinr(SINR1, SINR2, relay_mode)
        SINR_baseline = signal3 / (interf3 + Noise())

        if SINR_e2e > SINR_baseline:
            SINR_Best = SINR_e2e
        else:
            SINR_Best = SINR_baseline



        #_____ Power consumption _______
        P_total_UAV  = P_tx_BS + num_UAV * (P_tx_UAV + P_UAV_flight)
        performance_metrics(SINR_Best, True)
        performance_metrics(SINR_baseline, False)

   
    

    SINR_list.append(SINR_sum / num_iterations)
    Capacity_list.append(Capacity_sum / num_iterations)
    Outage_list.append(outage_sum / num_iterations)
    BER_list.append(BER_sum / num_iterations)
    Energy_Eff_list.append(Energy_Eff_sum / num_iterations)

    SINR_list_BL.append(SINR_sum_BL / num_iterations)
    Capacity_list_BL.append(Capacity_sum_BL / num_iterations)
    Outage_list_BL.append(outage_sum_BL / num_iterations)
    BER_list_BL.append(BER_sum_BL / num_iterations)
    Energy_Eff_list_BL.append(Energy_Eff_sum_BL / num_iterations)
    


def print_results():
    uav_range = range(1, num_UAVs_total + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('UAV Relay Performance Metrics vs Number of UAVs', fontsize=14)

    # ── SINR ─────────────────────────────────────────────────────────────────
    axes[0, 0].plot(uav_range, SINR_list, 'b-o', markersize=3)
    axes[0, 0].plot(uav_range, SINR_list_BL, 'r-o', markersize=3)
    axes[0, 0].set_xlabel('Number of UAVs')
    axes[0, 0].set_ylabel('Mean SINR (linear)')
    axes[0, 0].set_title('Mean SINR')
    axes[0, 0].grid(True)

    # ── Capacity ──────────────────────────────────────────────────────────────
    axes[0, 1].plot(uav_range, Capacity_list, 'b-o', markersize=3)
    axes[0, 1].plot(uav_range, Capacity_list_BL, 'r-o', markersize=3)
    axes[0, 1].set_xlabel('Number of UAVs')
    axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    axes[0, 1].set_title('Channel Capacity')
    axes[0, 1].grid(True)

    # ── Outage ────────────────────────────────────────────────────────────────
    axes[0, 2].plot(uav_range, Outage_list, 'b-o', markersize=3)
    axes[0, 2].plot(uav_range, Outage_list_BL, 'r-o', markersize=3)
    axes[0, 2].set_xlabel('Number of UAVs')
    axes[0, 2].set_ylabel('Outage Probability')
    axes[0, 2].set_title('Outage Probability')
    axes[0, 2].grid(True)

    # ── BER ───────────────────────────────────────────────────────────────────
    axes[1, 0].semilogy(uav_range, BER_list, 'b-o', markersize=3)
    axes[1, 0].semilogy(uav_range, BER_list_BL, 'r-o', markersize=3)
    axes[1, 0].set_xlabel('Number of UAVs')
    axes[1, 0].set_ylabel('Bit Error Rate (BER)')
    axes[1, 0].set_title('BER (log scale)')
    axes[1, 0].grid(True, which='both')

    # ── Energy Efficiency ─────────────────────────────────────────────────────
    axes[1, 1].plot(uav_range, Energy_Eff_list, 'b-o', markersize=3)
    axes[1, 1].plot(uav_range, Energy_Eff_list_BL, 'r-o', markersize=3)
    axes[1, 1].set_xlabel('Number of UAVs')
    axes[1, 1].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
    axes[1, 1].set_title('Energy Efficiency')
    axes[1, 1].grid(True)

    # ── hide the unused 6th subplot ───────────────────────────────────────────
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig('uav_performance_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()




print_results()
