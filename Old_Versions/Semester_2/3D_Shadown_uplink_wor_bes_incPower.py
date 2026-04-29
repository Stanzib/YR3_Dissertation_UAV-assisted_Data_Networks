# 3D model a host within an Base Stations (BS) range
# There is a large urban area
# We increase the amount of UAVs within the BS reuse distance
# uplink

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from dataclasses import dataclass, field

# ------- Models Map ----------------------------------------
# [dc, 0] = x, [dc, 1] = y, [dc, 2] = z
BS_Pos = [[0, 200, 0],   # [0, dc] = BS0 coordinates
          [200, 200, 0], # [1, dc] = BS1 coordinates
          [100, 100, 0],  # BS2 failed
          [0, 0, 0],
          [200, 0, 0]]
R_max = (np.sqrt(100**2 + 100**2)) / 2
servicing_BS = 2
iAngle = np.radians(45)
best_case_interf_list = [[BS_Pos[0][0] - (R_max * np.cos(iAngle)), BS_Pos[0][1] - (R_max * np.sin(iAngle)), BS_Pos[0][2]],
                          [BS_Pos[1][0] + (R_max * np.cos(iAngle)), BS_Pos[1][1] - (R_max * np.sin(iAngle)), BS_Pos[1][2]],
                          [BS_Pos[3][0] - (R_max * np.cos(iAngle)), BS_Pos[3][1] + (R_max * np.sin(iAngle)), BS_Pos[3][2]],
                          [BS_Pos[4][0] + (R_max * np.cos(iAngle)), BS_Pos[4][1] + (R_max * np.sin(iAngle)), BS_Pos[4][2]]]

worst_case_interf_list =  [[BS_Pos[0][0] + (R_max * np.cos(iAngle)), BS_Pos[0][1] - (R_max * np.sin(iAngle)), BS_Pos[0][2]],
                          [BS_Pos[1][0] - (R_max * np.cos(iAngle)), BS_Pos[1][1] - (R_max * np.sin(iAngle)), BS_Pos[1][2]],
                          [BS_Pos[3][0] + (R_max * np.cos(iAngle)), BS_Pos[3][1] + (R_max * np.sin(iAngle)), BS_Pos[3][2]],
                          [BS_Pos[4][0] - (R_max * np.cos(iAngle)), BS_Pos[4][1] + (R_max * np.sin(iAngle)), BS_Pos[4][2]]]



# ------- Performance Metric variables ------------------------
P_tx_UAV = 10 # 2
#P_tx_Host = 50 # 10
PathLossExponent_urban = 2.5 # typical urban enviroment beta = 2 to 4
PathLossExponent_UAV = 2 # free space
P_UAV_flight = 150
Outage_threshold = 0.0001 # 2.5e-5
K_factor = 10 # for ricain fading, 10 is strong LoS, infinate is perfect LoS. 
relay_mode     = 'AF'   # 'DF' = decode and forward or 'AF' = amplify and forward
num_UAV = 10

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

def random_pos(center_x, center_y, is_UAV):
    theta = 2 * np.pi * np.random.uniform()
    if is_UAV == False:
        r     = R_max * np.random.uniform(0.2, 1)
        return [center_x + r * np.cos(theta),
                center_y + r * np.sin(theta),
                0]
    else:
        r     = R_max/2
        return [center_x + r * np.cos(theta),
                center_y + r * np.sin(theta),
                5]
    
def Noise():
    # need to rework this
    print ("help help help!! normailse noise to unity")
    return np.random.uniform(0.01, 1)

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
        Energy_Eff = Capacity / P_tx_UAV

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

    

# =========== Monte Carlo ==========================================
#____Variables_______
power_range_total = 200
num_iterations = 1_000
acc_gen = PerfAccumulator()   # general / UAV-relay case
acc_wor = PerfAccumulator()   # worst-case
acc_bes = PerfAccumulator()   # best-case
acc_baseline = PerfAccumulator()

# -------- Each UAV -----------------------------------------
for P_tx_Host in range(1, power_range_total + 1):

    #____Initalising____
    acc_gen.reset()
    acc_wor.reset()
    acc_bes.reset()

    print("starting Power: ", P_tx_Host, "\n")

    

    # ------- Each Indervidual Host -------------------------
    for j in range(num_iterations):

        #____Reset and regenerate UAV positions each iteration____
        UAV_pos_list = [] # [x][0] gives UAVx x-coordinate, [x][1] gives UAVx y-coordinate, [x][2] gives UAVx z-coordinate
        for _ in range(num_UAV):
            UAV_pos_list.append(random_pos(BS_Pos[2][0], BS_Pos[2][1], True))

        
        #___Initalising___
        signal_strength_UAV_Host = []
        interf_BS_Host = []

        #____Random Host Location___
        Host_Pos_cart = random_pos(BS_Pos[2][0], BS_Pos[2][1], False)

        #____Closest UAV to Host_____
        dists = [np.sqrt((Host_Pos_cart[0]-u[0])**2 +
                         (Host_Pos_cart[1]-u[1])**2 +
                         (Host_Pos_cart[2]-u[2])**2) for u in UAV_pos_list]
        best  = int(np.argmin(dists))
        d_uav_host = dists[best]
        best_uav   = UAV_pos_list[best]

        
        #____Interfering UD locations_____
        interf_rand_pos_list = []
        for k, u in enumerate(BS_Pos):
            if k != servicing_BS:
                rand_pos = random_pos(u[0], u[1], False)
                interf_rand_pos_list.append(rand_pos)


        #=========== Host -> UAV -> BS general case ======================
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_gen = P_tx_Host * rician_gain(K_factor)[0] * path_loss(d_uav_host, PathLossExponent_UAV)

        #____ interferance from other hosts_____
        interf1_gen = sum(
            P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)
            for u in interf_rand_pos_list
        )

        SINR1_gen = signal1_gen / (interf1_gen + Noise())

        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        d_bs_uav = np.sqrt((BS_Pos[servicing_BS][0] - best_uav[0])**2 +
                           (BS_Pos[servicing_BS][1] - best_uav[1])**2 +
                           (BS_Pos[servicing_BS][2] - best_uav[2])**2)
        
        signal2_gen = P_tx_UAV * rician_gain(K_factor)[0] * path_loss(d_bs_uav, PathLossExponent_UAV)

        #______ interference from other UAV______
        interf2_gen = sum(
            P_tx_UAV * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)
            for k, u in enumerate(UAV_pos_list) if k != best
        )

        SINR2_gen = signal2_gen / (interf2_gen + Noise())

        
        #============== Host -> UAV -> BS Worst case ==================

        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_wor = signal1_gen

        #____ interferance from other hosts_____
        interf1_wor = sum(
            P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)
            for u in worst_case_interf_list
        )

        SINR1_wor = signal1_wor / (interf1_wor + Noise())

        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        signal2_wor = signal2_gen

        #______ interference from other UAV______
        interf2_wor = interf1_gen

        SINR2_wor = signal2_wor / (interf2_wor + Noise())



        #============== Host -> UAV -> BS Best case ===================
        
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_bes = signal1_gen

        #____ interferance from other hosts_____
        interf1_bes = sum(
            P_tx_Host * rician_gain(K_factor)[0] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)
            for u in best_case_interf_list
        )

        SINR1_bes = signal1_bes / (interf1_bes + Noise())

        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        signal2_bes = signal2_gen

        #______ interference from other UAV______
        interf2_bes = interf1_gen

        SINR2_bes = signal2_bes / (interf2_bes + Noise())


    

        

    

        #_____ Power consumption _______
        P_total_UAV  = P_tx_UAV + num_UAV * (P_tx_Host + P_UAV_flight)


        #---------- Calculating performance metrics--------------
        SINR_e2e_gen = e2e_sinr(SINR1_gen, SINR2_gen, relay_mode)
        SINR_e2e_wor = e2e_sinr(SINR1_wor, SINR2_wor, relay_mode)
        SINR_e2e_bes = e2e_sinr(SINR1_bes, SINR2_bes, relay_mode)
        performance_metrics(SINR_e2e_gen, True, acc_gen)
        performance_metrics(SINR_e2e_wor, True, acc_wor)
        performance_metrics(SINR_e2e_bes, True, acc_bes)

   
    

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

    
    


def print_results():
    uav_range = range(1, power_range_total + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('UAV Relay Performance Metrics vs Number of UAVs', fontsize=14)

    # ── SINR ─────────────────────────────────────────────────────────────────
    axes[0, 0].plot(uav_range, SINR_list, 'b-o', markersize=3, label='General')
    axes[0, 0].plot(uav_range, SINR_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 0].plot(uav_range, SINR_list_bes, 'g--', markersize=3, label='Best')
    #axes[0, 0].plot(uav_range, SINR_list_baseline, 'y-o', markersize=3, label='Baseline')
    axes[0, 0].set_xlabel('Number of UAVs')
    axes[0, 0].set_ylabel('Mean SINR (linear)')
    axes[0, 0].set_title('Mean SINR')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # ── Capacity ──────────────────────────────────────────────────────────────
    axes[0, 1].plot(uav_range, Capacity_list, 'b-o', markersize=3)
    axes[0, 1].plot(uav_range, Capacity_list_wor, 'r--', markersize=3)
    axes[0, 1].plot(uav_range, Capacity_list_bes, 'g--', markersize=3)
    axes[0, 1].set_xlabel('Number of UAVs')
    axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    axes[0, 1].set_title('Channel Capacity')
    axes[0, 1].grid(True)

    # ── Outage ────────────────────────────────────────────────────────────────
    axes[0, 2].plot(uav_range, Outage_list, 'b-o', markersize=3)
    axes[0, 2].plot(uav_range, Outage_list_wor, 'r--', markersize=3)
    axes[0, 2].plot(uav_range, Outage_list_bes, 'g--', markersize=3)
    axes[0, 2].set_xlabel('Number of UAVs')
    axes[0, 2].set_ylabel('Outage Probability')
    axes[0, 2].set_title('Outage Probability')
    axes[0, 2].grid(True)

    # ── BER ───────────────────────────────────────────────────────────────────
    axes[1, 0].semilogy(uav_range, BER_list, 'b-o', markersize=3)
    axes[1, 0].semilogy(uav_range, BER_list_wor, 'r--', markersize=3)
    axes[1, 0].semilogy(uav_range, BER_list_bes, 'g--', markersize=3)
    axes[1, 0].set_xlabel('Number of UAVs')
    axes[1, 0].set_ylabel('Bit Error Rate (BER)')
    axes[1, 0].set_title('BER (log scale)')
    axes[1, 0].grid(True, which='both')

    # ── Energy Efficiency ─────────────────────────────────────────────────────
    axes[1, 1].plot(uav_range, Energy_Eff_list, 'b-o', markersize=3)
    axes[1, 1].plot(uav_range, Energy_Eff_list_wor, 'r--', markersize=3)
    axes[1, 1].plot(uav_range, Energy_Eff_list_bes, 'g--', markersize=3)
    axes[1, 1].set_xlabel('Number of UAVs')
    axes[1, 1].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
    axes[1, 1].set_title('Energy Efficiency')
    axes[1, 1].grid(True)

    # -- SINR Baseline ---------------------------------------------------------
    # axes[1, 2].plot(uav_range, SINR_list_baseline, 'y-o', markersize=3, label='Baseline')
    # axes[1, 2].set_xlabel('Number of UAVs')
    # axes[1, 2].set_ylabel('Mean SINR (linear)')
    # axes[1, 2].set_title('Mean SINR')
    # axes[1, 2].legend()
    # axes[1, 2].grid(True)


    # ── hide the unused 6th subplot ───────────────────────────────────────────
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.show()


def print_test_1():
    plt.plot(range(1, power_range_total + 1), Outage_list, 'b-')
    plt.xlabel('Number of UAVs in dead zone')
    plt.ylabel('Outage probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_test_2():
    plt.subplot(1, 2, 1)
    plt.plot(range(1, power_range_total + 1), Outage_list, 'b-')
    plt.xlabel('Number of UAVs in dead zone')
    plt.ylabel('Outage probability')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, power_range_total + 1), SINR_list, 'r-')
    plt.xlabel('Number of UAVs in dead zone')
    plt.ylabel('Mean SINR')
    plt.grid(True)
    plt.show()



print_results()
#print_test_1()
#print_test_2()
