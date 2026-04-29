# 3D model a host within an Base Stations (BS) range
# There is a Ideal urban area
# We increase the height of UAVs within the BS reuse distance
# Uplink
# 6 interfering BS
# best case worst case calculated
print("script started: Increasing Height, Ideal Model")

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from dataclasses import dataclass, field

# ------- Models Map ----------------------------------------
Dist_between_BS = 2500
serving_x = 5000
serving_y = 5000
BS_Height = 25
# hexagon with each corner as a BS, Centre as the serving BS. Listed from top left to right.
# [dc, 0] = x, [dc, 1] = y, [dc, 2] = z
BS_Pos = [
          [serving_x,                                   serving_y + Dist_between_BS,        BS_Height], # [0, dc] = BS0 coordinates
          [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    BS_Height], # [1, dc] = BS1 coordinates
          [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y + Dist_between_BS*0.5,    BS_Height],
          [serving_x,                                   serving_y,                          BS_Height], # Serving BS (3)
          [serving_x - Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    BS_Height],
          [serving_x + Dist_between_BS*(np.sqrt(3)/2),  serving_y - Dist_between_BS*0.5,    BS_Height],
          [serving_x,                                   serving_y - Dist_between_BS,        BS_Height]
        ]

servicing_BS = 3
R_max = Dist_between_BS * 0.5
Dist_to_worst_case = Dist_between_BS - R_max
Dist_to_Best_case = Dist_between_BS + R_max
BS_Height = 25
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
P_tx_UAV = 5 # 2
P_circuit_UAV = 10 #!!! need to find reference for these
P_circuit_host = 5
R_0 = 20 # minimum distance a random position can be to the centre its spawned around
P_UAV_flight = 150
Outage_threshold = 1 # 2.5e-5
relay_mode     = 'AF'   # 'DF' = decode and forward or 'AF' = amplify and forward
#no_uav variable for inc power
# --------- scenario parameters --------------------------
PathLossExponent_urban = 2.5 # typical urban enviroment beta = 2 to 4
PathLossExponent_UAV = 2 # free space
shadowing_angle_cutoff = 30
#UAV_Height = 100
num_UAV = 100
P_tx_Host = 0.2
# no rayleigh fading


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
        r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform(0, 1))
        return [center_x + r * np.cos(theta),
                center_y + r * np.sin(theta),
                0]
    else:
        r     = R_0 + (R_max - R_0) * np.sqrt(np.random.uniform(0, 1))
        return [center_x + r * np.cos(theta),
                center_y + r * np.sin(theta),
                UAV_Height] # how high the UAV hovers
    
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
        Energy_Eff = Capacity / (P_circuit_host + P_tx_Host)

    if sinr < Outage_threshold:
        acc.outage_sum += 1

    acc.SINR_sum       += sinr
    acc.Capacity_sum   += Capacity
    acc.Energy_Eff_sum += Energy_Eff
    acc.BER_sum        += BER

def shadowing_gain(angle_deg=None):
    if angle_deg is None:
        # Ground-to-ground outdoor: use mid-range of [4, 12] dB
        sigma_dB = 12
    elif angle_deg > shadowing_angle_cutoff:
        return 1
    else:
        # A2G: interpolate sigma across [4, 12] dB range
        # At 0° (horizontal, near-NLoS) -> sigma = 12 dB  
        # At 90° (overhead, strong LoS)  -> sigma = 4 dB  
        sigma_dB = 12.0 - (8.0 * (angle_deg / 90.0))
        sigma_dB = np.clip(sigma_dB, 4.0, 12.0)
    # Sample X_sigma ~ N(0, sigma_dB^2)  [the log-normal shadowing term]
    X_sigma_dB = np.random.normal(0, sigma_dB)
    # Convert dB to linear power ratio
    gain = 10 ** (X_sigma_dB / 10)
    return min(gain, 1)

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



# =========== Monte Carlo ==========================================
#____Variables_______
uav_height_total = 150
num_iterations = 1_00 # 10_000 for results
acc_gen = PerfAccumulator()   # general / UAV-relay case
acc_wor = PerfAccumulator()   # worst-case
acc_bes = PerfAccumulator()   # best-case
acc_baseline = PerfAccumulator()

sinr_basline_total = 0

# -------- Each UAV -----------------------------------------
for UAV_Height in range(1, uav_height_total + 1):

    #____Initalising____
    acc_gen.reset()
    acc_wor.reset()
    acc_bes.reset()
    acc_baseline.reset()

    print("starting height: ", UAV_Height, "\n")

    

    # ------- Each Indervidual Host -------------------------
    for j in range(num_iterations):

        #____Reset and regenerate UAV positions each iteration____
        UAV_pos_list = [] # [x][0] gives UAVx x-coordinate, [x][1] gives UAVx y-coordinate, [x][2] gives UAVx z-coordinate
        for _ in range(num_UAV):
            UAV_pos_list.append(random_pos(BS_Pos[servicing_BS][0], BS_Pos[servicing_BS][1], True))

        
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
        # mid_x = (Host_Pos_cart[0] + BS_Pos[servicing_BS][0]) / 2
        # mid_y = (Host_Pos_cart[1] + BS_Pos[servicing_BS][1]) / 2

        # dists_to_mid = [np.sqrt((mid_x - u[0])**2 +
        #                         (mid_y - u[1])**2) for u in UAV_pos_list]

        best = int(np.argmin(dists)) #!!! now using closest UAV not midpoint
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

        #_____angle of UAV to interferers_____
        angles_interf_to_uav_general = [
            elevation_angle(best_uav, u) 
            for u in interf_rand_pos_list
        ]
        angles_interf_to_uav_worst = [
            elevation_angle(best_uav, u) 
            for u in worst_case_interf_list
        ]
        angles_interf_to_uav_best = [
            elevation_angle(best_uav, u) 
            for u in best_case_interf_list
        ]
         #_____angle of BS to interferers_____
        angles_interf_to_BS_general = [
            elevation_angle(BS_Pos[servicing_BS], u) 
            for u in interf_rand_pos_list
        ]
        angles_interf_to_BS_worst = [
            elevation_angle(BS_Pos[servicing_BS], u) 
            for u in worst_case_interf_list
        ]
        angles_interf_to_BS_best = [
            elevation_angle(BS_Pos[servicing_BS], u) 
            for u in best_case_interf_list
        ]


        #____Pathloss and Shadowing (should be constant across cases)____
        # One draw per unique physical link  
        # general      
        fading_shadow_interf_hop1_general = [rician_gain(K_factor_elevation(angles_interf_to_uav_general[i]))[0] * shadowing_gain(angles_interf_to_uav_general[i]) #!!!could add angle calc
                        for i, u in enumerate(interf_rand_pos_list)]              # one per interferer
        fading_shadow_interf_hop2_general = [rician_gain(K_factor_elevation(angles_interf_to_BS_general[i]))[0] * shadowing_gain(angles_interf_to_BS_general[i]) 
                        for i, u in enumerate(interf_rand_pos_list)]              
        #worst
        fading_shadow_interf_hop1_worst = [rician_gain(K_factor_elevation(angles_interf_to_uav_worst[i]))[0] * shadowing_gain(angles_interf_to_uav_worst[i])
                        for i, u in enumerate(worst_case_interf_list)]              
        fading_shadow_interf_hop2_worst = [rician_gain(K_factor_elevation(angles_interf_to_BS_worst[i]))[0] * shadowing_gain(angles_interf_to_BS_worst[i]) 
                        for i, u in enumerate(worst_case_interf_list)]  
        #best
        fading_shadow_interf_hop1_best = [rician_gain(K_factor_elevation(angles_interf_to_uav_best[i]))[0] * shadowing_gain(angles_interf_to_uav_best[i])
                        for i, u in enumerate(best_case_interf_list)]              
        fading_shadow_interf_hop2_best = [rician_gain(K_factor_elevation(angles_interf_to_BS_best[i]))[0] * shadowing_gain(angles_interf_to_BS_best[i]) 
                        for i, u in enumerate(best_case_interf_list)]  
        #desired sig
        fading_shadow_desired_hop1 = rician_gain(K_hop1)[0] * shadowing_gain(angle_uav_host)
        fading_shadow_desired_hop2 = rician_gain(K_hop2)[0]
        #baseline interf
        fading_shadow_interf_baseline = [rician_gain(K_factor_elevation(angles_interf_to_BS_general[i])) * shadowing_gain(angles_interf_to_BS_general[i]) 
                for i, u in enumerate(interf_rand_pos_list)]
        #baseline desired sig
        fading_shadow_desired_baseline = rician_gain(K_factor_elevation(elevation_angle(BS_Pos[servicing_BS], Host_Pos_cart))) * shadowing_gain(elevation_angle(BS_Pos[servicing_BS], Host_Pos_cart))



        #=========== Host -> UAV -> BS general case ======================
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_gen = (P_tx_Host * fading_shadow_desired_hop1 * path_loss(d_uav_host, PathLossExponent_UAV)) / Noise() #bottle neck for SINR
        #____ interferance from other hosts_____
        #!!! aproximate K-factor for interferers
        interf1_gen = sum(
            (P_tx_Host * fading_shadow_interf_hop1_general[i] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for i, u in enumerate(interf_rand_pos_list)
        )
        SINR1_gen = signal1_gen / (interf1_gen + 1)
        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        d_bs_uav = np.sqrt((BS_Pos[servicing_BS][0] - best_uav[0])**2 +
                           (BS_Pos[servicing_BS][1] - best_uav[1])**2 +
                           (BS_Pos[servicing_BS][2] - best_uav[2])**2
                           )
        signal2_gen = (P_tx_UAV * fading_shadow_desired_hop2 * path_loss(d_bs_uav, PathLossExponent_UAV)) / Noise()
        #______ interference from other UAV______
        interf2_gen = sum(
            (P_tx_Host * fading_shadow_interf_hop2_general[i] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for i, u in enumerate(interf_rand_pos_list)
        )
        SINR2_gen = signal2_gen / (interf2_gen + 1)

        


        #============== Host -> UAV -> BS Worst case ==================
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_wor = signal1_gen
        #____ interferance from other hosts_____
        interf1_wor = sum(
            (P_tx_Host * fading_shadow_interf_hop1_worst[i] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for i, u in enumerate(worst_case_interf_list)
        )
        SINR1_wor = signal1_wor / (interf1_wor + 1)
        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        signal2_wor = signal2_gen
        #______ interference from hosts______
        interf2_wor = sum(
            (P_tx_Host * fading_shadow_interf_hop2_worst[i] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for i, u in enumerate(worst_case_interf_list)
        )
        SINR2_wor = signal2_wor / (interf2_wor + 1)



        #============== Host -> UAV -> BS Best case ===================
        # --------- Hop 1: Host to UAV (Rician A2G)-----------------
        signal1_bes = signal1_gen
        #____ interferance from other hosts_____
        interf1_bes = sum(
            (P_tx_Host * fading_shadow_interf_hop1_best[i] * path_loss(
                np.sqrt((best_uav[0]-u[0])**2 +
                        (best_uav[1]-u[1])**2 +
                        (best_uav[2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for i, u in enumerate(best_case_interf_list)
        )
        SINR1_bes = signal1_bes / (interf1_bes + 1)
        # ------ Hop 2: UAV to BS (Rician A2G)--------------
        signal2_bes = signal2_gen
        #______ interference from other UAV______
        interf2_bes = sum(
            (P_tx_Host * fading_shadow_interf_hop2_best[i] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_UAV)) / Noise()
            for i, u in enumerate(best_case_interf_list)
        )
        SINR2_bes = signal2_bes / (interf2_bes + 1)

        
        
        #========= Baseline: Host -> BS (general) ====================
        d_bs_host = np.sqrt((BS_Pos[servicing_BS][0] - Host_Pos_cart[0])**2 +
                           (BS_Pos[servicing_BS][1] - Host_Pos_cart[1])**2 +
                           (BS_Pos[servicing_BS][2] - Host_Pos_cart[2])**2
                           )
        singal_baseline = (P_tx_Host * fading_shadow_desired_baseline * path_loss(d_bs_host, PathLossExponent_urban)) / Noise()
        #______ interference from other Host______
        interf_baseline = sum(
            (P_tx_Host * fading_shadow_interf_baseline[i] * path_loss(
                np.sqrt((BS_Pos[servicing_BS][0]-u[0])**2 +
                        (BS_Pos[servicing_BS][1]-u[1])**2 +
                        (BS_Pos[servicing_BS][2]-u[2])**2), PathLossExponent_urban)) / Noise()
            for i, u in enumerate(interf_rand_pos_list)
        )

    

        #_____ Power consumption _______
        P_total_UAV  = P_tx_UAV + P_tx_Host + P_circuit_host + num_UAV * (P_circuit_UAV + P_UAV_flight)

        #---------- Calculating performance metrics--------------
        SINR_e2e_gen = e2e_sinr(SINR1_gen, SINR2_gen, relay_mode)
        SINR_e2e_wor = e2e_sinr(SINR1_wor, SINR2_wor, relay_mode)
        SINR_e2e_bes = e2e_sinr(SINR1_bes, SINR2_bes, relay_mode)
        SINR_e2e_baseline = singal_baseline / (interf_baseline + 1)
        

        # After computing SINR_e2e_gen
        if UAV_Height == 50:  # just check one UAV count
            print(f"SINR1:{SINR1_gen:.3f}  SINR2:{SINR2_gen:.3f}  e2e:{SINR_e2e_gen:.3f}")

        performance_metrics(SINR_e2e_gen, True, acc_gen)
        performance_metrics(SINR_e2e_wor, True, acc_wor)
        performance_metrics(SINR_e2e_bes, True, acc_bes)
        performance_metrics(SINR_e2e_baseline, False, acc_baseline)

        sinr_basline_total += SINR_e2e_baseline

   
    

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
    
    


def print_results():
    height_range = range(1, uav_height_total + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('UAV Relay Performance Metrics vs UAV Height', fontsize=14)

    axes[0, 0].plot(height_range, SINR_list,     'b-o', markersize=3, label='General')
    axes[0, 0].plot(height_range, SINR_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 0].plot(height_range, SINR_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 0].set_xlabel('UAV Height (m)')
    axes[0, 0].set_ylabel('Mean SINR (linear)')
    axes[0, 0].set_title('Mean SINR')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(height_range, Capacity_list,     'b-o', markersize=3, label='General')
    axes[0, 1].plot(height_range, Capacity_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 1].plot(height_range, Capacity_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 1].set_xlabel('UAV Height (m)')
    axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    axes[0, 1].set_title('Channel Capacity')
    axes[0, 1].grid(True)

    axes[0, 2].plot(height_range, Outage_list,     'b-o', markersize=3, label='General')
    axes[0, 2].plot(height_range, Outage_list_wor, 'r--', markersize=3, label='Worst')
    axes[0, 2].plot(height_range, Outage_list_bes, 'g--', markersize=3, label='Best')
    axes[0, 2].set_xlabel('UAV Height (m)')
    axes[0, 2].set_ylabel('Outage Probability')
    axes[0, 2].set_title('Outage Probability')
    axes[0, 2].grid(True)

    axes[1, 0].semilogy(height_range, BER_list,     'b-o', markersize=3, label='General')
    axes[1, 0].semilogy(height_range, BER_list_wor, 'r--', markersize=3, label='Worst')
    axes[1, 0].semilogy(height_range, BER_list_bes, 'g--', markersize=3, label='Best')
    axes[1, 0].set_xlabel('UAV Height (m)')
    axes[1, 0].set_ylabel('Bit Error Rate (BER)')
    axes[1, 0].set_title('BER (log scale)')
    axes[1, 0].grid(True, which='both')

    axes[1, 1].plot(height_range, Energy_Eff_list,     'b-o', markersize=3, label='General')
    axes[1, 1].plot(height_range, Energy_Eff_list_wor, 'r--', markersize=3, label='Worst')
    axes[1, 1].plot(height_range, Energy_Eff_list_bes, 'g--', markersize=3, label='Best')
    axes[1, 1].set_xlabel('UAV Height (m)')
    axes[1, 1].set_ylabel('Energy Efficiency (bits/s/Hz/W)')
    axes[1, 1].set_title('Energy Efficiency')
    axes[1, 1].grid(True)

    axes[1, 2].plot(height_range, SINR_list_baseline, 'y-o', markersize=3, label='Baseline (no UAV)')
    axes[1, 2].set_xlabel('UAV Height (m)')
    axes[1, 2].set_ylabel('Mean SINR (linear)')
    axes[1, 2].set_title('Baseline SINR (Host → Serving BS, No Relay)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

baseline_average = sinr_basline_total / (num_iterations * uav_height_total)
print(baseline_average)

print_results()

