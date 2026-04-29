# 2D model a host within an Base Stations (BS) range
# Downlink
# Finding SINR as a function of fading (random), distance function, power, and noise.
# SNR is power over noise
# 3 base stations one at 0, another at 1, the last at 2
# a host is at a random distance within the 2nd base station reuse distance. between {-0.5, 0.5}

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# [dc, 0] = x, [dc, 1] = y, [dc, 2] = z
BS_Pos = [[0, 20, 0],  # [0, dc] = BS0 coordinates
          [20 , 20, 0],
          [10, 10, 0], # BS2 is wanted signal
          [0, 0, 0],
          [20, 0, 0]]
R_max = 5
UAV_pos = [0, 0, 1]

interference = []

#Free Space: 2
#Urban/Cellular: 2 - 4
#Indoor/Obstructed: 4 -6
#Body Area Networks: 4- 7

PathLossExponent_urban = 2.5 # typical urban enviroment beta = 2 to 4
PathLossExponent_UAV = 2 # free space

num_iterations = 1000
num_power = 1000

SINR_list = []
SINR_UAV_list = []
SNR_list = []
Capacity_list = []
Capacity_UAV_list = []
BER_list = []
BER_UAV_list = []

for i in range(1, num_power + 1):

    PowerTransmitted = i
    SNR_sum = 0

    SINR_sum = 0
    SINR_UAV_sum = 0
    Capacity_sum = 0 
    Capacity_UAV_sum = 0
    BER_sum = 0
    BER_UAV_sum = 0

    for j in range(num_iterations):
        
        # host position compared to servicing BS
        rand = np.random.uniform(0.1, 1)
        Host_R = R_max * np.sqrt(rand)
        Host_theta = 2*np.pi * np.random.uniform(0,1)
        # host position in world frame
        Host_Pos_cart = [BS_Pos[2][0] + Host_R * np.cos(Host_theta), BS_Pos[2][1] + Host_R * np.sin(Host_theta)]
        
        Noise = np.random.uniform(0.001, 1)
        SNR = PowerTransmitted / Noise


        
        # interferers for SINR
        for i_row in range(len(BS_Pos)):
            dist = np.sqrt((Host_Pos_cart[0] - BS_Pos[i_row][0])**2 +
                           (Host_Pos_cart[1] - BS_Pos[i_row][1])**2 )
            fading = np.random.gamma(1, 1)
            inter = fading * dist ** (-PathLossExponent_urban)
            interference.append(inter) 
        
        Fading = np.random.gamma(1, 1)
        # Servicing BS is BS2. all other BS are interferers
        SINR = (Fading * (Host_R)**(-PathLossExponent_urban)) / (interference[0] + interference[1] + interference[3] + interference[4] + (1/SNR) )
        SINR_UAV = (Fading * (UAV_pos[2])**(-PathLossExponent_UAV)) / (interference[0] + interference[1] + interference[3] + interference[4] + (1/SNR) )
        
        Capacity = np.log2(1 + SINR)
        Capacity_UAV = np.log2(1 + SINR_UAV)
        
        BER = 0.5 * special.erfc(np.sqrt(SINR))
        BER_UAV = 0.5 * special.erfc(np.sqrt(SINR_UAV))

        SNR_sum += SNR
        SINR_sum += SINR
        SINR_UAV_sum += SINR_UAV
        Capacity_sum += Capacity
        Capacity_UAV_sum += Capacity_UAV
        BER_sum += BER
        BER_UAV_sum += BER_UAV

    SINR_mean = SINR_sum / num_iterations
    SINR_UAV_mean = SINR_UAV_sum / num_iterations
    SNR_mean = SNR_sum / num_iterations
    Capacity_mean = Capacity_sum / num_iterations
    Capacity_UAV_mean = Capacity_UAV_sum / num_iterations
    BER_mean = BER_sum / num_iterations
    BER_UAV_mean = BER_UAV_sum / num_iterations

    SINR_list.append(SINR_mean)
    SINR_UAV_list.append(SINR_UAV_mean)
    SNR_list.append(SNR_mean)
    Capacity_list.append(Capacity_mean)
    Capacity_UAV_list.append(Capacity_UAV_mean)
    BER_list.append(BER_mean)
    BER_UAV_list.append(BER_UAV_mean)


def print_results():
    # Create figure with 2x2 subplots
    plt.figure(figsize=(14, 10))

    # Subplot 1: SINR vs Transmitted Power
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_power + 1), SINR_list, 'b-', label='Regular Host (Urban)')
    plt.plot(range(1, num_power + 1), SINR_UAV_list, 'r-', label='UAV (Free Space)')
    plt.xlabel('Transmitted Power')
    plt.ylabel('Mean SINR')
    plt.title('SINR vs Transmitted Power')
    plt.legend()
    plt.grid(True)

    # Subplot 2: SNR vs Transmitted Power
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_power + 1), SNR_list, 'm-')
    plt.xlabel('Transmitted Power')
    plt.ylabel('Mean SNR')
    plt.title('SNR vs Transmitted Power')
    plt.legend()
    plt.grid(True)

    # Subplot 3: BER vs SNR
    plt.subplot(2, 2, 3)
    plt.semilogy(SNR_list, BER_list, 'b-', label='Regular Host (Urban)')
    plt.semilogy(SNR_list, BER_UAV_list, 'r-', label='UAV (Free Space)')
    plt.xlabel('Mean SNR')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR')
    plt.legend()
    plt.grid(True)

    # Subplot 4: Capacity vs SNR
    plt.subplot(2, 2, 4)
    plt.plot(SNR_list, Capacity_list, 'b-', label='Regular Host (Urban)')
    plt.plot(SNR_list, Capacity_UAV_list, 'r-', label='UAV (Free Space)')
    plt.xlabel('Mean SNR')
    plt.ylabel('Channel Capacity (bits/s/Hz)')
    plt.title('Channel Capacity vs SNR')
    plt.legend()
    plt.grid(True)



    # Additional plot showing BER on log scale with SNR in dB
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    SNR_dB = 10 * np.log10(SNR_list)
    plt.semilogy(SNR_dB, BER_list, 'b-', label='Regular Host (Urban)')
    plt.semilogy(SNR_dB, BER_UAV_list, 'r-', label='UAV (Free Space)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(SNR_dB, Capacity_list, 'b-', label='Regular Host (Urban)')
    plt.plot(SNR_dB, Capacity_UAV_list, 'r-', label='UAV (Free Space)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Channel Capacity (bits/s/Hz)')
    plt.title('Channel Capacity vs SNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def print_test():
    plt.plot(range(1, num_power + 1), SINR_list, 'b-', label='Regular Host (Urban)')
    plt.plot(range(1, num_power + 1), SINR_UAV_list, 'r-', label='UAV (Free Space)')
    plt.xlabel('Transmitted Power')
    plt.ylabel('Mean SINR')
    plt.title('SINR vs Transmitted Power')
    plt.legend()
    plt.grid(True)
    plt.show()

print_results()
#print_test()