# 2D model a host within an Base Stations (BS) range
# Downlink
# Finding SINR as a function of fading (random), distance function, power, and noise.
# SNR is power over noise
# 3 base stations one at 0, another at 1, the last at 2
# a host is at a random distance within the 2nd base station reuse distance. between {-0.5, 0.5}

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


BS1Pos = 0
BS2Pos = 1
BS3Pos = 2
PathLossExponent = 3 # typical urban enviroment beta = 2 to 4
num_iterations = 10000
num_power = 100

SINR_list = []
SNR_list = []
Capacity_list = []
BER_list = []
for i in range(1, num_power + 1):

    PowerTransmitted = i
    SINR_sum = 0
    SNR_sum = 0
    Capacity_sum = 0
    BER_sum = 0
    #HostPos = np.random.uniform(-0.5, 0.5)

    for j in range(num_iterations):
        
        HostPos = np.random.uniform(0.1, 0.5)
        x = np.random.uniform(-1, 1)
        if x < 0:
            HostPos = HostPos * -1

        Fading1 = np.random.gamma(1, 1)
        Fading2 = np.random.gamma(1, 1)
        Fading3 = np.random.gamma(1, 1)
        Noise = np.random.uniform(0.5, 1)

        SNR = PowerTransmitted / Noise

        dist_to_BS1 = abs(BS2Pos - BS1Pos) + HostPos
        dist_to_BS3 = abs(BS2Pos - BS3Pos) - HostPos
        SINR = (Fading2 * (abs(HostPos))**(-PathLossExponent)) / ((Fading1 * dist_to_BS1**(-PathLossExponent)) + (Fading3 * dist_to_BS3**(-PathLossExponent)) + (1/SNR) )

        Capacity = np.log2(1 + SINR)
        BER = 0.5 * special.erfc(np.sqrt(SINR))

        Capacity_sum += Capacity
        BER_sum += BER

        if SINR > 0:
            SINR_sum += SINR

        SNR_sum += SNR



    SINR_mean = SINR_sum / num_iterations
    SNR_mean = SNR_sum / num_iterations
    Capacity_mean = Capacity_sum / num_iterations
    BER_mean = BER_sum / num_iterations



    SINR_list.append(SINR_mean)
    SNR_list.append(SNR_mean)
    Capacity_list.append(Capacity_mean)
    BER_list.append(BER_mean)




# Create figure with 2x2 subplots
plt.figure(figsize=(14, 10))

# Subplot 1: SINR vs Transmitted Power
plt.subplot(2, 2, 1)
plt.plot(range(1, num_power + 1), SINR_list, 'b-')
plt.xlabel('Transmitted Power')
plt.ylabel('Mean SINR')
plt.title('SINR vs Transmitted Power')
plt.grid(True)

# Subplot 2: SNR vs Transmitted Power
plt.subplot(2, 2, 2)
plt.plot(range(1, num_power + 1), SNR_list, 'r-')
plt.xlabel('Transmitted Power')
plt.ylabel('Mean SNR')
plt.title('SNR vs Transmitted Power')
plt.grid(True)

# Subplot 3: BER vs SNR
plt.subplot(2, 2, 3)
plt.semilogy(SNR_list, BER_list, 'g-')
plt.xlabel('Mean SNR')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR')
plt.grid(True)

# Subplot 4: Capacity vs SNR
plt.subplot(2, 2, 4)
plt.plot(SNR_list, Capacity_list, 'm-')
plt.xlabel('Mean SNR')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Channel Capacity vs SNR')
plt.grid(True)



# Additional plot showing BER on log scale with SNR in dB
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
SNR_dB = 10 * np.log10(SNR_list)
plt.semilogy(SNR_dB, BER_list, 'g-')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR (dB)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(SNR_dB, Capacity_list, 'm-')
plt.xlabel('SNR (dB)')
plt.ylabel('Channel Capacity (bits/s/Hz)')
plt.title('Channel Capacity vs SNR (dB)')
plt.grid(True)

plt.tight_layout()
plt.show()

