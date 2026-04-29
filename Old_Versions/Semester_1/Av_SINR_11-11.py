import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Working out the average SINR at different recived powers (from 1 to 10). 
# Interference is simplified to one variable (X) which is modelled as a random unoform distrabutio between 0 and 1.

def plot_histogram(values):
    # Plot histogram
    data = np.array(values)
    length = len(data)
    var_name = get_var_name(values, globals())

    plt.figure(figsize=(10, 6))
    # Plot bars
    #plt.bar(range(1, length + 1), data, alpha=0.7, color='lightblue', edgecolor='blue', label='Average Values')
    # Plot line on top
    plt.plot(range(1, length + 1), data, 'bo-', linewidth=2, markersize=8, label='Trend Line')
    plt.xlabel('Transmit Power P')
    plt.ylabel(var_name)
    plt.grid(True, alpha=0.3)
    plt.show()

def get_var_name(var, namespace):
    """Get variable name as string"""
    return [name for name, value in namespace.items() if value is var][0]


# how many rand values and how many times SINR is calculated to find avg
N = 100
# SINR is evaluated at every number between 1 and Numb_of_P
Numb_of_P = 100


# X is the variation of the interference between wanted signal power and unwanted signal power
X = []
i = 0
for i in range(N):
    X_rand = np.random.uniform(0,1)
    X.append(X_rand)

X = np.array(X)


avg_SINR = []
BER = [] # Bit Error Rate
Shannon = []
Outage = []
SINR_array = []

# Working out the average SINR
# stepping through all values of p
Numb_of_P = Numb_of_P + 1
for P in range(1, Numb_of_P):

    # average SINR at a value of P
    SINR = 0
    for n in range(N):
        sinr_at_n = (P / (1 + X[n] * P))
        SINR = SINR + sinr_at_n
        SINR_array.append(sinr_at_n)


    
    avg_SINR.append(SINR / N)


    BER.append(np.sqrt(SINR_array)) # wrong
    Shannon.append(np.log2(1 + (P / (1 + X[n] * P))))

    # outage!!



#plot_histogram(avg_SINR)
#plot_histogram(BER)
plot_histogram(Shannon)