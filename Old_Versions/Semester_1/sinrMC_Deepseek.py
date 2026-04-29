import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class SINRModel:
    def __init__(self, carrier_freq=2.4e9, bandwidth=20e6, temperature=290):
        """
        Initialize SINR model parameters
        
        Parameters:
        carrier_freq: Carrier frequency in Hz
        bandwidth: System bandwidth in Hz
        temperature: Noise temperature in Kelvin
        """
        self.carrier_freq = carrier_freq
        self.bandwidth = bandwidth
        self.temperature = temperature
        self.boltzmann_constant = 1.38e-23  # Boltzmann constant
        
    def path_loss(self, distance, reference_distance=1.0, path_loss_exponent=3.0,
                 shadow_fading_std=8.0, include_shadowing=True):
        """
        Calculate path loss using log-distance model
        
        Parameters:
        distance: Distance between transmitter and receiver (meters)
        reference_distance: Reference distance (meters)
        path_loss_exponent: Path loss exponent (2-6, typical: 3-4)
        shadow_fading_std: Standard deviation of shadow fading (dB)
        include_shadowing: Whether to include shadow fading
        """
        # Free space path loss at reference distance
        lambda_wave = 3e8 / self.carrier_freq  # Wavelength
        free_space_loss = (4 * np.pi * reference_distance / lambda_wave) ** 2
        
        # Log-distance path loss
        if distance <= reference_distance:
            path_loss_db = 10 * np.log10(free_space_loss)
        else:
            path_loss_db = 10 * np.log10(free_space_loss) + \
                          10 * path_loss_exponent * np.log10(distance / reference_distance)
        
        # Add shadow fading
        if include_shadowing:
            shadow_fading = np.random.normal(0, shadow_fading_std)
            path_loss_db += shadow_fading
            
        return path_loss_db
    
    def thermal_noise_power(self):
        """Calculate thermal noise power"""
        return self.boltzmann_constant * self.temperature * self.bandwidth
    
    def calculate_received_power(self, tx_power, tx_gain, rx_gain, distance, 
                               frequency=None, **path_loss_kwargs):
        """
        Calculate received power considering path loss and antenna gains
        """
        if frequency is None:
            frequency = self.carrier_freq
            
        # Calculate path loss
        path_loss_db = self.path_loss(distance, **path_loss_kwargs)
        path_loss_linear = 10 ** (path_loss_db / 10)
        
        # Calculate received power
        received_power = tx_power * tx_gain * rx_gain / path_loss_linear
        
        return received_power
    
    def calculate_sinr(self, desired_signal_params, interferers_params, 
                      noise_figure=3.0):
        """
        Calculate SINR for a desired signal with multiple interferers
        
        Parameters:
        desired_signal_params: Dictionary with parameters for desired signal
        interferers_params: List of dictionaries with parameters for interferers
        noise_figure: Receiver noise figure in dB
        """
        # Calculate desired signal power
        desired_power = self.calculate_received_power(**desired_signal_params)
        
        # Calculate interference power (sum of all interferers)
        interference_power = 0
        for interferer in interferers_params:
            interference_power += self.calculate_received_power(**interferer)
        
        # Calculate noise power
        thermal_noise = self.thermal_noise_power()
        noise_power = thermal_noise * (10 ** (noise_figure / 10))
        
        # Calculate SINR
        sinr_linear = desired_power / (interference_power + noise_power)
        sinr_db = 10 * np.log10(sinr_linear)
        
        return {
            'sinr_db': sinr_db,
            'sinr_linear': sinr_linear,
            'signal_power': desired_power,
            'interference_power': interference_power,
            'noise_power': noise_power
        }

# Example usage
def example_sinr_simulation():
    # Initialize SINR model
    sinr_model = SINRModel(carrier_freq=2.4e9, bandwidth=20e6)
    
    # Desired signal parameters
    desired_signal = {
        'tx_power': 1.0,  # 1 Watt (30 dBm)
        'tx_gain': 1.0,   # 0 dBi
        'rx_gain': 1.0,   # 0 dBi
        'distance': 100,  # 100 meters
        'path_loss_exponent': 3.5,
        'shadow_fading_std': 8.0
    }
    
    # Interferers parameters
    interferers = [
        {
            'tx_power': 0.1,   # 100 mW (20 dBm)
            'tx_gain': 1.0,
            'rx_gain': 1.0,
            'distance': 50,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8.0
        },
        {
            'tx_power': 0.05,  # 50 mW (17 dBm)
            'tx_gain': 1.0,
            'rx_gain': 1.0,
            'distance': 75,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8.0
        }
    ]
    
    # Calculate SINR
    result = sinr_model.calculate_sinr(desired_signal, interferers, noise_figure=3.0)
    
    print("SINR Simulation Results:")
    print(f"SINR: {result['sinr_db']:.2f} dB")
    print(f"Signal Power: {10*np.log10(result['signal_power']):.2f} dBW")
    print(f"Interference Power: {10*np.log10(result['interference_power']):.2f} dBW")
    print(f"Noise Power: {10*np.log10(result['noise_power']):.2f} dBW")
    
    return result

# Run the example
#example_sinr_simulation()


def monte_carlo_sinr_analysis():
    """Perform Monte Carlo analysis of SINR with random positions"""
    
    sinr_model = SINRModel()
    num_simulations = 1000
    sinr_values = []
    
    for i in range(num_simulations):
        # Random distances for desired signal and interferers
        desired_distance = np.random.uniform(50, 200)
        interferer1_distance = np.random.uniform(30, 150)
        interferer2_distance = np.random.uniform(40, 180)
        
        desired_signal = {
            'tx_power': 1.0,
            'tx_gain': 1.0,
            'rx_gain': 1.0,
            'distance': desired_distance,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8.0
        }
        
        interferers = [
            {
                'tx_power': 0.1,
                'tx_gain': 1.0,
                'rx_gain': 1.0,
                'distance': interferer1_distance,
                'path_loss_exponent': 3.5,
                'shadow_fading_std': 8.0
            },
            {
                'tx_power': 0.05,
                'tx_gain': 1.0,
                'rx_gain': 1.0,
                'distance': interferer2_distance,
                'path_loss_exponent': 3.5,
                'shadow_fading_std': 8.0
            }
        ]
        
        result = sinr_model.calculate_sinr(desired_signal, interferers)
        sinr_values.append(result['sinr_db'])
    
    # Analyze results
    sinr_values = np.array(sinr_values)
    
    print(f"Monte Carlo SINR Analysis ({num_simulations} simulations):")
    print(f"Mean SINR: {np.mean(sinr_values):.2f} dB")
    print(f"Std SINR: {np.std(sinr_values):.2f} dB")
    print(f"Min SINR: {np.min(sinr_values):.2f} dB")
    print(f"Max SINR: {np.max(sinr_values):.2f} dB")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sinr_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('SINR (dB)')
    plt.ylabel('Frequency')
    plt.title('SINR Distribution - Monte Carlo Simulation')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return sinr_values

# Run Monte Carlo analysis
sinr_values = monte_carlo_sinr_analysis()



# input locations of antenna
# declare which is desired signal
# work out distances from antenna at random points
# output values from analysis above
# output a plotted surface
