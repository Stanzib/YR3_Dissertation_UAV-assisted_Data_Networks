import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
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
    

    # need to calculate host and antenna distnace
    def calculate_received_power(self, x_host, y_host, rx_gain, tx_power, tx_gain, x_pos, y_pos, reuse, transmitting, 
                               frequency=None, **path_loss_kwargs):
        """
        Calculate received power considering path loss and antenna gains
        """
        if frequency is None:
            frequency = self.carrier_freq
            
        distance = np.sqrt((x_pos - x_host)**2 + (y_pos - y_host)**2)

        # Calculate path loss
        path_loss_db = self.path_loss(distance, **path_loss_kwargs)
        path_loss_linear = 10 ** (path_loss_db / 10)
        
        # Calculate received power
        received_power = tx_power * tx_gain * rx_gain / path_loss_linear
        
        return received_power
    
    def calculate_sinr(self, antenna_params, host_pos, 
                      noise_figure=3.0):
        """
        Calculate SINR for a desired signal with multiple interferers
        
        Parameters:
        antenna_params: Dictionary with parameters for all antenna
        desired_signal: which antenna is producing the desired signal
        noise_figure: Receiver noise figure in dB
        """
        # Calculate desired and interference power
        desired_power = 0
        interference_power = 0

        for antenna in antenna_params:
            combined_params = {**host_pos, **antenna}
            power = self.calculate_received_power(**combined_params)

            if antenna.get('transmitting', False) == True:
                desired_power += power
            elif antenna.get('transmitting', False) == False:
                interference_power += power
            else:
                print('failed calculating desired and interference power')
        
       
        
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

class AntennaMap:
    def __init__(self):
        pass


    def generate_circle_points(self, h, k, r, num_points=100):
        """
        Generate points on a circle
        (h, k) is the centre of the cirlce
        r is the radius
        """
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = h + r * np.cos(theta)
        y = k + r * np.sin(theta)
        return x, y
    
    def plot_map(self, antenna_variables):
        """Plot antenna positions and reuse distances"""
        plt.figure(figsize=(10, 8))
        
        # Plot each antenna and its reuse distance circle
        for i, antenna in enumerate(antenna_variables):
            x_circle, y_circle = self.generate_circle_points(
                antenna['x_pos'], 
                antenna['y_pos'], 
                antenna['reuse']
            )
            plt.plot(x_circle, y_circle, 'b-', alpha=0.5, linewidth=1.5)
            plt.plot(antenna['x_pos'], antenna['y_pos'], 'ro', markersize=10, 
                    label=f'Antenna {i+1}')
            
            # Add antenna label
            plt.annotate(f'Ant {i+1}\n({antenna["x_pos"]},{antenna["y_pos"]})', 
                        (antenna['x_pos'], antenna['y_pos']),
                        textcoords="offset points",
                        xytext=(0,15),
                        ha='center',
                        fontsize=9)
        
        
        # Set plot properties
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title("Antenna Deployment Map with Reuse Distances")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()


# input locations of antenna
# declare which is desired signal
# work out distances from antenna at set point
# output values from analysis

def sinr_simulation1(antenna_var):
    
    # Initialize SINR model
    sinr_model = SINRModel(carrier_freq=2.4e9, bandwidth=20e6)

    # Define Host
    # position is going to be random for monte carlo
    pos_host = {
        'x_host':50,
        'y_host':50,
        'rx_gain': 1
    }
    
    
    # Calculate SINR
    
    result = sinr_model.calculate_sinr(antenna_var, pos_host, noise_figure=3.0)
    
    print("SINR Simulation Results:")
    print(f"SINR: {result['sinr_db']:.2f} dB")
    print(f"Signal Power: {10*np.log10(result['signal_power']):.2f} dBW")
    print(f"Interference Power: {10*np.log10(result['interference_power']):.2f} dBW")
    print(f"Noise Power: {10*np.log10(result['noise_power']):.2f} dBW")
    
    return result



def monte_carlo_sinr_analysis(antenna_var):
    """Perform Monte Carlo analysis of SINR with random positions"""
    
    sinr_model = SINRModel()
    num_simulations = 1000
    
    #for plot
    x_positions = []
    y_positions = []
    sinr_values = []
    
    for i in range(num_simulations):
        # Random distances for desired signal and interferers
        host_x_rand = np.random.uniform(0, 100)
        host_y_rand = np.random.uniform(0, 100)
        
        
        host_pos = {
            'x_host': host_x_rand,
            'y_host': host_y_rand,
            'rx_gain': 1
        }
     
        
        result = sinr_model.calculate_sinr(antenna_var, host_pos)
    
        # Store positions and SINR
        x_positions.append(host_x_rand)
        y_positions.append(host_y_rand)
        sinr_values.append(result['sinr_db'])

    # Convert to numpy arrays
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    sinr_values = np.array(sinr_values) 
    
    print(f"Monte Carlo SINR Analysis ({num_simulations} simulations):")
    print(f"Mean SINR: {np.mean(sinr_values):.2f} dB")
    print(f"Std SINR: {np.std(sinr_values):.2f} dB")
    print(f"Min SINR: {np.min(sinr_values):.2f} dB")
    print(f"Max SINR: {np.max(sinr_values):.2f} dB")
    
    # Plot histogram
 #   plt.figure(figsize=(10, 6))
  #  plt.hist(sinr_values, bins=50, alpha=0.7, edgecolor='black')
   # plt.xlabel('SINR (dB)')
   # plt.ylabel('Frequency')
   # plt.title('SINR Distribution - Monte Carlo Simulation')
   # plt.grid(True, alpha=0.3)
   # plt.show()

    return x_positions, y_positions, sinr_values


def plot_sinr_3d_surface(x_positions, y_positions, sinr_values):
    """Create a 3D surface plot of SINR vs position similar to the example"""
    
    # Create grid for surface plot
    grid_size = 50
    xi = np.linspace(0, 100, grid_size)
    yi = np.linspace(0, 100, grid_size)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate SINR values onto the grid
    Z = griddata((x_positions, y_positions), sinr_values, (X, Y), method='cubic')
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')
    
    # Plot the 3D surface (similar to your example)
    surf = ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=2, cstride=2,
                        alpha=0.7, cmap='viridis')
    
    # Plot projections of the contours on the walls
    z_min = np.nanmin(Z) - 5
    z_max = np.nanmax(Z) + 5

    ax.contour(X, Y, Z, zdir='z', offset=z_min, cmap='coolwarm', levels=10)
    ax.contour(X, Y, Z, zdir='x', offset=0, cmap='coolwarm', levels=10)
    ax.contour(X, Y, Z, zdir='y', offset=100, cmap='coolwarm', levels=10)
    
    # Set axis limits and labels
    ax.set(xlim=(0, 100), ylim=(0, 100), zlim=(z_min, z_max),
        xlabel='X Position (m)', ylabel='Y Position (m)', zlabel='SINR (dB)')
    
    ax.set_title('SINR Distribution vs Host Position (3D Surface)')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='SINR (dB)')
    
    plt.tight_layout()


    """Create a 3D scatter plot of SINR vs position"""
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')
    
    # Create scatter plot colored by SINR value
    scatter = ax.scatter(x_positions, y_positions, sinr_values, 
                        c=sinr_values, cmap='viridis', 
                        s=20, alpha=0.6)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('SINR (dB)')
    ax.set_title('SINR vs Host Position (3D Scatter)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('SINR (dB)')
    
    plt.tight_layout()

    plt.show()
    
    return X, Y, Z
        
      

antenna_map = AntennaMap()
antenna_var = [
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 50,
            'y_pos': 50, ## center
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': True
        },
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 50,
            'y_pos': 80,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': False
        },
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 50,
            'y_pos': 20,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': False
        },
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 76,
            'y_pos': 65,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': False
        },
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 76,
            'y_pos': 35,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': False
        },
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 24,
            'y_pos': 35,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': False
        },
        {
            'tx_power': 1.0,  # 1 Watt (30 dBm)
            'tx_gain': 10.0,   # 0 dBi
            'x_pos': 24,
            'y_pos': 65,
            'path_loss_exponent': 3.5,
            'shadow_fading_std': 8,
            'reuse': 15,
            'transmitting': False
        },
    ]

antenna_map.plot_map(antenna_var)

#sinr_simulation1(antenna_var)


x_pos, y_pos, sinr_vals = monte_carlo_sinr_analysis(antenna_var)

plot_sinr_3d_surface(x_pos, y_pos, sinr_vals)



