import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class AreaSpectralEfficiency:
    def __init__(self, R=200, R_o=20, R_u=4, N_I=6, W=1e6, 
                 a=2, b=2, fc=900e6, h_BS=10, h_m=2):
        """
        Initialize ASE simulation parameters
        
        Parameters:
        R: Cell radius (m)
        R_o: Minimum distance from BS (m)
        R_u: Normalized reuse distance
        N_I: Number of interferers
        W: Total bandwidth (Hz)
        a, b: Path loss exponents
        fc: Carrier frequency (Hz)
        h_BS: BS antenna height (m)
        h_m: Mobile antenna height (m)
        """
        self.R = R
        self.R_o = R_o
        self.R_u = R_u
        self.N_I = N_I
        self.W = W
        self.a = a
        self.b = b
        self.fc = fc
        self.h_BS = h_BS
        self.h_m = h_m
        
        # Calculate break point distance
        self.lambda_c = 3e8 / fc
        self.g = (4 * h_BS * h_m) / self.lambda_c
        
        # Reuse distance
        self.D = R_u * R
        
    def two_slope_path_loss(self, r):
        """Two-slope path loss model - Equation (1)"""
        return 1 / (r**self.a * (1 + r/self.g)**self.b)
    
    def generate_user_position(self, num_users):
        """Generate user positions (polar coordinates) - Equations (7-8)"""
        u = np.random.uniform(0, 1, num_users)
        v = np.random.uniform(0, 1, num_users)
        
        # Generate distance according to Equation (17)
        r = self.R_o + (self.R - self.R_o) * np.sqrt(u)
        # Generate angle
        theta = 2 * np.pi * v
        
        return r, theta
    
    def calculate_interferer_distance(self, x_i, theta_i):
        """Calculate interferer distance to target BS - Equation (20)"""
        r_i = np.sqrt(self.D**2 + x_i**2 + 2 * self.D * x_i * np.sin(theta_i))
        return r_i
    
    def calculate_CIR(self, r_d, r_i_list):
        """Calculate Carrier-to-Interference Ratio - Equation (6)"""
        # Desired user signal power
        S_d = self.two_slope_path_loss(r_d)
        
        # Total interference power
        S_I = 0
        for r_i in r_i_list:
            S_I += self.two_slope_path_loss(r_i)
        
        # CIR
        gamma_d = S_d / S_I if S_I > 0 else float('inf')
        return gamma_d
    
    def calculate_ASE_single_realization(self, r_d, gamma_d):
        """Calculate ASE for single realization - Equation (21)"""
        if np.isinf(gamma_d):
            return 0
        
        # Shannon capacity
        C = np.log2(1 + gamma_d)
        
        # ASE calculation
        A_e = (4 / (np.pi * self.R_u**2 * self.R**2)) * C
        return A_e
    


    def monte_carlo_simulation(self, num_iterations=10000, verbose=True):
        """
        Main Monte Carlo simulation function
        
        Parameters:
        num_iterations: Number of Monte Carlo iterations
        verbose: Whether to show progress
        """
        ASE_list = []
        
        for i in range(num_iterations):
            if verbose and i % 1000 == 0:
                print(f"Progress: {i}/{num_iterations}")
            
            # 1. Generate desired user position
            r_d, theta_d = self.generate_user_position(1)
            r_d = r_d[0]
            
            # 2. Generate interferer positions
            r_i_positions, theta_i_positions = self.generate_user_position(self.N_I)
            
            # 3. Calculate interferer distances to target BS
            r_i_list = []
            for j in range(self.N_I):
                r_i = self.calculate_interferer_distance(r_i_positions[j], theta_i_positions[j])
                r_i_list.append(r_i)
            
            # 4. Calculate CIR
            gamma_d = self.calculate_CIR(r_d, r_i_list)
            
            # 5. Calculate ASE
            ASE = self.calculate_ASE_single_realization(r_d, gamma_d)
            ASE_list.append(ASE)
        
        return np.array(ASE_list)
    
    def monte_carlo_sim_case(self, num_iterations=10000, verbose=True, bestcase = True):
        """
        Main Monte Carlo simulation function
        
        Parameters:
        num_iterations: Number of Monte Carlo iterations
        verbose: Whether to show progress
        """
        ASE_list = []
        
        for i in range(num_iterations):
            if verbose and i % 1000 == 0:
                print(f"Progress: {i}/{num_iterations}")
            
            # 1. Generate desired user position
            r_d, theta_d = self.generate_user_position(1)
            r_d = r_d[0]
            
            # 2. Generate interferer positions
            r_i_positions, theta_i_positions = self.generate_user_position(self.N_I)
            
            # 3. Calculate interferer distances to target BS
            r_i_list = []
            for j in range(self.N_I):
                if bestcase == True:
                    r_i = self.D + self.R
                else:
                    r_i = self.D - self.R
                r_i_list.append(r_i)
            
            # 4. Calculate CIR
            gamma_d = self.calculate_CIR(r_d, r_i_list)
            
            # 5. Calculate ASE
            ASE = self.calculate_ASE_single_realization(r_d, gamma_d)
            ASE_list.append(ASE)
        
        return np.array(ASE_list)
    
    def MCS_Shannon(self, num_iterations=10000, verbose=True):
        """
        Main Monte Carlo simulation function
        
        Parameters:
        num_iterations: Number of Monte Carlo iterations
        verbose: Whether to show progress
        """
        shannon_list = []
        
        for i in range(num_iterations):
            if verbose and i % 1000 == 0:
                print(f"Progress: {i}/{num_iterations}")
            
            # 1. Generate desired user position
            r_d, theta_d = self.generate_user_position(1)
            r_d = r_d[0]
            
            # 2. Generate interferer positions
            r_i_positions, theta_i_positions = self.generate_user_position(self.N_I)
            
            # 3. Calculate interferer distances to target BS
            r_i_list = []
            for j in range(self.N_I):
                r_i = self.calculate_interferer_distance(r_i_positions[j], theta_i_positions[j])
                r_i_list.append(r_i)
            
            # 4. Calculate CIR
            gamma_d = self.calculate_CIR(r_d, r_i_list)
        
            
            shannon_list.append(gamma_d)
        
        return np.array(shannon_list)
    

    def simulate_with_shadowing(self, num_iterations=10000, sigma_dB=4):
        """ASE simulation with lognormal shadowing"""
        ASE_list = []
        
        for i in range(num_iterations):
            if i % 1000 == 0:
                print(f"Shadowing simulation progress: {i}/{num_iterations}")
            
            # Generate user positions
            r_d, theta_d = self.generate_user_position(1)
            r_d = r_d[0]
            
            r_i_positions, theta_i_positions = self.generate_user_position(self.N_I)
            
            # Calculate average received power (path loss only)
            PL_d = self.two_slope_path_loss(r_d)
            
            PL_i_list = []
            for j in range(self.N_I):
                r_i = self.calculate_interferer_distance(r_i_positions[j], theta_i_positions[j])
                PL_i = self.two_slope_path_loss(r_i)
                PL_i_list.append(PL_i)
            
            # Add lognormal shadowing
            # Convert to dB
            PL_d_dB = 10 * np.log10(PL_d)
            shadow_d_dB = np.random.normal(0, sigma_dB)
            S_d_dB = PL_d_dB + shadow_d_dB
            S_d = 10**(S_d_dB / 10)
            
            S_I = 0
            for PL_i in PL_i_list:
                PL_i_dB = 10 * np.log10(PL_i)
                shadow_i_dB = np.random.normal(0, sigma_dB)
                S_i_dB = PL_i_dB + shadow_i_dB
                S_i = 10**(S_i_dB / 10)
                S_I += S_i
            
            gamma_d = S_d / S_I if S_I > 0 else float('inf')
            ASE = self.calculate_ASE_single_realization(r_d, gamma_d)
            ASE_list.append(ASE)
        
        return np.array(ASE_list)
    
    def simulate_with_nakagami_fading(self, num_iterations=10000, m_d=1, m_I=1):
        """ASE simulation with Nakagami fading"""
        ASE_list = []
        
        for i in range(num_iterations):
            if i % 1000 == 0:
                print(f"Nakagami fading simulation progress: {i}/{num_iterations}")
            
            # Generate user positions
            r_d, theta_d = self.generate_user_position(1)
            r_d = r_d[0]
            
            r_i_positions, theta_i_positions = self.generate_user_position(self.N_I)
            
            # Calculate local mean powers (path loss)
            Omega_d = self.two_slope_path_loss(r_d)
            
            Omega_i_list = []
            for j in range(self.N_I):
                r_i = self.calculate_interferer_distance(r_i_positions[j], theta_i_positions[j])
                Omega_i = self.two_slope_path_loss(r_i)
                Omega_i_list.append(Omega_i)
            
            # Generate Nakagami fading
            # Desired user signal (Nakagami-m distributed amplitude)
            shape_d = m_d
            scale_d = Omega_d / m_d
            S_d = np.random.gamma(shape_d, scale_d)
            
            # Interferers
            S_I = 0
            for Omega_i in Omega_i_list:
                shape_i = m_I
                scale_i = Omega_i / m_I
                S_i = np.random.gamma(shape_i, scale_i)
                S_I += S_i
            
            gamma_d = S_d / S_I if S_I > 0 else float('inf')
            ASE = self.calculate_ASE_single_realization(r_d, gamma_d)
            ASE_list.append(ASE)
        
        return np.array(ASE_list)

def run_ASE_reuse():
     
    # Test different reuse distances
    R_u_values = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    results = {}

    print("Starting comprehensive Area Spectral Efficiency analysis...")

    for R_u in R_u_values:
        print(f"\n--- Simulating R_u = {R_u} ---")
        ase_model = AreaSpectralEfficiency(R=200, R_u=R_u)
        
        # simulation (path loss only)
        ASE_basic = ase_model.monte_carlo_simulation(num_iterations=5000, verbose=True)
        # best case interferer positions
        ASE_bestcase = ase_model.monte_carlo_sim_case(num_iterations=5000, verbose=True, bestcase=True)
        # worst case interferer positions
        ASE_worstcase = ase_model.monte_carlo_sim_case(num_iterations=5000, verbose=True, bestcase=False)
        
        
        results[R_u] = {
            'basic_mean': np.mean(ASE_basic),
            'basic_std': np.std(ASE_basic),
            'best_mean': np.mean(ASE_bestcase),
            'best_std': np.std(ASE_bestcase),
            'worst_mean': np.mean(ASE_worstcase),
            'worst_std': np.std(ASE_worstcase),
        }
        
    """Plot comprehensive results"""
    R_u_values = list(results.keys())
    basic_means = [results[ru]['basic_mean'] for ru in R_u_values]
    best_means = [results[ru]['best_mean'] for ru in R_u_values]
    worst_means = [results[ru]['worst_mean'] for ru in R_u_values]

    plt.figure(figsize=(12, 8))

    plt.plot(R_u_values, basic_means, 'bo--', label='Path Loss Only', linewidth=2, markersize=8)
    plt.plot(R_u_values, best_means, 'r-', label='Best Case', linewidth=2, markersize=8)
    plt.plot(R_u_values, worst_means, 'g-', label='Worst Case', linewidth=2, markersize=8)

    plt.xlabel('Normalized Reuse Distance R_u')
    plt.ylabel('Area Spectral Efficiency (bps/Hz/m²)')
    plt.title('Area Spectral Efficiency vs Reuse Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()


    plt.show()

def run_BER():
    #want to add a best case and a worst case

    # Test different reuse distances
    W_values = [0, 0.5e6, 2e6, 2.5e6, 3e6, 3.5e6, 4e6, 4.5e6, 5e6]
    results = {}

    print("Starting comprehensive Area Spectral Efficiency analysis...")

    for W in W_values:
        print(f"\n--- Simulating W = {W} ---")
        ase_model = AreaSpectralEfficiency(R=200, W=W)
        
        # Basic simulation (path loss only)
        shannon_basic = ase_model.MCS_Shannon(num_iterations=5000, verbose=True)
        #ASE_bestcase = ase_model.monte_carlo_sim_case(num_iterations=5000, verbose=True, bestcase=True)
        #ASE_worstcase = ase_model.monte_carlo_sim_case(num_iterations=5000, verbose=True, bestcase=False)
        
        
        results[W] = {
            'basic_mean': np.mean(shannon_basic),
            'basic_std': np.std(shannon_basic),
            #'best_mean': np.mean(ASE_bestcase),
            #'best_std': np.std(ASE_bestcase),
            #'worst_mean': np.mean(ASE_worstcase),
            #'worst_std': np.std(ASE_worstcase),
        }
        
    """Plot comprehensive results"""
    W_values = list(results.keys())
    basic_means = [np.log2(1 + results[w]['basic_mean']) for w in W_values]
    #best_means = [np.log2(1 + results[w]['best_mean']) for w in W_values]
    #worst_means = [np.log2(1 + results[w]['worst_mean']) for w in W_values]


    plt.figure(figsize=(12, 8))

    plt.plot(W_values, basic_means, 'bo--', label='Path Loss Only', linewidth=2, markersize=8)
    #plt.plot(W_values, best_means, 'r-', label='Best Case', linewidth=2, markersize=8)
    #plt.plot(W_values, worst_means, 'g-', label='Worst Case', linewidth=2, markersize=8)

    plt.xlabel('Bandwidth W')
    plt.ylabel('BER')
    plt.title('Bit Error Rate vs Bandwidth')
    plt.grid(True, alpha=0.3)
    plt.legend()


    plt.show()


if __name__ == "__main__":

    run_ASE_reuse()
    #run_BER()