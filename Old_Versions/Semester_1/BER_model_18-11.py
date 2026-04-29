import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import convolve

class BERModel:
    """
    A comprehensive Bit Error Rate (BER) model for wireless communication systems
    """
    
    def __init__(self):
        self.modulation_schemes = {
            'BPSK': 1,
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6
        }
    
    def awgn_channel(self, tx_signal, snr_db):
        """
        Add Additive White Gaussian Noise to the transmitted signal
        
        Parameters:
        tx_signal: transmitted signal
        snr_db: Signal-to-Noise Ratio in dB
        
        Returns:
        rx_signal: received signal with noise
        """
        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db / 10)
        
        # Calculate signal power
        signal_power = np.mean(np.abs(tx_signal)**2)
        
        # Calculate noise power
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power/2) * (np.random.randn(*tx_signal.shape) + 
                                        1j * np.random.randn(*tx_signal.shape))
        
        return tx_signal + noise
    
    def generate_bits(self, num_bits):
        """Generate random binary data"""
        return np.random.randint(0, 2, num_bits)
    
    def modulate_bpsk(self, bits):
        """BPSK Modulation: 0 -> +1, 1 -> -1"""
        return 1 - 2 * bits
    
    def demodulate_bpsk(self, rx_signal):
        """BPSK Demodulation"""
        return (rx_signal.real < 0).astype(int)
    
    def modulate_qpsk(self, bits):
        """QPSK Modulation"""
        # Reshape bits into symbols
        symbols = bits.reshape(-1, 2)
        
        # Map to constellation points
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): 1 - 1j,
            (1, 0): -1 + 1j,
            (1, 1): -1 - 1j
        }
        
        modulated = np.array([mapping[tuple(symbol)] for symbol in symbols])
        return modulated / np.sqrt(2)  # Normalize power
    
    def demodulate_qpsk(self, rx_signal):
        """QPSK Demodulation"""
        decisions = []
        for symbol in rx_signal:
            if symbol.real >= 0:
                if symbol.imag >= 0:
                    decisions.extend([0, 0])
                else:
                    decisions.extend([0, 1])
            else:
                if symbol.imag >= 0:
                    decisions.extend([1, 0])
                else:
                    decisions.extend([1, 1])
        return np.array(decisions)
    
    def theoretical_ber_bpsk(self, snr_db):
        """Theoretical BER for BPSK in AWGN"""
        snr_linear = 10**(snr_db / 10)
        return 0.5 * special.erfc(np.sqrt(snr_linear))
    
    def theoretical_ber_qpsk(self, snr_db):
        """Theoretical BER for QPSK in AWGN"""
        # QPSK has same BER as BPSK but requires 3dB more power for same performance
        return self.theoretical_ber_bpsk(snr_db)
    
    def theoretical_ber_16qam(self, snr_db):
        """Theoretical BER for 16-QAM in AWGN"""
        snr_linear = 10**(snr_db / 10)
        return (3/8) * special.erfc(np.sqrt(0.4 * snr_linear))
    
    def theoretical_ber_64qam(self, snr_db):
        """Theoretical BER for 64-QAM in AWGN"""
        snr_linear = 10**(snr_db / 10)
        return (7/24) * special.erfc(np.sqrt(snr_linear / 21))
    
    def simulate_ber(self, modulation='BPSK', snr_range=range(0, 11), num_bits=100000):
        """
        Simulate BER for given modulation scheme and SNR range
        
        Parameters:
        modulation: Modulation scheme ('BPSK', 'QPSK', '16QAM', '64QAM')
        snr_range: Range of SNR values in dB
        num_bits: Number of bits to simulate
        
        Returns:
        ber_simulated: Simulated BER values
        ber_theoretical: Theoretical BER values
        """
        
        ber_simulated = []
        ber_theoretical = []
        
        for snr_db in snr_range:
            # Generate random bits
            bits = self.generate_bits(num_bits)
            
            # Modulate based on scheme
            if modulation == 'BPSK':
                tx_signal = self.modulate_bpsk(bits)
                rx_signal = self.awgn_channel(tx_signal, snr_db)
                rx_bits = self.demodulate_bpsk(rx_signal)
                theoretical_ber = self.theoretical_ber_bpsk(snr_db)
                
            elif modulation == 'QPSK':
                # Ensure even number of bits for QPSK
                if num_bits % 2 != 0:
                    num_bits += 1
                bits = self.generate_bits(num_bits)
                tx_signal = self.modulate_qpsk(bits)
                rx_signal = self.awgn_channel(tx_signal, snr_db)
                rx_bits = self.demodulate_qpsk(rx_signal)
                theoretical_ber = self.theoretical_ber_qpsk(snr_db)
                
            else:
                raise ValueError(f"Modulation {modulation} not implemented in simulation")
            
            # Calculate BER
            bit_errors = np.sum(bits != rx_bits)
            ber = bit_errors / len(bits)
            
            ber_simulated.append(ber)
            ber_theoretical.append(theoretical_ber)
            
            print(f"SNR: {snr_db:2d} dB, BER: {ber:.6f}, Theoretical: {theoretical_ber:.6f}")
        
        return np.array(ber_simulated), np.array(ber_theoretical)
    
    def plot_ber_curves(self, snr_range=range(0, 16)):
        """Plot BER curves for different modulation schemes"""
        plt.figure(figsize=(12, 8))
        
        # Theoretical curves
        snr_array = np.array(snr_range)
        
        # BPSK/QPSK
        ber_bpsk = self.theoretical_ber_bpsk(snr_array)
        plt.semilogy(snr_array, ber_bpsk, 'b-', linewidth=2, label='BPSK/QPSK Theoretical')
        
        # 16-QAM
        ber_16qam = self.theoretical_ber_16qam(snr_array)
        plt.semilogy(snr_array, ber_16qam, 'r-', linewidth=2, label='16-QAM Theoretical')
        
        # 64-QAM
        ber_64qam = self.theoretical_ber_64qam(snr_array)
        plt.semilogy(snr_array, ber_64qam, 'g-', linewidth=2, label='64-QAM Theoretical')
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('Bit Error Rate vs SNR for Different Modulation Schemes')
        plt.legend()
        plt.ylim(1e-6, 1)
        plt.xlim(min(snr_range), max(snr_range))
        plt.show()
    
    def fading_channel_ber(self, snr_db, fading_type='rayleigh'):
        """
        BER for fading channels
        """
        if fading_type == 'rayleigh':
            # Average BER for Rayleigh fading
            snr_linear = 10**(snr_db / 10)
            return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))
        else:
            raise ValueError("Only Rayleigh fading implemented")

    def plot_comparison(self, modulations=['BPSK', 'QPSK'], snr_range=range(0, 11), num_bits=10000):
        """Compare simulated vs theoretical BER for multiple modulations"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, modulation in enumerate(modulations):
            # Get theoretical values
            snr_array = np.array(snr_range)
            if modulation == 'BPSK':
                ber_theo = self.theoretical_ber_bpsk(snr_array)
            elif modulation == 'QPSK':
                ber_theo = self.theoretical_ber_qpsk(snr_array)
            elif modulation == '16QAM':
                ber_theo = self.theoretical_ber_16qam(snr_array)
            elif modulation == '64QAM':
                ber_theo = self.theoretical_ber_64qam(snr_array)
            
            # Plot theoretical
            plt.semilogy(snr_range, ber_theo, color=colors[i], linestyle='-', 
                        linewidth=2, label=f'{modulation} Theoretical')
            
            # Plot simulated if available
            if modulation in ['BPSK', 'QPSK']:
                ber_sim, _ = self.simulate_ber(modulation, snr_range, num_bits)
                plt.semilogy(snr_range, ber_sim, color=colors[i], marker='o', 
                           linestyle='--', markersize=6, label=f'{modulation} Simulated')
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER Comparison: Simulated vs Theoretical')
        plt.legend()
        plt.ylim(1e-6, 1)
        plt.show()

# Example usage and demonstration
def main():
    # Initialize BER model
    ber_model = BERModel()
    
    print("=== Wireless Communication BER Modeling ===")
    print("\n1. Simulating BPSK BER performance...")
    
    # Simulate BPSK with fewer points for faster execution
    snr_range = range(0, 11, 2)
    ber_sim, ber_theo = ber_model.simulate_ber('BPSK', snr_range, num_bits=50000)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_sim, 'ro-', linewidth=2, markersize=8, label='Simulated BER')
    plt.semilogy(snr_range, ber_theo, 'b--', linewidth=2, label='Theoretical BER')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BPSK Modulation: Simulated vs Theoretical BER')
    plt.legend()
    plt.show()
    
    print("\n2. Comparing different modulation schemes...")
    # Plot comprehensive BER curves
    ber_model.plot_ber_curves(range(0, 16))
    
    # Additional analysis: BER under fading conditions
    print("\n3. BER under Rayleigh Fading...")
    snr_values = [0, 5, 10, 15, 20]
    print("SNR (dB) | AWGN BER    | Rayleigh BER")
    print("-" * 40)
    for snr in snr_values:
        ber_awgn = ber_model.theoretical_ber_bpsk(snr)
        ber_fading = ber_model.fading_channel_ber(snr, 'rayleigh')
        print(f"{snr:8d} | {ber_awgn:.6f}   | {ber_fading:.6f}")
    
    print("\n4. Comprehensive comparison...")
    ber_model.plot_comparison(['BPSK', 'QPSK', '16QAM'], range(0, 16))

# Quick test function
def quick_test():
    """Quick test to verify the BER model works"""
    ber_model = BERModel()
    
    # Quick BER calculation
    snr = 10
    ber = ber_model.theoretical_ber_bpsk(snr)
    print(f"Quick test - BPSK BER at {snr} dB: {ber:.6f}")
    
    # Simple simulation
    bits = ber_model.generate_bits(1000)
    tx_signal = ber_model.modulate_bpsk(bits)
    rx_signal = ber_model.awgn_channel(tx_signal, snr)
    rx_bits = ber_model.demodulate_bpsk(rx_signal)
    errors = np.sum(bits != rx_bits)
    print(f"Simulated errors: {errors}/1000 = {errors/1000:.4f}")

if __name__ == "__main__":
    # Run quick test first
    quick_test()
    print("\n" + "="*50 + "\n")
    
    # Run main demonstration
    main()