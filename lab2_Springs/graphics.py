import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

class GraphicsRecorder:
    def __init__(self, blocks):
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.positions = []
        self.velocities = []
        self.displacements = [] 
        self.time_steps = []

    def record_step(self, t):
        self.time_steps.append(t)
        pos = np.array([block.position.copy() for block in self.blocks])
        vel = np.array([block.velocity.copy() for block in self.blocks])
        disp = np.linalg.norm(pos - np.array([b.initial_position for b in self.blocks]), axis=1)


        self.positions.append(pos)
        self.velocities.append(vel)
        self.displacements.append(disp)


    def plot_analysis(self):
        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        self.displacements = np.array(self.displacements)
        self.time_steps = np.array(self.time_steps)

        center_index = self.num_blocks // 2

        plt.figure(figsize=(8,4))
        plt.plot(self.time_steps, self.displacements[:, center_index], label="Displacement")
        plt.xlabel("Time [s]")
        plt.ylabel("Displacement")
        plt.title("Central Block Displacement vs Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        n_rows = int(np.sqrt(self.num_blocks))
        n_cols = n_rows
        for t_idx in [0, len(self.time_steps)//2, -1]:
            plt.figure(figsize=(6,5))
            disp_grid = self.displacements[t_idx].reshape(n_rows, n_cols)
            plt.imshow(disp_grid, cmap='viridis', origin='lower')
            plt.colorbar(label="Displacement")
            plt.title(f"Displacement Map at t={self.time_steps[t_idx]:.2f}s")
            plt.show()

        y = self.displacements[:, center_index]
        dt = self.time_steps[1] - self.time_steps[0]
        N = len(y)
        yf = fft(y)
        xf = fftfreq(N, dt)[:N//2]

        plt.figure(figsize=(8,4))
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("FFT of Central Block Displacement")
        plt.grid(True)
        plt.show()
        
        kinetic_energy = 0.5 * np.sum(np.sum(self.velocities**2, axis=2), axis=1)
        plt.figure(figsize=(8,4))
        plt.plot(self.time_steps, kinetic_energy)
        plt.xlabel("Time [s]")
        plt.ylabel("Total Kinetic Energy")
        plt.title("Kinetic Energy of the System")
        plt.grid(True)
        plt.show()
