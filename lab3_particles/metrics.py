import numpy as np

def mean_kinetic_energy(velocities, mass=1.0):
    ke = 0.5 * mass * np.sum(velocities**2, axis=1)
    return np.mean(ke)

def velocity_distribution(velocities, bins=50):
    speeds = np.linalg.norm(velocities, axis=1)
    return np.histogram(speeds, bins=bins, density=True)


class MSDCalculator:
    def __init__(self, positions_history=None):
        self.positions_history = positions_history if positions_history is not None else []

    def update(self, positions):
        self.positions_history.append(positions.copy())

    def calculate_msd(self, tau_max=None):
        if len(self.positions_history) < 2:
            return np.array([0]), np.array([0])

        if tau_max is None:
            tau_max = len(self.positions_history) // 2

        N_times = len(self.positions_history)
        msd = np.zeros(tau_max)
        counts = np.zeros(tau_max)

        for tau in range(1, min(tau_max, N_times)):
            for t0 in range(N_times - tau):
                dr = self.positions_history[t0 + tau] - self.positions_history[t0]
                msd[tau] += np.mean(np.sum(dr ** 2, axis=1))
                counts[tau] += 1

        msd[counts > 0] /= counts[counts > 0]
        return np.arange(tau_max), msd


class VirialPressureCalculator:
    def __init__(self, W, H, mass=1.0):
        self.W = W
        self.H = H
        self.mass = mass
        self.virial_sum = 0.0
        self.step_count = 0
        self.particle_count = 0

    def update(self, positions, velocities, radius):
        current_virial = 0.0
        N = len(positions)

        kinetic_contrib = 0.0
        for i in range(N):
            kinetic_contrib += self.mass * (velocities[i, 0] ** 2 + velocities[i, 1] ** 2)

        wall_virial = 0.0
        for i in range(N):
            if positions[i, 0] <= radius:
                wall_virial += positions[i, 0]
            if positions[i, 0] >= self.W - radius:
                wall_virial += positions[i, 0] - self.W
            if positions[i, 1] <= radius:
                wall_virial += positions[i, 1]
            if positions[i, 1] >= self.H - radius:
                wall_virial += positions[i, 1] - self.H

        self.virial_sum += kinetic_contrib + wall_virial
        self.step_count += 1
        self.particle_count = N

    def get_pressure(self):
        if self.step_count == 0:
            return 0.0

        average_virial = self.virial_sum / self.step_count
        area = self.W * self.H

        pressure = 0.5 * average_virial / area

        return pressure

    def reset(self):
        self.virial_sum = 0.0
        self.step_count = 0


class MetricsCollector:
    def __init__(self, W, H, dt):
        self.W = W
        self.H = H
        self.dt = dt
        self.msd_calc = MSDCalculator()
        self.pressure_calc = VirialPressureCalculator(W, H, dt)
        self.history = {
            'time': [],
            'kinetic_energy': [],
            'pressure': [],
            'mean_speed': []
        }

    def update(self, positions, velocities, radius, step):
        self.msd_calc.update(positions)
        self.pressure_calc.update(positions, velocities, radius)
        ke = mean_kinetic_energy(velocities)
        pressure = self.pressure_calc.get_pressure()
        mean_speed = np.mean(np.linalg.norm(velocities, axis=1))

        self.history['time'].append(step * self.dt)
        self.history['kinetic_energy'].append(ke)
        self.history['pressure'].append(pressure)
        self.history['mean_speed'].append(mean_speed)

    def get_summary(self):
        summary = {}
        for key, values in self.history.items():
            if len(values) > 0:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
        return summary
