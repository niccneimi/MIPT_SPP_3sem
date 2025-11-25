import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import numba as nb
import math
import time
from plot_metrics import plot_all_metrics

from metrics import MetricsCollector

W, H = 10.0, 6.0
N = 100000
radius = 0.005
mass = 1.0
temperature = 1.0
dt = 0.004
frames = 400
fps = 200
cell_size = radius * 3.0
steps_per_frame = 3
np.random.seed(1)

COLOR_BY_VELOCITY = True  # False для отключения раскраски по скорости

ncols = int(W // cell_size) + 1
nrows = int(H // cell_size) + 1
num_cells = ncols * nrows


@nb.njit(fastmath=True)
def init_positions():
    pos = np.empty((N, 2))
    for i in range(N):
        pos[i, 0] = np.random.uniform(radius, W - radius)
        pos[i, 1] = np.random.uniform(radius, H - radius)
    return pos


def init_velocities():
    v = np.random.normal(0, 1, (N, 2))
    cur_mean_v2 = np.mean(np.sum(v ** 2, axis=1))
    scale = math.sqrt(2 * temperature / cur_mean_v2)
    v *= scale
    return v


positions = init_positions()
velocities = init_velocities()

MAX_PER_CELL = 20
cells = np.full((num_cells, MAX_PER_CELL), -1, dtype=np.int32)
cell_counts = np.zeros(num_cells, dtype=np.int32)


@nb.njit(fastmath=True)
def clear_cells(cell_counts, cells):
    cell_counts[:] = 0
    cells[:] = -1


@nb.njit(fastmath=True)
def build_cells(pos, cells, cell_counts, ncols, nrows, cell_size, W, H):
    N = pos.shape[0]
    for i in range(N):
        ci = int(pos[i, 0] / cell_size)
        cj = int(pos[i, 1] / cell_size)
        if ci < 0:
            ci = 0
        if cj < 0:
            cj = 0
        if ci >= ncols:
            ci = ncols - 1
        if cj >= nrows:
            cj = nrows - 1
        idx = ci + cj * ncols
        count = cell_counts[idx]
        if count < cells.shape[1]:
            cells[idx, count] = i
            cell_counts[idx] = count + 1


@nb.njit(fastmath=True)
def resolve_pairwise(pos, vel, i, j, radius, mass):
    rijx = pos[j, 0] - pos[i, 0]
    rijy = pos[j, 1] - pos[i, 1]
    dist2 = rijx * rijx + rijy * rijy
    min_d = 2.0 * radius
    if dist2 < min_d * min_d and dist2 > 0.0:
        dist = math.sqrt(dist2)
        nx = rijx / dist
        ny = rijy / dist
        dvx = vel[j, 0] - vel[i, 0]
        dvy = vel[j, 1] - vel[i, 1]
        vn = dvx * nx + dvy * ny
        if vn < 0.0:
            imp = -2.0 * vn / (1.0 / mass + 1.0 / mass)
            ix = imp * nx
            iy = imp * ny
            vel[i, 0] -= ix / mass
            vel[i, 1] -= iy / mass
            vel[j, 0] += ix / mass
            vel[j, 1] += iy / mass


@nb.njit(fastmath=True)
def wall_collisions(pos, vel, radius, W, H):
    for i in range(pos.shape[0]):
        if pos[i, 0] < radius:
            pos[i, 0] = radius + (radius - pos[i, 0])
            vel[i, 0] *= -1
        if pos[i, 0] > W - radius:
            pos[i, 0] = (W - radius) - (pos[i, 0] - (W - radius))
            vel[i, 0] *= -1
        if pos[i, 1] < radius:
            pos[i, 1] = radius + (radius - pos[i, 1])
            vel[i, 1] *= -1
        if pos[i, 1] > H - radius:
            pos[i, 1] = (H - radius) - (pos[i, 1] - (H - radius))
            vel[i, 1] *= -1


@nb.njit(fastmath=True, parallel=False)
def step_numba(pos, vel, radius, mass,
               cells, cell_counts, ncols, nrows,
               cell_size, W, H, dt):
    N = pos.shape[0]
    for i in range(N):
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt

    wall_collisions(pos, vel, radius, W, H)

    clear_cells(cell_counts, cells)
    build_cells(pos, cells, cell_counts, ncols, nrows, cell_size, W, H)

    for cj in range(nrows):
        for ci in range(ncols):
            idx = ci + cj * ncols
            countA = cell_counts[idx]
            if countA == 0:
                continue
            for dj in (-1, 0, 1):
                nj = cj + dj
                if nj < 0 or nj >= nrows:
                    continue
                for di in (-1, 0, 1):
                    ni = ci + di
                    if ni < 0 or ni >= ncols:
                        continue
                    nidx = ni + nj * ncols
                    countB = cell_counts[nidx]
                    if countB == 0:
                        continue
                    for a in range(countA):
                        i = cells[idx, a]
                        for b in range(countB):
                            j = cells[nidx, b]
                            if i < 0 or j < 0 or i >= N or j >= N:
                                continue
                            if i < j:
                                resolve_pairwise(pos, vel, i, j, radius, mass)


def kinetic_energy(v):
    return 0.5 * mass * np.sum(v * v)


step_numba(positions.copy(), velocities.copy(), radius, mass,
           cells.copy(), cell_counts.copy(),
           ncols, nrows, cell_size, W, H, dt)

metrics = MetricsCollector(W, H, dt)

t0 = time.time()
for step in range(50):
    step_numba(positions, velocities, radius, mass,
               cells, cell_counts,
               ncols, nrows, cell_size, W, H, dt)
t1 = time.time()
print(f"≈{(t1 - t0) / 50:.6f} sec/step  → {1 / ((t1 - t0) / 50):.1f} steps/s")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.set_title(f"Молекулы газа (N={N})", fontsize=12)

if COLOR_BY_VELOCITY:
    speeds = np.sqrt(np.sum(velocities ** 2, axis=1))
    scat = ax.scatter(positions[:, 0], positions[:, 1], s=(radius * 2000),
                      c=speeds, cmap='viridis', edgecolors='none', alpha=0.7)
    cbar = plt.colorbar(scat, label='Скорость')
else:
    scat = ax.scatter(positions[:, 0], positions[:, 1], s=(radius * 2000),
                      c='blue', edgecolors='none', alpha=0.7)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

step_counter = 0


def animate(frame):
    global step_counter
    for _ in range(steps_per_frame):
        step_numba(positions, velocities, radius, mass,
                   cells, cell_counts,
                   ncols, nrows, cell_size, W, H, dt)
        step_counter += 1

    current_time = step_counter * dt
    metrics.update(positions, velocities, radius, current_time)

    scat.set_offsets(positions)

    if COLOR_BY_VELOCITY:
        speeds = np.sqrt(np.sum(velocities ** 2, axis=1))
        scat.set_array(speeds)

    ke = kinetic_energy(velocities) / N
    time_text.set_text(f"Время: {current_time:.2f}\nKE={ke:.3f}")
    return scat, time_text


anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / fps, blit=True)
plt.show()

plot_all_metrics(metrics, velocities, mass, dt)
