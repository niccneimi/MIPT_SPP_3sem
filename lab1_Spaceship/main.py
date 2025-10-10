import math
import pygame
import numpy as np
from pygame.locals import *

MU = 3.986004418e14  # Гравитационный параметр Земли (м^3/с^2)
R_EARTH = 6371000.0
STANDART_G = 9.81

# Атмосферные параметры
RHO_0 = 1.225  # плотность воздуха на уровне моря (кг/м^3)
SCALE_HEIGHT = 8500.0  # шкала высоты атмосферы (м)

# ISS
H_ISS = 408000.0  # высота МКС над поверхностью (м)
R_ISS = R_EARTH + H_ISS
OMEGA_ISS = math.sqrt(MU / (R_ISS ** 3))  # угловая скорость для круговой орбиты

# Ракета
INITIAL_ALT = 0.0  # начальная высота ракеты м
R_ROCKET = R_EARTH + INITIAL_ALT
ROCKET_MASS = 10.0e3  # произвольная масса ракеты кг
MAX_THRUST = 4100e3  # максимальная тяга Н
ISP = 300  # Удельный импульс сек
FUEL_MASS = 400e3  # масса топлива кг
DRAG_COEFFICIENT = 0.75  # коэффициент лобового сопротивления
CROSS_SECTIONAL_AREA = 50.0  # площадь поперечного сечения (м²)

DT = 0.2  # шаг интеграции, сек

# pygame
WIDTH, HEIGHT = 1000, 800
CENTER = np.array([WIDTH // 2, HEIGHT // 2])
BG_COLOR = (8, 10, 30)
EARTH_COLOR = (30, 80, 150)
ISS_COLOR = (200, 200, 30)
ROCKET_COLOR = (220, 100, 100)
ORBIT_COLOR = (120, 120, 180)
TRAJ_COLOR = (180, 180, 255)
ATMOSPHERE_COLOR = (30, 150, 200, 50)

# Масштаб (пиксели на метр)
SCALE = 3.58e-5
MIN_SCALE, MAX_SCALE = 2e-7, 5e-4


def world_to_screen(pos, scale, center=CENTER):
    x, y = pos
    sx = center[0] + x * scale
    sy = center[1] - y * scale
    return int(sx), int(sy)


def draw_text(screen, text, x, y, size=18, color=(220, 220, 220)):
    font = pygame.font.SysFont('consolas', size)
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def atmospheric_density(altitude):
    if altitude < 0:
        return RHO_0
    return RHO_0 * math.exp(-altitude / SCALE_HEIGHT)


def drag_force(rocket_vel, altitude):
    speed = np.linalg.norm(rocket_vel)
    if speed == 0:
        return np.array([0.0, 0.0])

    rho = atmospheric_density(altitude)
    drag_magnitude = 0.5 * rho * speed ** 2 * DRAG_COEFFICIENT * CROSS_SECTIONAL_AREA

    drag_direction = -rocket_vel / speed
    return drag_magnitude * drag_direction


class SpaceObject:
    def __init__(self, pos, vel, mass=1000.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.mass = mass

    def grav_acc(self):
        r = np.linalg.norm(self.pos)
        if r == 0:
            return np.array([0.0, 0.0])
        return -MU * self.pos / (r ** 3)


class ISS(SpaceObject):
    def __init__(self, r=R_ISS, phase=0.0):
        pos = np.array([r * math.cos(phase), r * math.sin(phase)])
        speed = math.sqrt(MU / r)
        vel = np.array([-speed * math.sin(phase), speed * math.cos(phase)])
        super().__init__(pos, vel, mass=419725.0)

    def step(self, dt=DT):
        theta = math.atan2(self.pos[1], self.pos[0])
        theta += OMEGA_ISS * dt
        r = np.linalg.norm(self.pos)
        self.pos = np.array([r * math.cos(theta), r * math.sin(theta)])
        speed = math.sqrt(MU / r)
        self.vel = np.array([-speed * math.sin(theta), speed * math.cos(theta)])


class Rocket(SpaceObject):
    def __init__(self, r=R_ROCKET, phase=0.1):
        self.fuel_mass = FUEL_MASS
        self.dry_mass = ROCKET_MASS
        self.mass = self.dry_mass + self.fuel_mass

        pos = np.array([r * math.cos(phase), r * math.sin(phase)])
        speed = 0
        vel = np.array([-speed * math.sin(phase), speed * math.cos(phase)])
        super().__init__(pos, vel, mass=self.mass)
        self.angle = math.atan2(self.vel[1], self.vel[0])
        self.thrust_on = False
        self.throttle = 2.0
        self.traj = []
        self.drag_force_history = []

    def step(self, dt=DT, thrust_vec=None):
        a_grav = self.grav_acc()
        a_thrust = np.array([0.0, 0.0])

        # Вычисление силы сопротивления
        altitude = np.linalg.norm(self.pos) - R_EARTH
        drag_force_vec = drag_force(self.vel, altitude)
        a_drag = drag_force_vec / self.mass if self.mass > 0 else np.array([0.0, 0.0])

        self.drag_force_history.append(np.linalg.norm(drag_force_vec))
        if len(self.drag_force_history) > 100:
            self.drag_force_history.pop(0)

        if thrust_vec is not None and np.linalg.norm(thrust_vec) > 0:
            a_thrust = np.linalg.norm(thrust_vec)
            mdot = a_thrust / (ISP * STANDART_G)
            fuel_used = mdot * dt
            self.fuel_mass = max(0.0, self.fuel_mass - fuel_used)
            self.mass = self.dry_mass + self.fuel_mass
            if self.fuel_mass <= 0:
                thrust_vec = np.array([0.0, 0.0])
            a_thrust = thrust_vec / self.mass

        # Суммарное ускорение
        total_acceleration = a_grav + a_thrust + a_drag
        self.vel += total_acceleration * dt
        self.pos += self.vel * dt
        self.traj.append(self.pos.copy())

        if len(self.traj) > 2000:
            self.traj.pop(0)
        r = np.linalg.norm(self.pos)
        if r < R_EARTH:
            self.pos = self.pos / r * R_EARTH
            self.vel = np.array([0.0, 0.0])

class AutoPilot:
    def __init__(self, max_thrust=MAX_THRUST, target_alt=H_ISS):
        self.max_thrust = max_thrust
        self.target_r = R_EARTH + target_alt
        self.cutoff = False

    def compute_thrust(self, rocket: Rocket, target=None):
        r_vec = rocket.pos
        r = np.linalg.norm(r_vec)
        radial_dir = r_vec / r
        tangential_dir = np.array([-r_vec[1], r_vec[0]]) / r

        v_r = np.dot(rocket.vel, radial_dir)

        if not self.cutoff:
            v_r_cutoff = math.sqrt(max(0.0, 2 * MU * (1/r - 1/self.target_r)))
            if v_r >= v_r_cutoff / 2.5:
                self.cutoff = True

        alpha = 0.8 * (r - R_EARTH) / (self.target_r - R_EARTH)
        alpha = np.clip(alpha, 0.0, 0.8)

        if not self.cutoff:
            thrust_dir = (1 - alpha) * radial_dir + alpha * tangential_dir
        else:
            thrust_dir = tangential_dir

        thrust_dir /= np.linalg.norm(thrust_dir)

        thrust_vec = self.max_thrust * thrust_dir
        return thrust_vec


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption('Симулятор полёта на МКС')

    iss = ISS()
    rocket = Rocket()
    autopilot = AutoPilot()
    autopilot_enabled = False

    running = True
    global SCALE
    global DT

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    rocket.thrust_on = not rocket.thrust_on
                elif event.key == K_a:
                    autopilot_enabled = not autopilot_enabled
                elif event.key == K_r:
                    iss = ISS()
                    rocket = Rocket()
                elif event.key == K_PLUS or event.key == K_EQUALS:
                    SCALE *= 1.2
                    SCALE = min(SCALE, MAX_SCALE)
                elif event.key == K_MINUS:
                    SCALE /= 1.2
                    SCALE = max(SCALE, MIN_SCALE)
                elif event.key == K_9:
                    DT -= 0.1
                    DT = max(0.1, DT)
                elif event.key == K_0:
                    DT += 0.1
                    DT = min(3.0, DT)

        keys = pygame.key.get_pressed()

        if keys[K_LEFT]:
            rocket.angle += 0.02
        if keys[K_RIGHT]:
            rocket.angle -= 0.02

        if keys[K_q]:
            rocket.throttle = min(2.0, rocket.throttle + 0.01)
        if keys[K_e]:
            rocket.throttle = max(0.0, rocket.throttle - 0.01)

        steps = 1
        for _ in range(steps):
            iss.step(DT)
            thrust_vec = None
            if autopilot_enabled:
                thrust_vec = autopilot.compute_thrust(rocket, iss)
                rocket.angle = math.atan2(thrust_vec[1], thrust_vec[0])
            else:
                if rocket.thrust_on:
                    thrust = MAX_THRUST * rocket.throttle
                    thrust_vec = thrust * np.array([math.cos(rocket.angle), math.sin(rocket.angle)])
            rocket.step(DT, thrust_vec)

        screen.fill(BG_COLOR)

        # Атмосфера
        for i in range(5):
            alt = i * 20000
            r_atm = R_EARTH + alt
            if r_atm * SCALE > 0:
                atm_radius = max(2, int(r_atm * SCALE))
                alpha = max(10, int(255 * math.exp(-alt / (SCALE_HEIGHT * 2))))
                s = pygame.Surface((atm_radius * 2, atm_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (30, 100, 200, alpha), (atm_radius, atm_radius), atm_radius)
                screen.blit(s, (CENTER[0] - atm_radius, CENTER[1] - atm_radius))

        # Земля
        earth_screen_r = max(8, int(R_EARTH * SCALE))
        pygame.draw.circle(screen, EARTH_COLOR, CENTER, earth_screen_r)

        # ISS orbit
        pygame.draw.circle(screen, ORBIT_COLOR, CENTER, max(2, int(R_ISS * SCALE)), 1)

        # траектории
        sim_pos = rocket.pos.copy()
        sim_vel = rocket.vel.copy()

        future_traj = []

        dt_sim = 10
        eps = 1.0
        max_steps = 1000

        for _ in range(max_steps):
            r = np.linalg.norm(sim_pos)
            if r < R_EARTH:
                break

            a_grav = -MU * sim_pos / r ** 3
            sim_vel += a_grav * dt_sim
            sim_pos_new = sim_pos + sim_vel * dt_sim

            if np.linalg.norm(sim_pos_new - sim_pos) < eps:
                break

            sim_pos = sim_pos_new
            future_traj.append(sim_pos.copy())

        pts = [world_to_screen(p, SCALE) for p in future_traj]
        if len(pts) >= 2:
            pygame.draw.lines(screen, (100, 255, 100), False, pts, 1)

        if len(rocket.traj) > 2:
            pts = [world_to_screen(p, SCALE) for p in rocket.traj]
            if len(pts) >= 2:
                pygame.draw.lines(screen, TRAJ_COLOR, False, pts, 1)

        # ISS
        iss_s = world_to_screen(iss.pos, SCALE)
        pygame.draw.circle(screen, ISS_COLOR, iss_s, 5)
        draw_text(screen, 'МКС', iss_s[0] + 8, iss_s[1] - 8, size=14, color=ISS_COLOR)

        # Ракета
        ch_s = world_to_screen(rocket.pos, SCALE)
        pygame.draw.polygon(screen, ROCKET_COLOR,
                            [
                                (ch_s[0] + 10 * math.cos(rocket.angle), ch_s[1] - 10 * math.sin(rocket.angle)),
                                (
                                    ch_s[0] - 6 * math.cos(rocket.angle + 0.6),
                                    ch_s[1] + 6 * math.sin(rocket.angle + 0.6)),
                                (
                                    ch_s[0] - 6 * math.cos(rocket.angle - 0.6),
                                    ch_s[1] + 6 * math.sin(rocket.angle - 0.6)),
                            ])

        if rocket.thrust_on or autopilot_enabled:
            if autopilot_enabled:
                thrust_vec = autopilot.compute_thrust(rocket, iss)
            else:
                thrust_vec = MAX_THRUST * rocket.throttle * np.array([math.cos(rocket.angle), math.sin(rocket.angle)])
            t_end_world = rocket.pos + thrust_vec * 0.00005
            t_end = world_to_screen(t_end_world, SCALE)
            pygame.draw.line(screen, (255, 100, 0), ch_s, t_end, 3)
            draw_text(screen, 'T', t_end[0] + 4, t_end[1] - 8)

        # Полётные данные
        draw_text(screen, f'Autopilot: {"ON" if autopilot_enabled else "OFF"}', 10, 10)
        draw_text(screen, f'Thrust: {"ON" if rocket.thrust_on else "OFF"}  Throttle: {rocket.throttle:.2f}', 10, 30)
        draw_text(screen, f'Scale: {SCALE:.2e} px/m  DT: {DT:.1f}s', 10, 50)
        draw_text(screen, f'Fuel: {rocket.fuel_mass:.0f} kg', 10, 130)
        r = np.linalg.norm(rocket.pos)
        alt = r - R_EARTH
        rel_pos = iss.pos - rocket.pos
        rel_dist = np.linalg.norm(rel_pos)
        draw_text(screen, f'Rel. dist to ISS: {rel_dist / 1000:.1f} km', 10, 90)
        rel_vel = np.linalg.norm(iss.vel - rocket.vel)
        draw_text(screen, f'Rel. vel to ISS: {rel_vel:.2f} m/s', 10, 110)

        # Данные об атмосфере
        density = atmospheric_density(alt)
        drag_f = np.linalg.norm(drag_force(rocket.vel, alt)) if alt > 0 else 0
        draw_text(screen, f'Atm. density: {density:.4f} kg/m³', 10, 150)
        draw_text(screen, f'Drag force: {drag_f / 1000:.1f} kN', 10, 170)
        draw_text(screen, f'Altitude: {alt / 1000:.1f} km', 10, 70)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
