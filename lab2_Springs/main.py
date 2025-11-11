import numpy as np
import pygame
from graphics import GraphicsRecorder
import colorsys

class Scene2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
        self.springs = []
        self.blocks = []
        self.gravity = np.array([0.0, 0.0])
        
        self._positions_cache = None
        self._velocities_cache = None
        self._forces_cache = None
        self._masses_cache = None
        self._fixed_mask = None
        self._needs_rebuild = True

    def add_wall(self, wall):
        self.walls.append(wall)

    def add_spring(self, spring):
        self.springs.append(spring)
        self._needs_rebuild = True

    def add_block(self, block):
        self.blocks.append(block)
        self._needs_rebuild = True

    def get_last_block(self):
        return self.blocks[-1]

    def get_all_objects(self):
        return self.walls + self.springs + self.blocks
    
    def rebuild_caches(self):
        n = len(self.blocks)
        self._positions_cache = np.array([b.position for b in self.blocks])
        self._velocities_cache = np.array([b.velocity for b in self.blocks])
        self._forces_cache = np.zeros((n, 2))
        self._masses_cache = np.array([b.mass for b in self.blocks])
        self._fixed_mask = np.array([b.fixed for b in self.blocks])
        
        if self.springs:
            self._spring_indices = []
            self._spring_stiffness = []
            self._spring_damping = []
            self._spring_initial_lengths = []
            
            block_to_idx = {id(b): i for i, b in enumerate(self.blocks)}
            
            for spring in self.springs:
                if isinstance(spring.obj1, Block) and isinstance(spring.obj2, Block):
                    idx1 = block_to_idx.get(id(spring.obj1))
                    idx2 = block_to_idx.get(id(spring.obj2))
                    if idx1 is not None and idx2 is not None:
                        self._spring_indices.append([idx1, idx2])
                        self._spring_stiffness.append(spring.stiffness)
                        self._spring_damping.append(spring.damping)
                        self._spring_initial_lengths.append(spring.initial_length)
            
            if self._spring_indices:
                self._spring_indices = np.array(self._spring_indices)
                self._spring_stiffness = np.array(self._spring_stiffness)
                self._spring_damping = np.array(self._spring_damping)
                self._spring_initial_lengths = np.array(self._spring_initial_lengths)
            else:
                self._spring_indices = np.empty((0, 2), dtype=int)
        
        self._needs_rebuild = False


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.position = np.array([x1, y1, x2, y2], dtype=float)

    def get_position(self):
        return self.position.copy()


def _get_object_position(obj):
    if isinstance(obj, Block):
        return obj.position
    elif isinstance(obj, Wall):
        x1, y1, x2, y2 = obj.get_position()
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    else:
        raise ValueError("Неизвестный тип объекта")


class Spring:
    def __init__(self, obj1, obj2, stiffness=10, damping=0.1, max_force=1000.0, initial_length=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.stiffness = stiffness
        self.damping = damping
        self.max_force = max_force

        if initial_length is None:
            pos1 = _get_object_position(obj1)
            pos2 = _get_object_position(obj2)
            self.initial_length = np.linalg.norm(pos2 - pos1)
        else:
            self.initial_length = initial_length

        self.current_length = self.initial_length
        self.previous_length = self.initial_length

    def get_current_length(self):
        pos1 = _get_object_position(self.obj1)
        pos2 = _get_object_position(self.obj2)
        self.current_length = np.linalg.norm(pos2 - pos1)
        return self.current_length


class Block:
    def __init__(self, x, y, size=0.5, mass=1.0, fixed=False):
        self.position = np.array([x, y], dtype=float)
        self.initial_position = self.position.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.force = np.zeros(2, dtype=float)
        self.size = size
        self.mass = mass
        self.fixed = fixed
        
    def get_block_color(self):
        max_disp = 0.5
        displacement = np.linalg.norm(self.position - self.initial_position)
        t = np.clip(displacement / max_disp, 0, 1)
        hue = t * 0.0 + (1 - t) * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)


def get_all_block_colors_vectorized(blocks):
    n = len(blocks)
    colors = np.zeros((n, 3), dtype=int)
    
    positions = np.array([b.position for b in blocks])
    initial_positions = np.array([b.initial_position for b in blocks])
    
    displacements = np.linalg.norm(positions - initial_positions, axis=1)
    max_disp = 0.5
    t = np.clip(displacements / max_disp, 0, 1)
    hues = t * 0.0 + (1 - t) * 0.75
    
    for i, hue in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return colors


def create_grid_system():
    rows, cols = 120, 120
    scene = Scene2D(100, 100)

    top_wall = Wall(10, 10, 90, 10)
    bottom_wall = Wall(10, 90, 90, 90)
    left_wall = Wall(10, 10, 10, 90)
    right_wall = Wall(90, 10, 90, 90)

    scene.add_wall(top_wall)
    scene.add_wall(bottom_wall)
    scene.add_wall(left_wall)
    scene.add_wall(right_wall)

    blocks = []
    spacing_x = 80 / (cols - 1)
    spacing_y = 80 / (rows - 1)

    for i in range(rows):
        row_blocks = []
        for j in range(cols):
            x = 10 + j * spacing_x
            y = 10 + i * spacing_y
            fixed = (i == 0 and j == 0) or (i == 0 and j == cols - 1) or \
                    (i == rows - 1 and j == 0) or (i == rows - 1 and j == cols - 1)
            block = Block(x, y, size=2, mass=1.0, fixed=fixed)
            scene.add_block(block)
            row_blocks.append(block)
        blocks.append(row_blocks)

    for i in range(rows):
        for j in range(cols):
            current_block = blocks[i][j]

            if j < cols - 1:
                spring = Spring(current_block, blocks[i][j + 1], stiffness=300)
                scene.add_spring(spring)

            if i < rows - 1:
                spring = Spring(current_block, blocks[i + 1][j], stiffness=300)
                scene.add_spring(spring)

            if i < rows - 1 and j < cols - 1:
                spring = Spring(current_block, blocks[i + 1][j + 1], stiffness=200)
                scene.add_spring(spring)

            if i < rows - 1 and j > 0:
                spring = Spring(current_block, blocks[i + 1][j - 1], stiffness=200)
                scene.add_spring(spring)

    return scene, scene.blocks, scene.springs


def draw_scene(screen, scene, width_px=800, height_px=600, draw_springs=True):
    screen.fill((255, 255, 255))
    scale_x = width_px / scene.width
    scale_y = height_px / scene.height

    for wall in scene.walls:
        x1, y1, x2, y2 = wall.get_position()
        pygame.draw.line(
            screen, (0, 0, 0),
            (x1 * scale_x, y1 * scale_y),
            (x2 * scale_x, y2 * scale_y), 3
        )

    if draw_springs and len(scene.springs) < 5000:
        for spring in scene.springs:
            p1 = _get_object_position(spring.obj1)
            p2 = _get_object_position(spring.obj2)

            pygame.draw.line(
                screen, (220, 220, 220),
                (p1[0] * scale_x, p1[1] * scale_y),
                (p2[0] * scale_x, p2[1] * scale_y), 1
            )

    colors = get_all_block_colors_vectorized(scene.blocks)
    
    for i, block in enumerate(scene.blocks):
        x, y = block.position
        size = block.size * scale_x

        if block.fixed:
            color = (100, 100, 100)
        else:
            color = tuple(colors[i])

        pygame.draw.circle(
            screen, color,
            (int(x * scale_x), int(y * scale_y)),
            int(size)
        )


def apply_physics_vectorized(scene, dt, damping=0.1):
    if scene._needs_rebuild:
        scene.rebuild_caches()
    for i, block in enumerate(scene.blocks):
        scene._positions_cache[i] = block.position
        scene._velocities_cache[i] = block.velocity
    
    scene._forces_cache[:] = 0.0
    movable_mask = ~scene._fixed_mask
    scene._forces_cache[movable_mask] += scene.gravity * scene._masses_cache[movable_mask, np.newaxis]
    
    if len(scene._spring_indices) > 0:
        idx1 = scene._spring_indices[:, 0]
        idx2 = scene._spring_indices[:, 1]
        
        pos1 = scene._positions_cache[idx1]
        pos2 = scene._positions_cache[idx2]
        
        deltas = pos2 - pos1
        distances = np.linalg.norm(deltas, axis=1)
    
        valid_mask = distances > 1e-10
        
        directions = np.zeros_like(deltas)
        directions[valid_mask] = deltas[valid_mask] / distances[valid_mask, np.newaxis]
        
        force_magnitudes = scene._spring_stiffness * (distances - scene._spring_initial_lengths)
        
        vel1 = scene._velocities_cache[idx1]
        vel2 = scene._velocities_cache[idx2]
        relative_velocities = vel2 - vel1
        damping_forces = scene._spring_damping * np.sum(relative_velocities * directions, axis=1)
        force_magnitudes += damping_forces
        
        forces = directions * force_magnitudes[:, np.newaxis]
        
        np.add.at(scene._forces_cache, idx1, forces)
        np.add.at(scene._forces_cache, idx2, -forces)
        
    scene._forces_cache[scene._fixed_mask] = 0.0
    
    scene._forces_cache[movable_mask] -= damping * scene._velocities_cache[movable_mask]
    
    accelerations = scene._forces_cache / scene._masses_cache[:, np.newaxis]
    scene._velocities_cache += accelerations * dt
    scene._positions_cache += scene._velocities_cache * dt
    
    x_min, x_max = 8, 92
    y_min, y_max = 8, 92
    
    sizes = np.array([b.size for b in scene.blocks])

    left_collision = scene._positions_cache[:, 0] - sizes < x_min
    scene._positions_cache[left_collision, 0] = x_min + sizes[left_collision]
    scene._velocities_cache[left_collision, 0] *= -1
    
    right_collision = scene._positions_cache[:, 0] + sizes > x_max
    scene._positions_cache[right_collision, 0] = x_max - sizes[right_collision]
    scene._velocities_cache[right_collision, 0] *= -1
    
    top_collision = scene._positions_cache[:, 1] - sizes < y_min
    scene._positions_cache[top_collision, 1] = y_min + sizes[top_collision]
    scene._velocities_cache[top_collision, 1] *= -1
    
    bottom_collision = scene._positions_cache[:, 1] + sizes > y_max
    scene._positions_cache[bottom_collision, 1] = y_max - sizes[bottom_collision]
    scene._velocities_cache[bottom_collision, 1] *= -1

    scene._velocities_cache[scene._fixed_mask] = 0.0
    
    for i, block in enumerate(scene.blocks):
        if not block.fixed:
            block.position[:] = scene._positions_cache[i]
            block.velocity[:] = scene._velocities_cache[i]

def main():
    pygame.init()
    width_px, height_px = 1200, 800
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("2D Physics Simulation - Grid System (Optimized)")

    scene, blocks, springs = create_grid_system()
    scene.rebuild_caches()
    recorder = GraphicsRecorder(blocks)
    clock = pygame.time.Clock()
    running = True

    if hasattr(scene, 'gravity'):
        scene.gravity = np.array([0.0, 0.0])

    time_sim = 0.0
    dt = 0.03  # ~60 FPS
    
    draw_springs_flag = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                scale_x = width_px / scene.width
                scale_y = height_px / scene.height
                mouse_pos = np.array([mouse_x / scale_x, mouse_y / scale_y])
                positions = np.array([b.position for b in scene.blocks])
                distances = np.linalg.norm(positions - mouse_pos, axis=1)
                close_blocks = distances < 5
                
                for i, is_close in enumerate(close_blocks):
                    if is_close and not scene.blocks[i].fixed:
                        block = scene.blocks[i]
                        direction = block.position - mouse_pos
                        dir_norm = np.linalg.norm(direction)
                        if dir_norm > 0:
                            direction = direction / dir_norm
                            block.velocity += direction * 10

        apply_physics_vectorized(scene, dt)
        draw_scene(screen, scene, width_px, height_px, draw_springs_flag)

        recorder.record_step(time_sim)
        
        fps = clock.get_fps()
        pygame.display.set_caption(f"2D Physics Simulation (Optimized) - FPS: {fps:.1f}")
        
        pygame.display.flip()
        clock.tick(60)

        time_sim += dt

    pygame.quit()
    recorder.plot_analysis()


if __name__ == "__main__":
    main()
