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

    def add_wall(self, wall):
        self.walls.append(wall)

    def add_spring(self, spring):
        self.springs.append(spring)

    def add_block(self, block):
        self.blocks.append(block)

    def get_last_block(self):
        return self.blocks[-1]

    def get_all_objects(self):
        return self.walls + self.springs + self.blocks


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


def create_grid_system():
    rows, cols = 10, 10
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
                    (i == rows - 1 and j == 0) or (i == rows - 1 and j == cols - 1) # or \
                    # (abs(i - rows // 2) < 5 and abs(j - cols // 2) < 5)
            block = Block(x, y, size=2, mass=1.0, fixed=fixed)
            scene.add_block(block)
            row_blocks.append(block)
        blocks.append(row_blocks)

    for i in range(rows):
        for j in range(cols):
            current_block = blocks[i][j]

            if j < cols - 1:
                spring = Spring(current_block, blocks[i][j + 1], stiffness=15)
                scene.add_spring(spring)

            if i < rows - 1:
                spring = Spring(current_block, blocks[i + 1][j], stiffness=15)
                scene.add_spring(spring)

            if i < rows - 1 and j < cols - 1:
                spring = Spring(current_block, blocks[i + 1][j + 1], stiffness=10)
                scene.add_spring(spring)

            if i < rows - 1 and j > 0:
                spring = Spring(current_block, blocks[i + 1][j - 1], stiffness=10)
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

    for block in scene.blocks:
        x, y = block.position
        size = block.size * scale_x

        if block.fixed:
            color = (100, 100, 100)
        else:
            color = block.get_block_color()

        pygame.draw.circle(
            screen, color,
            (int(x * scale_x), int(y * scale_y)),
            int(size)
        )


def apply_physics_optimized(scene, dt, damping=0.1):
    movable_blocks = [b for b in scene.blocks if not b.fixed]

    # Сбрасываем силы и добавляем гравитацию
    for block in movable_blocks:
        block.force[:] = 0.0
        block.force += scene.gravity * block.mass

    # Позиции блоков для пружин
    springs_to_remove = []
    for spring in scene.springs:
        pos1 = _get_object_position(spring.obj1)
        pos2 = _get_object_position(spring.obj2)
        delta = pos2 - pos1
        dist = np.linalg.norm(delta)
        if dist < 1e-10:
            continue

        # Разрыв пружины по длине
        max_length = spring.initial_length * 1.6  # естественный порог
        if dist > max_length:
            springs_to_remove.append(spring)
            continue

        direction = delta / dist
        force_magnitude = spring.stiffness * (dist - spring.initial_length)

        # Демпфирование
        if isinstance(spring.obj1, Block) and isinstance(spring.obj2, Block):
            relative_velocity = spring.obj2.velocity - spring.obj1.velocity
            force_magnitude += spring.damping * np.dot(relative_velocity, direction)

        force = direction * force_magnitude
        if isinstance(spring.obj1, Block) and not spring.obj1.fixed:
            spring.obj1.force += force
        if isinstance(spring.obj2, Block) and not spring.obj2.fixed:
            spring.obj2.force -= force

    # Удаляем разорванные пружины
    for spring in springs_to_remove:
        scene.springs.remove(spring)

    # Обновляем блоки
    for block in movable_blocks:
        block.force -= damping * block.velocity
        acceleration = block.force / block.mass
        block.velocity += acceleration * dt
        block.position += block.velocity * dt

        # Упругое отражение от стенок конструкции (границы)
        x_min, x_max = 8, 92
        y_min, y_max = 8, 92

        if block.position[0] - block.size < x_min:
            block.position[0] = x_min + block.size
            block.velocity[0] *= -1
        elif block.position[0] + block.size > x_max:
            block.position[0] = x_max - block.size
            block.velocity[0] *= -1

        if block.position[1] - block.size < y_min:
            block.position[1] = y_min + block.size
            block.velocity[1] *= -1
        elif block.position[1] + block.size > y_max:
            block.position[1] = y_max - block.size
            block.velocity[1] *= -1

def main():
    pygame.init()
    width_px, height_px = 1200, 800
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("2D Physics Simulation - Grid System")

    scene, blocks, springs = create_grid_system()

    recorder = GraphicsRecorder(blocks)
    clock = pygame.time.Clock()
    running = True

    if hasattr(scene, 'gravity'):
        scene.gravity = np.array([0.0, 0.0])

    center_block = blocks[len(blocks) // 2] if hasattr(blocks, '__len__') else blocks[10]
    if hasattr(center_block, 'position'):
        center_block.position += np.array([0, 0])

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

                for block in scene.blocks:
                    if not block.fixed:
                        dist = np.linalg.norm(block.position - mouse_pos)
                        if dist < 5:
                            direction = block.position - mouse_pos
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                block.velocity += direction * 10

        apply_physics_optimized(scene, dt)
        draw_scene(screen, scene, width_px, height_px, draw_springs_flag)

        fps = clock.get_fps()
        pygame.display.set_caption(f"2D Physics Simulation - FPS: {fps:.1f}")
        
        pygame.display.flip()
        clock.tick(60)

        recorder.record_step(time_sim)
        time_sim += dt

    pygame.quit()
    recorder.plot_positions()


if __name__ == "__main__":
    main()
