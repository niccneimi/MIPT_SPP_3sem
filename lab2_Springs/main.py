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
    def __init__(self, obj1, obj2, stiffness=10, max_force=1000.0, initial_length=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.stiffness = stiffness
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
    def __init__(self, x, y, size=0.5, mass=1.0):
        self.position = np.array([x, y], dtype=float)
        self.initial_position = self.position.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.force = np.zeros(2, dtype=float)
        self.size = size
        self.mass = mass


    def get_block_color(self):
        max_disp = 0.5
        displacement = np.linalg.norm(self.position - self.initial_position)
        t = np.clip(displacement / max_disp, 0, 1)
        hue = (1 - t) * 0.0 + t * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return (int(r*255), int(g*255), int(b*255))


def create_system():
    block_count = 40
    scene = Scene2D(100, 16)

    left_wall = Wall(1, 2, 1, 4)
    right_wall = Wall((block_count + 1)*2, 2, (block_count + 1)*2, 4)
    scene.add_wall(left_wall)
    scene.add_wall(right_wall)

    first_block = Block(3, 3)
    scene.add_block(first_block)
    
    for i in range(block_count - 1):
        last_block = scene.get_last_block()
        new_block = Block(last_block.position[0] + 2, last_block.position[1])
        scene.add_block(new_block)
        
    first_spring = Spring(left_wall, first_block)
    scene.add_spring(first_spring)
    
    for i in range(1, len(scene.blocks)):
        new_spring = Spring(scene.blocks[i - 1], scene.blocks[i])
        scene.add_spring(new_spring)
    
    last_spring = Spring(scene.get_last_block(), right_wall)
    scene.add_spring(last_spring)

    return scene, scene.blocks, scene.springs


def draw_scene(screen, scene, width_px=800, height_px=400):
    screen.fill((255, 255, 255))
    scale_x = width_px / scene.width
    scale_y = height_px / scene.height

    for wall in scene.walls:
        x1, y1, x2, y2 = wall.get_position()
        pygame.draw.line(
            screen, (0, 0, 0),
            (x1 * scale_x, y1 * scale_y),
            (x2 * scale_x, y2 * scale_y), 5
        )

    for spring in scene.springs:
        p1 = _get_object_position(spring.obj1)
        p2 = _get_object_position(spring.obj2)
        pygame.draw.line(
            screen, (0, 128, 255),
            (p1[0] * scale_x, p1[1] * scale_y),
            (p2[0] * scale_x, p2[1] * scale_y), 2
        )

    for block in scene.blocks:
        x, y = block.position
        size = block.size * scale_x
        color = block.get_block_color()
        pygame.draw.rect(
            screen, color,
            pygame.Rect(int(x * scale_x), int(y * scale_y - int(size // 2)), int(size), int(size))
        )

def main():
    pygame.init()
    width_px, height_px = 1600, 400
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("2D Signal Propagation")
    scene, blocks, springs = create_system()

    recorder = GraphicsRecorder(blocks)
    clock = pygame.time.Clock()
    running = True

    #SIGNAL
    blocks[0].position += np.array([-1, 0])

    time_sim = 0.0
    dt = 0.02
    damping = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        for block in blocks:
            block.force[:] = 0.0
            
        for spring in springs:
            p1 = _get_object_position(spring.obj1)
            p2 = _get_object_position(spring.obj2)
            delta = p2 - p1
            dist = np.linalg.norm(delta)
            if dist == 0:
                continue
            direction = delta / dist
            force = spring.stiffness * (dist - spring.initial_length)
            force = np.clip(force, -spring.max_force, spring.max_force)
            if isinstance(spring.obj1, Block):
                spring.obj1.force += direction * force
            if isinstance(spring.obj2, Block):
                spring.obj2.force -= direction * force

        for block in blocks:
            block.force -= damping * block.velocity
            acceleration = block.force / block.mass
            block.velocity += acceleration * dt
            block.position += block.velocity * dt

        draw_scene(screen, scene, width_px, height_px)
        pygame.display.flip()
        clock.tick(60)

        recorder.record_step(time_sim)
        time_sim += dt

    pygame.quit()
    recorder.plot_positions()

if __name__ == "__main__":
    main()
