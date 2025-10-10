import numpy as np


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

    def __init__(self, obj1, obj2, stiffness=1.0, max_force=10.0, initial_length=None):
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

    def apply_initial_displacement(self, displacement):
        if isinstance(self.obj1, Block):
            self.obj1.position += displacement
        elif isinstance(self.obj2, Block):
            self.obj2.position += displacement

        self.initial_length = self.get_current_length()
        self.previous_length = self.initial_length
        self.current_length = self.initial_length


class Block:

    def __init__(self, x, y, size=0.2):
        self.position = np.array([x, y], dtype=float)
        self.size = size


def create_system():
    scene = Scene2D(10, 6)

    left_wall = Wall(0, 2, 0, 4)
    right_wall = Wall(8, 2, 8, 4)
    scene.add_wall(left_wall)
    scene.add_wall(right_wall)

    block1 = Block(2, 3)
    block2 = Block(4, 3)
    block3 = Block(6, 3)
    scene.add_block(block1)
    scene.add_block(block2)
    scene.add_block(block3)

    spring1 = Spring(left_wall, block1, stiffness=2.0)
    spring2 = Spring(block1, block2, stiffness=1.5)
    spring3 = Spring(block2, block3, stiffness=1.5)
    spring4 = Spring(block3, right_wall, stiffness=2.0)

    scene.add_spring(spring1)
    scene.add_spring(spring2)
    scene.add_spring(spring3)
    scene.add_spring(spring4)

    return scene, [block1, block2, block3], [spring1, spring2, spring3, spring4]


if __name__ == "__main__":
    print(create_system())
