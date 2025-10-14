import matplotlib.pyplot as plt
import numpy as np

class GraphicsRecorder:
    def __init__(self, blocks):
        self.blocks = blocks
        self.positions_over_time = []
        self.times = []

    def record_step(self, current_time):
        """Вызывается каждый шаг симуляции"""
        self.times.append(current_time)
        self.positions_over_time.append([block.position[0] for block in self.blocks])

    def plot_positions(self):
        """Построить графики позиций блоков по времени"""
        positions_array = np.array(self.positions_over_time)
        plt.figure(figsize=(12, 6))
        for i in range(len(self.blocks)):
            plt.plot(self.times, positions_array[:, i], label=f'Block {i}')
        plt.xlabel('Time [s]')
        plt.ylabel('X Position')
        plt.title('Block positions over time')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    from time import time
    class DummyBlock:
        def __init__(self, x):
            self.position = np.array([x, 0], dtype=float)

    blocks = [DummyBlock(i) for i in range(5)]
    recorder = GraphicsRecorder(blocks)
    start = time()

    for step in range(100):
        for b in blocks:
            b.position[0] += np.random.rand() * 0.1
        recorder.record_step(time() - start)

    recorder.plot_positions()
