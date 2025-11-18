import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_all_metrics(metrics, velocities, mass=1.0, dt=0.004):
    print("Построение графиков...")
    
    # Улучшенная фигура с лучшим расположением
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Кинетическая энергия
    plt.subplot(2, 3, 1)
    ke = metrics.history['kinetic_energy']
    time = metrics.history['time']
    
    plt.plot(time, ke, linewidth=1.5, color='#2E86AB')
    plt.title('Кинетическая энергия', fontsize=12, fontweight='bold')
    plt.xlabel('Время')
    plt.ylabel('Энергия')
    plt.grid(True, alpha=0.3)
    
    # Форматируем оси чтобы убрать длинные числа
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 2. Давление
    plt.subplot(2, 3, 2)
    pressure = metrics.history['pressure']
    
    plt.plot(time, pressure, linewidth=1.5, color='#A23B72')
    plt.title('Давление', fontsize=12, fontweight='bold')
    plt.xlabel('Время')
    plt.ylabel('Давление')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 3. Средняя скорость
    plt.subplot(2, 3, 3)
    mean_speed = metrics.history['mean_speed']
    
    plt.plot(time, mean_speed, linewidth=1.5, color='#F18F01')
    plt.title('Средняя скорость', fontsize=12, fontweight='bold')
    plt.xlabel('Время')
    plt.ylabel('Скорость')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 4. Распределение скоростей
    plt.subplot(2, 3, 4)
    speeds = np.linalg.norm(velocities, axis=1)
    plt.hist(speeds, bins=50, alpha=0.7, density=True, label='Симуляция', 
             color='#6699CC', edgecolor='black', linewidth=0.5)
    
    mean_ke = np.mean(0.5 * mass * speeds**2)
    kT = 2 * mean_ke
    
    hist, bin_edges = np.histogram(speeds, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    theoretical = (mass / kT) * bin_centers * np.exp(-mass * bin_centers**2 / (2 * kT))
    
    plt.plot(bin_centers, theoretical, 'r-', linewidth=2, label='Теоретическое')
    plt.title('Распределение скоростей', fontsize=12, fontweight='bold')
    plt.xlabel('Скорость')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Компоненты скорости
    plt.subplot(2, 3, 5)
    plt.hist(velocities[:, 0], bins=50, alpha=0.6, density=True, label='v_x',
             color='#FF6B6B', edgecolor='black', linewidth=0.5)
    plt.hist(velocities[:, 1], bins=50, alpha=0.6, density=True, label='v_y',
             color='#4ECDC4', edgecolor='black', linewidth=0.5)
    
    x = np.linspace(velocities[:, 0].min(), velocities[:, 0].max(), 100)
    sigma = np.std(velocities[:, 0])
    plt.plot(x, norm.pdf(x, 0, sigma), 'k--', linewidth=2, label='Гаусс')
    
    plt.title('Компоненты скорости', fontsize=12, fontweight='bold')
    plt.xlabel('Компоненты скорости')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. MSD
    plt.subplot(2, 3, 6)
    times, msd = metrics.msd_calc.calculate_msd()
    
    if len(msd) > 1:
        plt.loglog(times * dt, msd, 'o-', markersize=4, color='#6A4C93', 
                  linewidth=1.5, markeredgecolor='black', markeredgewidth=0.5)
        plt.title('Среднеквадратичное смещение', fontsize=12, fontweight='bold')
        plt.xlabel('Время')
        plt.ylabel('MSD')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        if len(msd) > 10:
            log_t = np.log(times[1:10] * dt)
            log_msd = np.log(msd[1:10])
            coeffs = np.polyfit(log_t, log_msd, 1)
            exponent = coeffs[0]
            plt.text(0.05, 0.95, f'Наклон: {exponent:.3f}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Второй набор графиков
    fig2 = plt.figure(figsize=(12, 5))
    
    # 7. Фазовый портрет
    plt.subplot(1, 2, 1)
    ke = metrics.history['kinetic_energy']
    pressure = metrics.history['pressure']
    
    plt.scatter(ke, pressure, alpha=0.7, s=20, color='#2A9D8F', edgecolors='black', linewidth=0.3)
    plt.title('Фазовый портрет: энергия vs давление', fontsize=12, fontweight='bold')
    plt.xlabel('Кинетическая энергия')
    plt.ylabel('Давление')
    plt.grid(True, alpha=0.3)
    
    # 8. Статистика
    plt.subplot(1, 2, 2)
    summary = metrics.get_summary()
    
    mean_values = {k.replace('_mean', ''): v for k, v in summary.items() if 'mean' in k}
    
    names = ['Энергия', 'Давление', 'Скорость']
    values = [mean_values.get('kinetic_energy', 0), 
              mean_values.get('pressure', 0), 
              mean_values.get('mean_speed', 0)]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = plt.bar(names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    plt.title('Средние значения', fontsize=12, fontweight='bold')
    plt.ylabel('Значение')
    plt.grid(True, axis='y', alpha=0.3)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print("\n" + "="*50)
    print("СТАТИСТИКА СИМУЛЯЦИИ")
    print("="*50)
    for key, value in summary.items():
        print(f"{key:25}: {value:.6f}")