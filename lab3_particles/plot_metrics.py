import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit


def plot_all_metrics(metrics, velocities, mass=1.0, dt=0.004):
    print("Построение графиков...")

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

    mean_ke = np.mean(0.5 * mass * speeds ** 2)
    kT = 2 * mean_ke

    hist, bin_edges = np.histogram(speeds, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    theoretical = (mass / kT) * bin_centers * np.exp(-mass * bin_centers ** 2 / (2 * kT))

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
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\nПостроение графиков анализа диффузии...")
    fig3 = plt.figure(figsize=(15, 10))

    # 9. Анализ диффузионного коэффициента
    plt.subplot(2, 3, 1)
    times, msd = metrics.msd_calc.calculate_msd()

    if len(msd) > 10:
        # Линейная подгонка для определения коэффициента диффузии
        linear_times = times[1:20] * dt
        linear_msd = msd[1:20]

        def linear_func(t, D):
            return 4 * D * t  # Для 2D: MSD = 4Dt

        try:
            popt, pcov = curve_fit(linear_func, linear_times, linear_msd)
            D_measured = popt[0]
            D_error = np.sqrt(pcov[0, 0])

            plt.plot(linear_times, linear_msd, 'bo', markersize=4, label='Данные MSD')
            fitted_msd = linear_func(linear_times, D_measured)
            plt.plot(linear_times, fitted_msd, 'r-', linewidth=2,
                     label=f'Линейная подгонка\nD = {D_measured:.6f} ± {D_error:.6f}')

            plt.xlabel('Время')
            plt.ylabel('MSD')
            plt.title('Определение коэффициента диффузии', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

        except Exception as e:
            plt.text(0.5, 0.5, 'Ошибка подгонки', transform=plt.gca().transAxes,
                     ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'Недостаточно данных', transform=plt.gca().transAxes,
                 ha='center', va='center')

    # 10. Распределение смещений
    plt.subplot(2, 3, 2)
    if hasattr(metrics.msd_calc, 'positions_history') and len(metrics.msd_calc.positions_history) > 10:
        # Берем последние позиции для анализа
        recent_positions = metrics.msd_calc.positions_history[-1]
        initial_positions = metrics.msd_calc.positions_history[0]

        displacements = recent_positions - initial_positions
        dx = displacements[:, 0]
        dy = displacements[:, 1]

        plt.hist(dx, bins=50, alpha=0.6, density=True, label='Δx',
                 color='#FF6B6B', edgecolor='black', linewidth=0.5)
        plt.hist(dy, bins=50, alpha=0.6, density=True, label='Δy',
                 color='#4ECDC4', edgecolor='black', linewidth=0.5)

        # Гауссово распределение для сравнения
        x_range = np.linspace(min(dx.min(), dy.min()), max(dx.max(), dy.max()), 100)
        sigma_x = np.std(dx)
        sigma_y = np.std(dy)

        plt.plot(x_range, norm.pdf(x_range, 0, sigma_x), 'k--', linewidth=1, label='Гаусс (x)')
        plt.plot(x_range, norm.pdf(x_range, 0, sigma_y), 'k:', linewidth=1, label='Гаусс (y)')

        plt.xlabel('Смещение')
        plt.ylabel('Плотность вероятности')
        plt.title('Распределение смещений', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 11. Автокорреляция скорости
    plt.subplot(2, 3, 3)
    if hasattr(metrics, 'velocity_autocorrelation'):
        vacf = metrics.velocity_autocorrelation
        if len(vacf) > 1:
            times_vacf = np.arange(len(vacf)) * dt
            plt.plot(times_vacf[:min(100, len(vacf))], vacf[:min(100, len(vacf))],
                     linewidth=2, color='#9B59B6')
            plt.xlabel('Время')
            plt.ylabel('VACF')
            plt.title('Автокорреляция скорости', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # Вычисление времени релаксации
            if len(vacf) > 5 and vacf[0] > 0:
                tau_index = np.where(vacf < vacf[0] * np.exp(-1))[0]
                if len(tau_index) > 0:
                    tau = tau_index[0] * dt
                    plt.axvline(x=tau, color='red', linestyle='--',
                                label=f'τ = {tau:.3f}')
                    plt.legend()

    # 12. Анализ не-гауссовости
    plt.subplot(2, 3, 4)
    if hasattr(metrics.msd_calc, 'positions_history') and len(metrics.msd_calc.positions_history) > 10:
        alpha2_values = []
        time_points = []

        for i in range(1, min(20, len(metrics.msd_calc.positions_history))):
            positions = metrics.msd_calc.positions_history[i]
            initial_positions = metrics.msd_calc.positions_history[0]
            displacements = positions - initial_positions
            dr4 = np.mean(np.sum(displacements ** 2, axis=1) ** 2)
            dr2 = np.mean(np.sum(displacements ** 2, axis=1))

            alpha2 = (3 * dr4) / (2 * dr2 ** 2) - 1  # Для 2D
            alpha2_values.append(alpha2)
            time_points.append(i * dt)

        plt.plot(time_points, alpha2_values, 'o-', color='#E74C3C', linewidth=2)
        plt.xlabel('Время')
        plt.ylabel('α₂(t)')
        plt.title('Не-гауссовость смещений', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 13. Сравнение теоретического и измеренного D
    plt.subplot(2, 3, 5)
    if 'D_measured' in locals() and 'mean_speed' in metrics.history:
        mean_v = np.mean(metrics.history['mean_speed'])
        mean_free_path = 1.0 / (np.sqrt(2) * len(velocities) * np.pi * (2 * 0.005) ** 2 / (10.0 * 6.0))
        collision_time = mean_free_path / mean_v

        # Теоретический коэффициент диффузии (упрощенная формула)
        D_theoretical = mean_v ** 2 * collision_time / 2

        labels = ['Измеренный', 'Теоретический']
        values = [D_measured, D_theoretical]
        colors = ['#3498DB', '#E67E22']

        bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
        plt.ylabel('Коэффициент диффузии')
        plt.title('Сравнение коэффициентов диффузии', fontsize=12, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                     f'{value:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 14. Траектории нескольких частиц
    plt.subplot(2, 3, 6)
    if hasattr(metrics.msd_calc, 'positions_history') and len(metrics.msd_calc.positions_history) > 10:
        # Выбираем несколько случайных частиц для отображения траекторий
        num_trajectories = 5
        particle_indices = np.random.choice(len(velocities), num_trajectories, replace=False)

        for i, idx in enumerate(particle_indices):
            traj_x = [pos[idx, 0] for pos in metrics.msd_calc.positions_history]
            traj_y = [pos[idx, 1] for pos in metrics.msd_calc.positions_history]

            plt.plot(traj_x, traj_y, linewidth=1, alpha=0.7,
                     label=f'Частица {i + 1}')
            plt.plot(traj_x[0], traj_y[0], 'o', markersize=4)
            plt.plot(traj_x[-1], traj_y[-1], 's', markersize=4)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Траектории частиц', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("АНАЛИЗ ДИФФУЗИИ")
    print("=" * 60)

    if 'D_measured' in locals():
        print(f"Коэффициент диффузии (измеренный): {D_measured:.8f}")
        print(f"Погрешность измерения D: {D_error:.8f}")

    if hasattr(metrics, 'velocity_autocorrelation') and len(metrics.velocity_autocorrelation) > 5:
        vacf = metrics.velocity_autocorrelation
        tau_index = np.where(vacf < vacf[0] * np.exp(-1))[0]
        if len(tau_index) > 0:
            tau = tau_index[0] * dt
            print(f"Время релаксации скорости: {tau:.6f}")

    if 'mean_speed' in metrics.history:
        mean_v = np.mean(metrics.history['mean_speed'])
        print(f"Средняя скорость: {mean_v:.6f}")

    print("\n" + "=" * 50)
    print("СТАТИСТИКА СИМУЛЯЦИИ")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key:25}: {value:.6f}")
