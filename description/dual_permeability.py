import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_histograms_from_csv_folder(data_dir, params=['C_D', 'S', 'k'], bins=30, figsize=(15, 10), data_type='train', n_cols=3):
    """
    Построение гистограмм для указанных параметров из всех CSV файлов в папке
    
    Args:
        data_dir: путь к папке с CSV файлами
        params: список параметров для визуализации
        bins: количество бинов в гистограмме
        figsize: размер фигуры (ширина, высота)
        data_type: тип данных ('train' или 'test')
        n_cols: количество графиков в ряду
    """
    
    # Словарь для хранения значений параметров
    param_values = {param: [] for param in params}
    
    # Проходим по всем CSV файлам
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print(f"Найдено файлов: {len(csv_files)}")
    
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        
        try:
            # Читаем CSV файл
            df = pd.read_csv(file_path)

            df = df[df['data_type'] == data_type]

            # Извлекаем значения параметров
            if len(df) > 0: 
                for param in params:
                    if param in df.columns:
                        # Берем значение только если строка существует
                        raw_val = df[param].values[0] 
                        value = pd.to_numeric(raw_val, errors='coerce')
                        if not pd.isna(value):
                            param_values[param].append(value)
            else:
                # Если файла нет в списке 'test', мы просто идем к следующему файлу
                continue
                    
        except Exception as e:
            print(f"Ошибка при чтении файла {file}: {e}")
    
    # Выводим статистику
    print("\n" + "="*60)
    print("СТАТИСТИКА ПАРАМЕТРОВ")
    print("="*60)
    
    for param in params:
        values = param_values[param]
        if values:
            print(f"\n{param}:")
            print(f"  Количество значений: {len(values)}")
            print(f"  Среднее: {np.mean(values):.6e}")
            print(f"  Медиана: {np.median(values):.6e}")
            print(f"  Стандартное отклонение: {np.std(values):.6e}")
            print(f"  Мин: {np.min(values):.6e}")
            print(f"  Макс: {np.max(values):.6e}")
            print(f"  Диапазон: {np.max(values) - np.min(values):.6e}")
        else:
            print(f"\n{param}: Нет данных")
    
    # Рассчитываем количество строк и столбцов
    n_params = len(params)
    n_rows = (n_params + n_cols - 1) // n_cols  # округление вверх
    
    # Создаем фигуру с сеткой подграфиков
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Если есть только одна строка, axes может быть одномерным
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    # Если есть только один столбец в строке
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Строим гистограммы
    for idx, param in enumerate(params):
        row = idx // n_cols
        col = idx % n_cols
        
        # Получаем текущую ось
        ax = axes[row, col]
        
        values = param_values[param]
        
        if values:
            # Для параметров с большим разбросом используем логарифмическую шкалу
            if param in ['k_f']:  # проницаемость часто имеет широкий диапазон
                # Логарифмическая гистограмма
                log_values = np.array(values) / 1e-12
                ax.hist(log_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(f'{param}, 1e-12 м²', fontsize=10)
            elif param in ['C_D']:
                ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(f'{param}', fontsize=10)
            elif param in ['lambda']:
                log_values = np.array(values)
                ax.hist(log_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(f'{param}', fontsize=10)
            else:
                # Обычная гистограмма
                ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(param, fontsize=10)

            # Добавляем статистику в угол
            stats_text = f"n={len(values)}\nμ={np.mean(values):.2e}\nσ={np.std(values):.2e}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
            
            ax.set_ylabel('Частота', fontsize=10)
            ax.set_title(f'Распределение {param}', fontsize=12)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Нет данных для {param}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(param)
    
    # Скрываем неиспользуемые подграфики
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.show()
    
    return param_values

if __name__ == "__main__":
    data_dir = "/home/ilnaz/PycharmProjects/datasets/new/dual_permeability_fin/params"
    
    param_values = plot_histograms_from_csv_folder(
        data_dir=data_dir,
        params=['C_D', 'S1', 'k1', 'omega', 'lambda', 'kappa', 'R_eD'],
        bins=500,
        figsize=(15, 8),
        data_type='train',
        n_cols=3  # 3 графика в ряду
    )
    