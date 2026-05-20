import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Способ 1: Базовый - чтение всех файлов и построение гистограмм
def plot_histograms_from_csv_folder(data_dir, params=['C_D', 'S', 'k'], bins=30, figsize=(15, 5), data_type='train'):
    """
    Построение гистограмм для указанных параметров из всех CSV файлов в папке
    
    Args:
        data_dir: путь к папке с CSV файлами
        params: список параметров для визуализации
        bins: количество бинов в гистограмме
        figsize: размер фигуры
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
    
    # Создаем фигуру с тремя подграфиками
    fig, axes = plt.subplots(1, len(params), figsize=figsize)
    
    # Для случая одного параметра
    if len(params) == 1:
        axes = [axes]
    
    # Строим гистограммы
    for idx, param in enumerate(params):
        values = param_values[param]
        
        if values:
            # Для параметров с большим разбросом используем логарифмическую шкалу
            if param in ['k']:  # проницаемость часто имеет широкий диапазон
                # Логарифмическая гистограмма
                log_values = np.array(values) / 1e-12  # или / 10**(-12)
                axes[idx].hist(log_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                axes[idx].set_xlabel(f'{param}, 1e-12 м²', fontsize=12)  # понятная подпись
            elif param in ['C_D']:  # проницаемость часто имеет широкий диапазон
                # Логарифмическая гистограмма
                log_values = np.array(values)
                axes[idx].hist(log_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                axes[idx].set_xlabel(f'{param}', fontsize=12)  # понятная подпись
            else:
                # Обычная гистограмма
                axes[idx].hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                axes[idx].set_xlabel(param, fontsize=12)
            
            # Добавляем линии среднего и медианы
            mean_val = np.mean(values)
            median_val = np.median(values)
            
            # if param in ['k']:
            #     axes[idx].axvline(np.log10(mean_val), color='red', linestyle='--', 
            #                      linewidth=2, label=f'mean={mean_val:.2e}')
            #     axes[idx].axvline(np.log10(median_val), color='green', linestyle='--', 
            #                      linewidth=2, label=f'median={median_val:.2e}')
            # else:
            #     axes[idx].axvline(mean_val, color='red', linestyle='--', 
            #                      linewidth=2, label=f'mean={mean_val:.3e}')
            #     axes[idx].axvline(median_val, color='green', linestyle='--', 
            #                      linewidth=2, label=f'median={median_val:.3e}')
            
            # Добавляем статистику в угол
            stats_text = f"n={len(values)}\nμ={np.mean(values):.2e}\nσ={np.std(values):.2e}"
            axes[idx].text(0.95, 0.95, stats_text, transform=axes[idx].transAxes, 
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)
            
            axes[idx].set_ylabel('Частота', fontsize=12)
            axes[idx].set_title(f'Распределение {param}', fontsize=14)
            axes[idx].grid(True, alpha=0.3)
        else:
            axes[idx].text(0.5, 0.5, f'Нет данных для {param}', 
                          transform=axes[idx].transAxes, ha='center', va='center')
            axes[idx].set_title(param)
    
    plt.tight_layout()
    plt.show()
    
    return param_values

if __name__ == "__main__":
    # data_dir = "/home/ilnaz/PycharmProjects/datasets/new/homogeneous_fin/params"  # Замените на ваш путь
    
    # # Базовый вариант
    # param_values = plot_histograms_from_csv_folder(
    #     data_dir=data_dir,
    #     params=['C_D', 'S', 'k', 'r_D_e'],
    #     bins=500,
    #     figsize=(15, 5),
    #     data_type='val'
    # )

    data_dir = "/home/ilnaz/PycharmProjects/datasets/new/homogeneous_inf/params"  # Замените на ваш путь
    
    # Базовый вариант
    param_values = plot_histograms_from_csv_folder(
        data_dir=data_dir,
        params=['C_D', 'S', 'k'],
        bins=500,
        figsize=(15, 5),
        data_type='val'
    )
    