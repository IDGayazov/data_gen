from model.reservoir_model import ReservoirModel
import argparse
import matplotlib.pyplot as plt
import numpy as np


class CommonReservoirModel(ReservoirModel):
    """
    Общая для всех реализация модели пласта

    Можно загрузить датасет из файла
    """
    def F(self, s):
        print('Not supported method!')


def visualize_multiple_pressures(model_type, file_numbers, rows=2, cols=5, 
                                 horizontal_line=False, 
                                 save_path=None):
    """
    Визуализация нескольких кривых давления
    
    Parameters:
    - file_numbers: список номеров файлов
    - rows, cols: размеры сетки
    - horizontal_line: добавить горизонтальную линию y=0.5
    - save_path: путь для сохранения графика (опционально)
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.5))
    axes = axes.flatten()
    
    model = CommonReservoirModel()
    
    for idx, num in enumerate(file_numbers):
        if idx >= len(axes):
            print(f"Предупреждение: {len(file_numbers)} файлов, но только {len(axes)} subplot'ов")
            break
        
        file_name = get_file_template_for_model_type(model_type, num)
        
        try:
            df = model.load_model(file_name).get_pressure()
            ax = axes[idx]
            
            # Визуализация
            ax.loglog(df['t_D'], df['P_wD'], 'gx', label='Теоретическое давление',
                     markersize=3, linewidth=1)
            ax.loglog(df['t_D'], df['dP_wD'], 'b-', linewidth=2, label='dP/dln(t)', alpha=0.7)
            
            if 'P_wD_gauss' in df.columns:
                ax.loglog(df['t_D'], df['P_wD_gauss'], 'ro', markersize=2, 
                         label='С шумом', alpha=0.6)
            
            if horizontal_line:
                x_limits = ax.get_xlim()
                ax.hlines(y=0.5, xmin=x_limits[0], xmax=x_limits[1], 
                         colors='red', linestyles=':', linewidth=1, label='y=0.5')
            
            ax.set_xlabel('t_D')
            ax.set_ylabel('P_wD')
            ax.set_title(f'Кривая #{num}')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(fontsize=7, loc='best')
            
        except FileNotFoundError:
            axes[idx].text(0.5, 0.5, f'Файл #{num} не найден', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'Ошибка #{num}')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Ошибка: {str(e)[:50]}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'Ошибка #{num}')
    
    # Скрываем неиспользуемые subplot'ы
    for idx in range(len(file_numbers), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Анализ кривых восстановления давления', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    plt.show()


def get_file_template_for_model_type(model_type, num):
    if model_type == 'homogeneous_inf':
        return f'/home/ilnaz/PycharmProjects/datasets/new/homogeneous_inf/curve/homogeneous_inf_{num}.csv'
    if model_type == 'homogeneous_fin':
        return f'/home/ilnaz/PycharmProjects/datasets/new/homogeneous_fin/curve/homogeneous_fin_{num}.csv'
    if model_type == 'dual_porosity_inf':
        return f'/home/ilnaz/PycharmProjects/datasets/new/dual_porosity_inf/curve/dual_porosity_inf_{num}.csv'
    if model_type == 'dual_porosity_fin':
        return f'/home/ilnaz/PycharmProjects/datasets/new/dual_porosity_fin/curve/dual_porosity_fin_{num}.csv'
    if model_type == 'dual_permeability_inf':
        return f'/home/ilnaz/PycharmProjects/datasets/new/dual_permeability_inf/curve/dual_permeability_inf_{num}.csv'
    if model_type == 'dual_permeability_fin':
        return f'/home/ilnaz/PycharmProjects/datasets/new/dual_permeability_fin/curve/dual_permeability_fin_{num}.csv'
    if model_type == 'radial_composite_inf':   
        return f'/home/ilnaz/PycharmProjects/datasets/new/radial_composite_inf/curve/radial_composite_inf_{num}.csv'
    else:
        return None


VALID_MODEL_TYPES = [
    'homogeneous_inf', 'homogeneous_fin',
    'dual_porosity_inf', 'dual_porosity_fin',
    'dual_permeability_inf', 'dual_permeability_fin',
    'radial_composite_inf',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Визуализация кривых давления из датасета')
    parser.add_argument('model_type', choices=VALID_MODEL_TYPES, help='Тип модели пласта')
    parser.add_argument('--start', type=int, default=1, help='Первый номер кривой (default: 1)')
    parser.add_argument('--end', type=int, default=10, help='Последний номер кривой включительно (default: 10)')
    parser.add_argument('--rows', type=int, default=2, help='Строк в сетке (default: 2)')
    parser.add_argument('--cols', type=int, default=5, help='Столбцов в сетке (default: 5)')
    parser.add_argument('--hline', action='store_true', help='Нарисовать горизонтальную линию y=0.5')
    parser.add_argument('--save', type=str, default=None, help='Путь для сохранения графика')
    args = parser.parse_args()

    numbers = list(range(args.start, args.end + 1))
    visualize_multiple_pressures(
        args.model_type, numbers,
        rows=args.rows, cols=args.cols,
        horizontal_line=args.hline,
        save_path=args.save,
    )
