import os
import threading
import functools
import numpy as np

from pandas import DataFrame
from abc import ABC, abstractmethod
from conversion.homogeneous_converter import DimensionConverter


def extract_scalar(value):
    """
    Извлекает скалярное значение из pandas Series, numpy array или другого контейнера.
    
    Args:
        value: значение, которое может быть Series, array, list, tuple или скаляром
        
    Returns:
        скалярное значение
    """
    if hasattr(value, 'iloc'):
        return value.iloc[0]
    elif hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else value
    else:
        return value


def synchronized_method(lock_name):
    """Декоратор для синхронизации метода"""
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self.__class__, lock_name)
            with lock:
                return method(self, *args, **kwargs)
        return wrapper
    return decorator


class ParamGenerator(ABC):
    """
    Генерация параметров пласта
    """
    def __init__(self, size):
        self.size = size

    @abstractmethod
    def generate(self):
        pass


class DataGenerator(ABC):
    """
    Генерация данных по параметрам
    """
    num = 0 # сквозная нумерация всех данных в итоговом датасете
    save_lock = threading.Lock()

    def __init__(self, reservoir_type, t_max_days, points_count, size):
        self.size = size
        self.t_max_days = t_max_days
        self.points_count = points_count
        self.reservoir_type = reservoir_type


    @abstractmethod
    def generate(self):
        pass


    def generate_search_time(self, converter):
        t_max_seconds = self.t_max_days * 24 * 3600
        t_D_max = converter.time_from_dim_to_dimless(t_max_seconds)
        
        # Извлекаем скалярное значение, если это Series
        t_D_max = extract_scalar(t_D_max)
        
        t_D_min = 1e-2
        
        # Проверка на валидность значений
        if t_D_max <= 0 or np.isnan(t_D_max) or np.isinf(t_D_max):
            raise ValueError(f"Invalid t_D_max: {t_D_max}. Check converter parameters.")
        
        if t_D_max < t_D_min:
            # Если максимальное время меньше минимального, расширяем диапазон
            t_D_min = t_D_max / 100
        
        # Проверка на валидность логарифма
        if t_D_max <= 0 or t_D_min <= 0:
            raise ValueError(f"Invalid time range: t_D_min={t_D_min}, t_D_max={t_D_max}")
        
        t_D_array = np.logspace(np.log10(t_D_min), np.log10(t_D_max), self.points_count)
        
        # Проверка на NaN в результате
        if np.any(np.isnan(t_D_array)) or np.any(np.isinf(t_D_array)):
            raise ValueError(f"Generated time array contains NaN or Inf values. t_D_min={t_D_min}, t_D_max={t_D_max}")
        
        return t_D_array


    def get_converter(self, params: DataFrame):
        # Извлекаем скалярные значения из Series/DataFrame
        return DimensionConverter(
            extract_scalar(params['k']),
            extract_scalar(params['h']),
            extract_scalar(params['q']),
            extract_scalar(params['mu']),
            extract_scalar(params['B']),
            extract_scalar(params['p_i']),
            extract_scalar(params['phi']),
            extract_scalar(params['c_t']),
            extract_scalar(params['r_w'])
        )

    @synchronized_method('save_lock')
    def save(self, curve, params):
        """
        Сохраняет данные в файлы в формате:

        - Файл 1: таблица с данными давления и времени
        Название: <тип_пласта>_<num>.csv

        - Файл 2: таблица с параметрами пласта
        Название: <num>.csv
        """
        # Проверка на NaN перед сохранением
        if curve.isna().any().any():
            nan_cols = curve.columns[curve.isna().any()].tolist()
            raise ValueError(f"Curve data contains NaN values in columns: {nan_cols}")
        
        if params.isna().any().any():
            nan_cols = params.columns[params.isna().any()].tolist()
            raise ValueError(f"Params data contains NaN values in columns: {nan_cols}")
        
        curve_dir = './dataset/curve'
        params_dir = './dataset/params'

        os.makedirs(curve_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)

        file_name1 = f'{curve_dir}/{self.reservoir_type}_{self.num}.csv'
        file_name2 = f'{params_dir}/{self.num}.csv'

        curve.to_csv(file_name1, index=False)
        params.to_csv(file_name2, index=False)

        DataGenerator.num += 1

        # print(f"Сохранено: {file_name1}, {file_name2}")
