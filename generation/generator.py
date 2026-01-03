import os
import threading
import functools
import numpy as np

from pandas import DataFrame
from abc import ABC, abstractmethod
from conversion.homogeneous_converter import DimensionConverter


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
        t_D_min = 1e-2
        t_D_array = np.logspace(np.log10(t_D_min), np.log10(t_D_max), self.points_count)
        return t_D_array


    def get_converter(self, params: DataFrame):
        return DimensionConverter(
            params['k'],
            params['h'],
            params['q'],
            params['mu'],
            params['B'],
            params['p_i'],
            params['phi'],
            params['c_t'],
            params['r_w']
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
        curve_dir = '../dataset/curve'
        params_dir = '../dataset/params'

        os.makedirs(curve_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)

        file_name1 = f'{curve_dir}/{self.reservoir_type}_{self.num}.csv'
        file_name2 = f'{params_dir}/{self.num}.csv'

        curve.to_csv(file_name1, index=False)
        params.to_csv(file_name2, index=False)

        DataGenerator.num += 1

        # print(f"Сохранено: {file_name1}, {file_name2}")
