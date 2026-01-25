import os
from abc import ABC, abstractmethod

import numpy as np

from generation.utils import extract_scalar


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

    def __init__(self, reservoir_type, t_max_days, points_count, size, output_path):
        self.size = size
        self.t_max_days = t_max_days
        self.points_count = points_count
        self.reservoir_type = reservoir_type
        self.output_path = output_path

        self.curve_dir = os.path.join(self.output_path, 'curve')
        self.params_dir = os.path.join(self.output_path, 'params')

        os.makedirs(self.curve_dir, exist_ok=True)
        os.makedirs(self.params_dir, exist_ok=True)


    @abstractmethod
    def generate(self):
        pass


    def generate_search_time(self, converter):
        t_max_seconds = self.t_max_days * 24 * 3600
        t_D_max = converter.time_from_dim_to_dimless(t_max_seconds)

        t_D_max = extract_scalar(t_D_max)
        t_D_min = 1e-1

        t_D_array = np.logspace(np.log10(t_D_min), np.log10(t_D_max), self.points_count)

        return t_D_array

    def save(self, curve, params):
        """
        Сохраняет данные в файлы в формате:

        - Файл 1: таблица с данными давления и времени
        Название: <тип_пласта>_<num>.csv

        - Файл 2: таблица с параметрами пласта
        Название: <num>.csv
        """
        file_name1 = f'{self.curve_dir}/{self.reservoir_type}_{self.num}.csv'
        file_name2 = f'{self.params_dir}/{self.num}.csv'

        curve.to_csv(file_name1, index=False)
        params.to_csv(file_name2, index=False)

        DataGenerator.num += 1
