import os
from abc import ABC, abstractmethod

import numpy as np

from generation.utils import extract_scalar


class GenerationParams:
    """
    Параметры генерации пласта
    """

    def __init__(self, t_max_days: int, points_count: int, size: int, sigma: float, output_path: str):
        self.t_max_days = t_max_days
        self.points_count = points_count
        self.size = size
        self.sigma = sigma
        self.output_path = output_path

    def __str__(self):
        return (f'Generation params: days: {self.t_max_days}, size: {self.size}, points_count: {self.points_count}, '
                f'sigma: {self.sigma}, output_path: {self.output_path}')


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
    _global_num = 0  # сквозная нумерация всех данных в итоговом датасете

    def __init__(self, reservoir_type, params: GenerationParams):
        self.size = params.size
        self.sigma = params.sigma
        self.t_max_days = params.t_max_days
        self.points_count = params.points_count
        self.reservoir_type = reservoir_type
        self.output_path = params.output_path

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
        num = self.get_next_number()

        file_name1 = f'{self.curve_dir}/{self.reservoir_type}_{num}.csv'
        file_name2 = f'{self.params_dir}/{num}.csv'

        curve.to_csv(file_name1, index=False)
        params.to_csv(file_name2, index=False)

    @classmethod
    def get_next_number(cls):
        """Получить следующий номер"""
        DataGenerator._global_num += 1
        return DataGenerator._global_num

    @classmethod
    def reset_global_counter(cls):
        """Обнулить общий счетчик"""
        DataGenerator._global_num = 0
