from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from generation.generator import DataGenerator, ParamGenerator
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.homogeneous.finite_homogeneous_model import FiniteHomogeneousReservoirModel
from model.homogeneous.infinite_homogeneous_model import InfiniteHomogeneousReservoirModel


class HomogeneousModelParamGenerator(ParamGenerator):
    """
    Генерация параметров для гомогенного пласта
    """

    def generate(self) -> List[Dict[str, float]]:
        """
        Вариация параметров с учетом корреляций между ними.
        Например, проницаемость и пористость часто коррелируют.
        """
        base_params = {
            'k': 5e-13,
            'h': 10,
            'q': 2.31e-3,
            'mu': 1e-3,
            'B': 1.2,
            'p_i': 25e6,
            'phi': 0.18,
            'c_t': 1.5e-9,
            'r_w': 0.1
        }

        correlation_groups = [
            ['k', 'phi'],  # проницаемость и пористость часто коррелируют
            ['mu', 'B'],  # вязкость и объемный коэффициент
            ['p_i', 'c_t']  # давление и сжимаемость
        ]

        varied_params_list = []

        for i in range(self.size):
            params = base_params.copy()

            params['h'] *= np.random.uniform(0.5, 2.0)  # толщина
            params['q'] *= np.random.uniform(0.2, 3.0)  # дебит
            params['r_w'] *= np.random.uniform(0.8, 1.2)  # радиус скважины
            params['r_e'] = np.random.uniform(50, 1000)  # радиус границ

            params['C'] = 10 ** np.random.uniform(-9, -7)  # коэффициент влияния ствола скважины

            rand = np.random.random()
            if rand < 0.1:  # 10% - отрицательный скин (стимулированные скважины)
                params['S'] = np.random.uniform(-6, -0.5)
            elif rand < 0.3:  # 20% - нулевой или близкий к нулю
                params['S'] = np.random.uniform(-0.5, 0.5)
            elif rand < 0.8:  # 50% - положительный, но небольшой (1-10)
                params['S'] = np.random.uniform(0.5, 10)
            else:  # 20% - высокий положительный скин (поврежденные скважины)
                params['S'] = np.random.uniform(10, 50)

            for group in correlation_groups:
                group_factor = np.random.uniform(0.5, 2.0)

                for param_name in group:
                    if param_name in params:
                        individual_factor = np.random.uniform(0.9, 1.1)
                        params[param_name] *= group_factor * individual_factor

            params['phi'] = np.clip(params['phi'], 0.05, 0.35)  # пористость 5-35%
            params['mu'] = np.clip(params['mu'], 0.5e-3, 50e-3)  # вязкость 0.5-50 мПа·с
            params['k'] = np.clip(params['k'], 1e-16, 1e-11)  # проницаемость 0.1 мД - 10 Д
            params['B'] = np.clip(params['B'], 1.0, 1.8)  # объемный коэффициент
            params['c_t'] = np.clip(params['c_t'], 0.5e-9, 5e-9)  # сжимаемость

            varied_params_list.append(pd.DataFrame([params]))

        return varied_params_list


class InfiniteHomogeneousGenerator(DataGenerator):
    """
    Генерация данных для бесконечного гомогенного пласта
    """

    def __init__(self, t_max_days, points_count, size):
        self.reservoir_type = 'homogeneous_inf'
        self.t_max_days = t_max_days
        self.size = size
        self.points_count = points_count
        self.param_gen = HomogeneousModelParamGenerator(self.size)


    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating homogeneous infinite reservoir data"):
            param.drop('r_e', axis=1, inplace=True) # убираем информацию о радиусе границы

            converter = self.get_converter(param)
            t_D_array = self.generate_search_time(converter)

            C_D = converter.wellbore_storage_from_dim_to_dimless(param['C'])
            S = param['S']

            model = InfiniteHomogeneousReservoirModel(C_D=C_D, S=S)
            alg = ShtefestAlgorithm(N=16)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)


class FiniteHomogeneousGenerator(DataGenerator):
    """
    Генерация данных для гомогенного пласта с границами
    """

    def __init__(self, t_max_days, points_count, size):
        self.reservoir_type = 'homogeneous_fin'
        self.t_max_days = t_max_days
        self.size = size
        self.points_count = points_count
        self.param_gen = HomogeneousModelParamGenerator(self.size)


    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating homogeneous finite reservoir data"):
            converter = self.get_converter(param)
            t_D_array = self.generate_search_time(converter)

            C_D = converter.wellbore_storage_from_dim_to_dimless(param['C'])
            r_D_e = converter.reservoir_radius_from_dim_to_dimless(param['r_e'])
            S = param['S']

            model = FiniteHomogeneousReservoirModel(C_D=C_D, S=S, R_D_E=r_D_e)
            alg = ShtefestAlgorithm(N=16)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)
