from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.homogeneous_converter import HomogeneousConverter
from generation.generator import extract_scalar
from generation.generator import DataGenerator, ParamGenerator
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.homogeneous.finite_homogeneous_model import FiniteHomogeneousReservoirModel
from model.homogeneous.infinite_homogeneous_model import InfiniteHomogeneousReservoirModel


class HomogeneousModelParamGenerator(ParamGenerator):
    """
    Генерация параметров для гомогенного пласта
    """

    def generate(self) -> List[Dict[str, float]]:
        base_params = {
            'k': 5e-13,
            'h': 10,
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
            params['q'] = np.random.uniform(5, 50)  # дебит
            params['r_e'] = np.random.uniform(100, 1000)  # радиус границ

            # коэффициент влияния ствола скважины
            params['C'] = 10 ** np.random.uniform(-9, -7)

            # скин-фактор
            params['S'] = np.random.uniform(0, 10)

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

    def __init__(self, t_max_days, points_count, size, output_path):
        super().__init__('homogeneous_inf', t_max_days, points_count, size, output_path)
        self.param_gen = HomogeneousModelParamGenerator(self.size)


    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating homogeneous infinite reservoir data"):
            param.drop('r_e', axis=1, inplace=True) # убираем информацию о радиусе границы

            converter = HomogeneousConverter(
                extract_scalar(param['k']),
                extract_scalar(param['h']),
                extract_scalar(param['q']),
                extract_scalar(param['mu']),
                extract_scalar(param['B']),
                extract_scalar(param['p_i']),
                extract_scalar(param['phi']),
                extract_scalar(param['c_t']),
                extract_scalar(param['r_w'])
            )
            t_D_array = self.generate_search_time(converter)

            C_D = extract_scalar(converter.wellbore_storage_from_dim_to_dimless(extract_scalar(param['C'])))
            S = extract_scalar(param['S'])

            model = InfiniteHomogeneousReservoirModel(C_D=C_D, S=S)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)


class FiniteHomogeneousGenerator(DataGenerator):
    """
    Генерация данных для гомогенного пласта с границами
    """

    def __init__(self, t_max_days, points_count, size, output_path):
        super().__init__('homogeneous_fin', t_max_days, points_count, size, output_path)
        self.param_gen = HomogeneousModelParamGenerator(self.size)


    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating homogeneous finite reservoir data"):
            converter = HomogeneousConverter(
                extract_scalar(param['k']),
                extract_scalar(param['h']),
                extract_scalar(param['q']),
                extract_scalar(param['mu']),
                extract_scalar(param['B']),
                extract_scalar(param['p_i']),
                extract_scalar(param['phi']),
                extract_scalar(param['c_t']),
                extract_scalar(param['r_w'])
            )
            t_D_array = self.generate_search_time(converter)

            C_D = extract_scalar(converter.wellbore_storage_from_dim_to_dimless(extract_scalar(param['C'])))
            r_D_e = extract_scalar(converter.reservoir_radius_from_dim_to_dimless(extract_scalar(param['r_e'])))
            S = extract_scalar(param['S'])

            model = FiniteHomogeneousReservoirModel(C_D=C_D, S=S, R_D_E=r_D_e)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)
