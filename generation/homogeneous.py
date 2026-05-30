from typing import Dict, List

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.homogeneous_converter import HomogeneousConverter
from generation.generator import extract_scalar, GenerationParams
from generation.generator import DataGenerator, ParamGenerator
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.homogeneous.finite_homogeneous_model import FiniteHomogeneousReservoirModel
from model.homogeneous.infinite_homogeneous_model import InfiniteHomogeneousReservoirModel


class HomogeneousModelParamGenerator(ParamGenerator):
    """
    Генерация параметров для гомогенного пласта в заданных интервалах
    """

    def generate(self) -> List[Dict[str, float]]:
        train_ranges = {
            'k': [(0.01, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9)], # проницаемость, 1e-12*м^2 
            'h': [(5.0, 30.0)],         # толщина, м
            'phi': [(0.1, 0.3)],        # пористость, д.ед.
            'mu': [(0.5e-3, 10e-3)],    # вязкость, Па*с
            'B': [(1.0, 1.5)],          # объемный коэфф.
            'p_i': [(15e6, 40e6)],      # нач. давление, Па
            'c_t': [(1e-10, 5e-9)],     # общая сжимаемость, 1/Па
            'q': [(10, 100)],           # дебит, м3/сут
            'S': [(0.1, 1), (2, 3), (4, 5), (6, 7), (8, 9)], # скин-фактор
            'C_D': [(200, 300), (400, 500), (600, 700), (800, 900)], # эффект влияния ствола скважины
            'r_e': [(200, 300), (400, 500), (600, 700), (800, 900)] # радиус контура, м
        }

        val_ranges = {
            'k': [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1)], # проницаемость, 1e-12*м^2 
            'h': [(5.0, 30.0)],         # толщина, м
            'phi': [(0.1, 0.3)],        # пористость, д.ед.
            'mu': [(0.5e-3, 10e-3)],    # вязкость, Па*с
            'B': [(1.0, 1.5)],          # объемный коэфф.
            'p_i': [(15e6, 40e6)],      # нач. давление, Па
            'c_t': [(1e-10, 5e-9)],     # общая сжимаемость, 1/Па
            'q': [(10, 100)],           # дебит, м3/сут
            'S': [(1, 2), (3, 4), (5, 6), (7, 8)], # скин-фактор
            'C_D': [(300, 400), (500, 600), (700, 800), (900, 1000)], # эффект влияния ствола скважины
            'r_e': [(300, 400), (500, 600), (700, 800), (900, 1000)] # радиус контура, м
        }

        test_ranges = {
            'k': [(0.1, 1)], # проницаемость, 1e-12*м^2 
            'h': [(5.0, 30.0)],         # толщина, м
            'phi': [(0.1, 0.3)],        # пористость, д.ед.
            'mu': [(0.5e-3, 10e-3)],    # вязкость, Па*с
            'B': [(1.0, 1.5)],          # объемный коэфф.
            'p_i': [(15e6, 40e6)],      # нач. давление, Па
            'c_t': [(1e-10, 5e-9)],     # общая сжимаемость, 1/Па
            'q': [(10, 100)],           # дебит, м3/сут
            'S': [(1, 8)], # скин-фактор
            'C_D': [(200, 1000)], # эффект влияния ствола скважины
            'r_e': [(200, 1000)] # радиус контура, м
        }

        varied_params_list = []

        ranges = train_ranges       
        if self.type == 'val':
            ranges = val_ranges
        if self.type == 'test':
            ranges = test_ranges

        for _ in range(self.size):
            params = {}
            
            for param, param_list in ranges.items():
                low, high = random.choice(param_list)
                
                if param == 'k':
                    params[param] = np.random.uniform(low, high) * 1e-12
                else:
                    params[param] = np.random.uniform(low, high)
            
            params['r_w'] = 0.1
            denominator = (2 * np.pi * params['h'] * params['phi'] *
                          params['c_t'] * params['r_w']**2)
            params['C'] = params['C_D'] * denominator
            params['data_type'] = self.type
            varied_params_list.append(pd.DataFrame([params]))

        return varied_params_list


class InfiniteHomogeneousGenerator(DataGenerator):
    """
    Генерация данных для бесконечного гомогенного пласта
    """
    def __init__(self, params: GenerationParams):
        super().__init__('homogeneous_inf', params)
        self.param_gen = HomogeneousModelParamGenerator(self.size, self.type)

    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating homogeneous infinite reservoir data"):
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
            param['C_D'] = C_D
            param['r_D_e'] = 0

            model = InfiniteHomogeneousReservoirModel(C_D=C_D, S=S)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma_mpa=self.sigma, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)


class FiniteHomogeneousGenerator(DataGenerator):
    """
    Генерация данных для гомогенного пласта с границами
    """
    def __init__(self, params: GenerationParams):
        super().__init__('homogeneous_fin', params)
        self.param_gen = HomogeneousModelParamGenerator(self.size, self.type)


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
            param['C_D'] = C_D
            param['r_D_e'] = r_D_e

            model = FiniteHomogeneousReservoirModel(C_D=C_D, S=S, R_D_E=r_D_e)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma_mpa=self.sigma, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)
