from typing import List

import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from conversion.radial_composite_converter import RadialCompositeConverter
from generation.generator import ParamGenerator, DataGenerator, GenerationParams
from generation.utils import extract_scalar
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.radialcomposite.infinite_radial_composite_model import InfiniteRadialCompositeReservoirModel


class RadialCompositeParamGenerator(ParamGenerator):
    """
    Генерация параметров для радиально-композитной модели
    """

    def generate(self) -> List[pd.DataFrame]:
        train_ranges = {
            'h' : [(10, 15)],
            'k1' : [(0.01, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9)],
            'phi1' : [(0.1, 0.25)],
            'c_t1' : [(0.5e-10, 5e-9)],

            'k2' : [(1e-14, 1e-13)],
            'phi2' : [(0.1, 0.25)],
            'c_t2' : [(0.5e-10, 5e-9)],
            'q' : [(5, 50)],

            'B' : [(1.0, 1.8)],
            'p_i' : [(15e6, 40e6)],
            'mu' : [(0.5e-3, 50e-3)],
            'R_i' : [(10, 25), (40, 55), (70, 85)],
            'S': [(0.1, 1), (2, 3), (4, 5), (6, 7), (8, 9)],

            'C_D': [(200, 300), (400, 500), (600, 700), (800, 900)],
            'omega1' : [(0.05, 0.15), (0.25, 0.35), (0.45, 0.55), (0.65, 0.75), (0.85, 0.95)],
            'M2' : [(0.1, 0.5), (1.5, 3.0), (5.0, 7.0), (9.0, 10.0)],
        }

        val_ranges = {
            'h' : [(10, 15)],
            'k1' : [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1)],
            'phi1' : [(0.1, 0.25)],
            'c_t1' : [(0.5e-10, 5e-9)],

            'k2' : [(1e-14, 1e-13)],
            'phi2' : [(0.1, 0.25)],
            'c_t2' : [(0.5e-10, 5e-9)],

            'q' : [(5, 50)],
            'B' : [(1.0, 1.8)],
            'p_i' : [(15e6, 40e6)],
            'mu' : [(0.5e-3, 50e-3)],
            'R_i' : [(25, 40), (55, 70), (85, 100)],
            'S': [(1, 2), (3, 4), (5, 6), (7, 8)],

            'C_D': [(300, 400), (500, 600), (700, 800), (900, 1000)],
            'omega1' : [(0.15, 0.25), (0.35, 0.45), (0.55, 0.65), (0.75, 0.85)],
            'M2' : [(0.5, 1.5), (3.0, 5.0), (7.0, 9.0)],
        }

        test_ranges = {
            'h' : [(10, 15)],
            'k1' : [(0.1, 1)],
            'phi1' : [(0.1, 0.25)],
            'c_t1' : [(0.5e-10, 5e-9)],

            'k2' : [(1e-14, 1e-13)],
            'phi2' : [(0.1, 0.25)],
            'c_t2' : [(0.5e-10, 5e-9)],
            'q' : [(5, 50)],

            'B' : [(1.0, 1.8)],
            'p_i' : [(15e6, 40e6)],
            'mu' : [(0.5e-3, 50e-3)],
            'S': [(1, 9)],

            'C_D': [(200, 1000)],
            'omega1' : [(0.05, 0.95)],
            'M2' : [(0.1, 10)],
            'R_i' : [(10, 100)]
        }

        if self.type == 'val':
            ranges = val_ranges
        elif self.type == 'test':
            ranges = test_ranges
        else:
            ranges = train_ranges
        
        varied_params_list = []
        
        for _ in range(self.size):
            params = {}
            
            # 1. Генерация первичных параметров из интервалов
            for param_name, param_intervals in ranges.items():
                if not param_intervals: continue 
                low, high = random.choice(param_intervals)
                
                if param_name == 'k1':
                    params[param_name] = np.random.uniform(low, high) * 1e-12
                else:
                    params[param_name] = np.random.uniform(low, high)

            # 2. Фиксируем базовые параметры
            params['r_w'] = 0.1  # радиус скважины, м

            # 2. Определение зависимых физических параметров
            # M12 = (k/mu)1 / (k/mu)2 => k2 = k1 / (M12 * (mu2/mu1))
            # Для упрощения часто mu1 = mu2, тогда k2 = k1 / M12
            params['k2'] = params['k1'] / params['M2']
            
            # omega12 = (phi*ct)1 / (phi*ct)2
            # Чтобы не менять phi1/ct1, определим phi2*ct2 через omega1
            stor_zone1 = params['phi1'] * params['c_t1']
            params['phi2'] = params['phi1'] # упрощение
            # Согласно формуле 57: omega12 = stor1 / stor2
            # Если использовать логику omega1 как долю:
            # total_stor = stor1 + stor2; omega1 = stor1 / total_stor
            # Тогда stor2 = stor1 * (1 - omega1) / omega1
            params['c_t2'] = params['c_t1'] * (1 - params['omega1']) / params['omega1']

            # 3. Расчет размерного Wellbore Storage (C)
            phi_ct2 = params['phi2'] * params['c_t2']
            params['C'] = params['C_D'] * 2 * np.pi * phi_ct2 * params['h'] * params['r_w']**2

            converter = RadialCompositeConverter(
                # Общие параметры
                h=params['h'],
                q=params['q'],
                mu=params['mu'],
                B=params['B'],
                p_i=params['p_i'],
                r_w=params['r_w'],

                # Зона 1
                k1=params['k1'],
                phi1=params['phi1'],
                c_t1=params['c_t1'],

                # Зона 2
                k2=params['k2'],
                phi2=params['phi2'],
                c_t2=params['c_t2'],

                R_i=params['R_i']
            )

            result_params = {
                'C_D': converter.wellbore_storage_from_dim_to_dimless(params['C']),
                'S': params['S'],
                'M1': converter.M1,
                'M2': converter.M2,
                'omega1': converter.storage1,
                'omega2': converter.storage2,
                'r_fD': converter.r_fD,
                'k1': params['k1'],
                'k2': params['k2'],
                'phi1': params['phi1'],
                'phi2': params['phi2'],
                'c_t1': params['c_t1'],
                'c_t2': params['c_t2'],
                'R_i': params['R_i'],
                'r_w': params['r_w'],
                'h': params['h'],
                'q': params['q'],
                'mu': params['mu'],
                'B': params['B'],
                'p_i': params['p_i'],
                'C': params['C']
            }

            result_params['M12'] = converter.M1 / converter.M2
            result_params['omega12'] = converter.storage1 / converter.storage2
            result_params['data_type'] = self.type

            df = pd.DataFrame([result_params])
            varied_params_list.append(df)

        return varied_params_list
    

class InfiniteRadialCompositeGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта радиально-композитной модели
    """
    def __init__(self, params: GenerationParams):
        super().__init__('radial_composite_inf', params)
        self.param_gen = RadialCompositeParamGenerator(self.size, self.type)

    def generate_search_time(self, converter):
        t_min_seconds = 0.3
        t_max_seconds = self.t_max_days * 24 * 3600

        t_D_max = extract_scalar(converter.time_from_dim_to_dimless(t_max_seconds, use_zone=1))
        t_D_min = converter.time_from_dim_to_dimless(t_min_seconds, use_zone=1)

        return np.logspace(np.log10(t_D_min), np.log10(t_D_max), self.points_count)

    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating radial composite infinite reservoir data"):
            converter=RadialCompositeConverter(
                # Общие параметры
                h=extract_scalar(param['h']),
                q=extract_scalar(param['q']),
                mu=extract_scalar(param['mu']),
                B=extract_scalar(param['B']),
                p_i=extract_scalar(param['p_i']),
                r_w=extract_scalar(param['r_w']),

                # Зона 1
                k1=extract_scalar(param['k1']),
                phi1=extract_scalar(param['phi1']),
                c_t1=extract_scalar(param['c_t1']),

                # Зона 2
                k2=extract_scalar(param['k2']),
                phi2=extract_scalar(param['phi2']),
                c_t2=extract_scalar(param['c_t2']),

                # Геометрия
                R_i=extract_scalar(param['R_i'])
            )

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            S = param['S'][0]

            M1 = param['M1'][0]
            M2 =  param['M2'][0]
            omega1 = param['omega1'][0]
            omega2 = param['omega2'][0]
            r_fD = param['r_fD'][0]

            model = InfiniteRadialCompositeReservoirModel(C_D=C_D,
                                                          S=S,
                                                          M1=M1,
                                                          M2=M2,
                                                          omega1=omega1,
                                                          omega2=omega2,
                                                          r_fD=r_fD)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma_mpa=self.sigma, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)
