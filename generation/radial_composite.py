from typing import List

import numpy as np
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
        base_params = {
            # Общие параметры
            'h': 10,  # толщина (м)
            'mu': 1e-3,  # вязкость (Па·с)
            'B': 1.2,  # объемный коэффициент
            'p_i': 25e6,  # начальное давление (Па)
            'r_w': 0.1,  # радиус скважины (м)

            # Зона 1 (внутренняя)
            'k1': 5e-13,  # проницаемость зоны 1 (м²) ~ 500 мД
            'phi1': 0.15,  # пористость зоны 1 (15%)
            'c_t1': 1e-9,  # сжимаемость зоны 1 (1/Па)

            # Зона 2 (внешняя)
            'k2': 1e-13,  # проницаемость зоны 2 (м²) ~ 100 мД
            'phi2': 0.12,  # пористость зоны 2 (12%)
            'c_t2': 1.5e-9,  # сжимаемость зоны 2 (1/Па)
        }

        varied_params_list = []

        for i in range(self.size):
            params = base_params.copy()

            # Вариация основных параметров
            params['h'] *= np.random.uniform(0.5, 2.0)  # толщина: 5-20 м
            params['q'] = np.random.uniform(5, 50)  # дебит
            params['r_e'] = np.random.uniform(100, 1000)  # внешняя граница: 100-1000 м
            params['mu'] *= np.random.uniform(0.5, 50) # вязкость флюида

            M12 = np.random.uniform(1, 5)
            params['M1'] = 1.0
            params['M2'] = params['M1'] / M12

            # Коэффициент влияния ствола
            params['C'] = 10 ** np.random.uniform(-9, -7)  # м³/Па

            params['phi1'] = np.random.uniform(0.1, 0.25)  # 10-25%
            params['phi2'] = np.random.uniform(0.1, 0.25)

            params['c_t1'] = np.random.uniform(0.5e-10, 5e-9)
            params['c_t2'] = np.random.uniform(0.5e-10, 5e-9)

            params['p_i'] *= np.random.uniform(0.5, 2)

            # Скин-фактор
            params['S'] = np.random.uniform(0, 10)

            # Радиус интерфейса (должен быть между r_w и r_e)
            params['R_i'] = np.random.uniform(5, params['r_e'] * 0.7)

            # Зона 1
            params['phi1'] = np.clip(params['phi1'], 0.05, 0.35)  # 5-35%
            params['k1'] = np.clip(params['k1'], 1e-14, 1e-11)  # 10 мД - 10 Д
            params['c_t1'] = np.clip(params['c_t1'], 0.5e-10, 5e-9)  # сжимаемость

            # Зона 2
            params['phi2'] = np.clip(params['phi2'], 0.05, 0.35)  # 5-35%
            params['k2'] = np.clip(params['k2'], 1e-14, 1e-11)  # 10 мД - 10 Д
            params['c_t2'] = np.clip(params['c_t2'], 0.5e-10, 5e-9)  # сжимаемость

            # Флюидные свойства
            params['mu'] = np.clip(params['mu'], 0.5e-3, 50e-3)  # 0.5-50 мПа·с
            params['B'] = np.clip(params['B'], 1.0, 1.8)  # объемный коэффициент

            # корректировка параметров
            params['k1'] = params['M1'] * params['mu'] / params['h']
            params['k2'] = params['M2'] * params['mu'] / params['h']

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

                R_i=params['R_i'],
                r_e=params.get('r_e', 1000)
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
                'r_e': params.get('r_e', 1000),
                'r_w': params['r_w'],
                'h': params['h'],
                'q': params['q'],
                'mu': params['mu'],
                'B': params['B'],
                'p_i': params['p_i'],
                'C': params['C']
            }

            # r_fD = R_i/r_w
            result_params['r_fD'] = np.clip(result_params['r_fD'], 10, 100)

            df = pd.DataFrame([result_params])
            varied_params_list.append(df)

        return varied_params_list


class InfiniteRadialCompositeGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта радиально-композитной модели
    """
    def __init__(self, params: GenerationParams):
        super().__init__('radial_composite_inf', params)
        self.param_gen = RadialCompositeParamGenerator(self.size)

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
                R_i=extract_scalar(param['R_i']),
                r_e=extract_scalar(param['r_e'])
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
                .gauss_noize(mu=0, sigma=self.sigma) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)
