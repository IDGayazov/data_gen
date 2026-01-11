from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.radial_composite_converter import RadialCompositeConverter
from generation.generator import ParamGenerator, DataGenerator
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
            'q': 2.31e-3,  # дебит (м³/с)
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
            params['q'] *= np.random.uniform(0.2, 3.0)  # дебит
            params['r_w'] *= np.random.uniform(0.8, 1.2)  # радиус скважины: 0.08-0.12 м
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
            rand = np.random.random()
            if rand < 0.1:
                params['S'] = np.random.uniform(0, 0.1)
            elif rand < 0.3:
                params['S'] = np.random.uniform(0.1, 0.5)
            else:
                params['S'] = np.random.uniform(0.5, 10)


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

                # Геометрия
                R_i=params['R_i'],
                r_e=params.get('r_e', 1000)
            )

            result_params = {
                'C_D': converter.wellbore_storage_from_dim_to_dimless(params['C']),
                'S': params['S'],
                'M': converter.M,
                'M1': converter.M1,
                'M2': converter.M2,
                'omega1': converter.storage1,
                'omega2': converter.storage2,
                'r_fD': converter.r_fD,

                'eta1': converter.eta1,
                'eta2': converter.eta2,
                'eta_ratio': converter.eta_ratio,

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

            # Пьезопроводности
            result_params['eta1'] = np.clip(result_params.get('eta1', 0.1), 0.01, 100)
            result_params['eta2'] = np.clip(result_params.get('eta2', 0.1), 0.01, 100)
            result_params['eta_ratio'] = np.clip(result_params.get('eta_ratio', 1.0), 0.01, 100)

            df = pd.DataFrame([result_params])
            varied_params_list.append(df)

        return varied_params_list


class InfiniteRadialCompositeGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта радиально-композитной модели
    """

    def __init__(self, t_max_days, points_count, size):
        self.reservoir_type = 'radial_composite_inf'
        self.t_max_days = t_max_days
        self.size = size
        self.points_count = points_count
        self.param_gen = RadialCompositeParamGenerator(self.size)


    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual radial composite infinite reservoir data"):
            converter=RadialCompositeConverter(
                # Общие параметры
                h=param['h'].iloc[0],
                q=param['q'].iloc[0],
                mu=param['mu'].iloc[0],
                B=param['B'].iloc[0],
                p_i=param['p_i'].iloc[0],
                r_w=param['r_w'].iloc[0],

                # Зона 1
                k1=param['k1'].iloc[0],
                phi1=param['phi1'].iloc[0],
                c_t1=param['c_t1'].iloc[0],

                # Зона 2
                k2=param['k2'].iloc[0],
                phi2=param['phi2'].iloc[0],
                c_t2=param['c_t2'].iloc[0],

                # Геометрия
                R_i=param['R_i'].iloc[0],
                r_e=param['r_e'].iloc[0]
            )

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            S = param['S'][0]

            M1 = param['M1'][0]
            M2 =  param['M2'][0]
            omega1 = param['omega1'][0]
            omega2 = param['omega2'][0]
            r_fD = param['r_fD'][0]

            model = InfiniteRadialCompositeReservoirModel(C_D=C_D, S=S, M1=M1, M2=M2, omega1=omega1, omega2=omega2, r_fD=r_fD)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)
