from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.radial_composite_converter import RadialCompositeDualPorosityConverter
from generation.generator import ParamGenerator, DataGenerator
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.radialcomposite.infinite_radial_composite_model import InfiniteRadialCompositeReservoirModel


class RadialCompositeModelParamGenerator(ParamGenerator):
    """
    Генерация параметров для радиально-композитной модели
    с двойной пористостью в обеих зонах и РАЗНЫМИ проницаемостями
    """

    def generate(self) -> List[pd.DataFrame]:
        """
        Генерация данных для радиально-композитного пласта
        с двойной пористостью в обеих зонах.
        Использует RadialCompositeDualPorosityConverter.
        """
        # Базовые параметры
        base_params = {
            # Общие параметры
            'h': 10,  # толщина (м)
            'q': 2.31e-3,  # дебит (м³/с)
            'mu': 1e-3,  # вязкость (Па·с)
            'B': 1.2,  # объемный коэффициент
            'p_i': 25e6,  # начальное давление (Па)
            'r_w': 0.1,  # радиус скважины (м)

            # Зона 1 (внутренняя) - трещины
            'k_f1': 5e-13,  # проницаемость трещин зоны 1 (м²)
            'phi_f1': 0.02,  # пористость трещин зоны 1
            'c_tf1': 1e-9,  # сжимаемость трещин зоны 1 (1/Па)

            # Зона 1 (внутренняя) - матрица
            'k_m1': 5e-15,  # проницаемость матрицы зоны 1 (м²)
            'phi_m1': 0.18,  # пористость матрицы зоны 1
            'c_tm1': 1.5e-9,  # сжимаемость матрицы зоны 1 (1/Па)
            'alpha1': 12.0,  # shape factor зона 1 (1/м²)

            # Зона 2 (внешняя) - трещины
            'k_f2': 2e-13,  # проницаемость трещин зоны 2 (м²)
            'phi_f2': 0.015,  # пористость трещин зоны 2
            'c_tf2': 1e-9,  # сжимаемость трещин зоны 2 (1/Па)

            # Зона 2 (внешняя) - матрица
            'k_m2': 2e-15,  # проницаемость матрицы зоны 2 (м²)
            'phi_m2': 0.15,  # пористость матрицы зоны 2
            'c_tm2': 1.5e-9,  # сжимаемость матрицы зоны 2 (1/Па)
            'alpha2': 12.0,  # shape factor зона 2 (1/м²)

            # Геометрические параметры
            'R_i': 50.0,  # радиус интерфейса (м)

            # Коэффициент влияния ствола
            'C': 1e-8,  # м³/Па

            # Скин-фактор
            'S': 0.0,  # по умолчанию
        }

        # Группы корреляции
        correlation_groups = [
            ['k_f1', 'phi_f1', 'alpha1'],  # параметры трещин зоны 1
            ['k_m1', 'phi_m1', 'c_tm1'],  # параметры матрицы зоны 1
            ['k_f2', 'phi_f2', 'alpha2'],  # параметры трещин зоны 2
            ['k_m2', 'phi_m2', 'c_tm2'],  # параметры матрицы зоны 2
            ['mu', 'B'],  # флюидные свойства
            ['p_i', 'c_tf1', 'c_tf2'],  # давление и сжимаемости трещин
        ]

        varied_params_list = []

        for i in range(self.size):
            params = base_params.copy()

            # Вариация основных параметров
            params['h'] *= np.random.uniform(0.5, 2.0)  # толщина
            params['q'] *= np.random.uniform(0.2, 3.0)  # дебит
            params['r_w'] *= np.random.uniform(0.8, 1.2)  # радиус скважины
            params['r_e'] = np.random.uniform(100, 2000)  # внешняя граница

            # Коэффициент влияния ствола
            params['C'] = 10 ** np.random.uniform(-9, -7)  # м³/Па

            # Скин-фактор (генерируем реалистичные значения)
            rand = np.random.random()
            if rand < 0.1:  # 10% - отрицательный скин
                params['S'] = np.random.uniform(0, 0.1)
            elif rand < 0.3:  # 20% - нулевой или близкий
                params['S'] = np.random.uniform(0.1, 0.5)
            elif rand < 0.8:  # 50% - небольшой положительный
                params['S'] = np.random.uniform(0.5, 10)
            else:  # 20% - высокий положительный
                params['S'] = np.random.uniform(10, 50)

            # Коррелированная вариация параметров
            for group in correlation_groups:
                group_factor = np.random.uniform(0.5, 2.0)
                for param_name in group:
                    if param_name in params:
                        individual_factor = np.random.uniform(0.9, 1.1)
                        params[param_name] *= group_factor * individual_factor

            # Радиус интерфейса (должен быть между r_w и r_e)
            params['R_i'] = np.random.uniform(5, min(500, params['r_e'] * 0.8))

            # Гарантируем, что R_i < r_e
            while params['R_i'] >= params['r_e'] * 0.95:
                params['R_i'] *= 0.9

            # Клиппинг для реалистичных диапазонов
            # Зона 1 - трещины
            params['phi_f1'] = np.clip(params['phi_f1'], 0.001, 0.05)  # 0.1-5%
            params['k_f1'] = np.clip(params['k_f1'], 1e-14, 1e-11)  # 10 мД - 10 Д
            params['c_tf1'] = np.clip(params['c_tf1'], 0.5e-10, 5e-9)  # сжимаемость

            # Зона 1 - матрица
            params['phi_m1'] = np.clip(params['phi_m1'], 0.05, 0.35)  # 5-35%
            params['k_m1'] = np.clip(params['k_m1'], 1e-17, 1e-13)  # 0.01 мД - 100 мД
            params['c_tm1'] = np.clip(params['c_tm1'], 0.5e-9, 5e-9)  # сжимаемость
            params['alpha1'] = np.random.choice([4, 12, 32])  # геометрический фактор

            # Зона 2 - трещины
            params['phi_f2'] = np.clip(params['phi_f2'], 0.001, 0.05)
            params['k_f2'] = np.clip(params['k_f2'], 1e-14, 1e-11)
            params['c_tf2'] = np.clip(params['c_tf2'], 0.5e-10, 5e-9)

            # Зона 2 - матрица
            params['phi_m2'] = np.clip(params['phi_m2'], 0.05, 0.35)
            params['k_m2'] = np.clip(params['k_m2'], 1e-17, 1e-13)
            params['c_tm2'] = np.clip(params['c_tm2'], 0.5e-9, 5e-9)
            params['alpha2'] = np.random.choice([4, 12, 32])

            # Флюидные свойства
            params['mu'] = np.clip(params['mu'], 0.5e-3, 50e-3)  # 0.5-50 мПа·с
            params['B'] = np.clip(params['B'], 1.0, 1.8)  # объемный коэффициент

            # Скин-фактор
            params['S'] = np.clip(params['S'], -10, 100)

            converter = RadialCompositeDualPorosityConverter(
                # Общие параметры
                h=params['h'],
                q=params['q'],
                mu=params['mu'],
                B=params['B'],
                p_i=params['p_i'],

                # Зона 1
                k_f1=params['k_f1'],
                k_m1=params['k_m1'],
                phi_f1=params['phi_f1'],
                phi_m1=params['phi_m1'],
                c_tf1=params['c_tf1'],
                c_tm1=params['c_tm1'],
                alpha1=params['alpha1'],

                # Зона 2
                k_f2=params['k_f2'],
                k_m2=params['k_m2'],
                phi_f2=params['phi_f2'],
                phi_m2=params['phi_m2'],
                c_tf2=params['c_tf2'],
                c_tm2=params['c_tm2'],
                alpha2=params['alpha2'],

                # Геометрия
                r_w=params['r_w'],
                R_i=params['R_i'],
                r_e=params['r_e']
            )

            result_params = {
                'C_D': converter.wellbore_storage_from_dim_to_dimless(params['C']),
                'S': params['S'],
                'M1': converter.M1,
                'M2': converter.M2,
                'omega1': converter.omega1,
                'omega2': converter.omega2,
                'r_fD': converter.r_fD
            }

            result_params['C_D'] = np.clip(result_params['C_D'], 1, 10000)
            result_params['S'] = np.clip(result_params['S'], -10, 50)
            result_params['omega1'] = np.clip(result_params['omega1'], 0.001, 0.5)
            result_params['omega2'] = np.clip(result_params['omega2'], 0.001, 0.5)
            result_params['M1'] = np.clip(result_params['M1'], 1e-6, 1e-2)
            result_params['M2'] = np.clip(result_params['M2'], 1e-6, 1e-2)
            result_params['r_fD'] = np.clip(result_params['r_fD'], 10, 1000)

            result_params['k_f1'] = params['k_f1']
            result_params['k_f2'] = params['k_f2']
            result_params['R_i'] = params['R_i']
            result_params['phi_total1'] = params['phi_f1'] + params['phi_m1']
            result_params['phi_total2'] = params['phi_f2'] + params['phi_m2']

            df = pd.DataFrame([result_params])
            varied_params_list.append(df)

        return varied_params_list, converter


class InfiniteRadialCompositeGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта радиально-композитной модели
    """

    def __init__(self, t_max_days, points_count, size):
        self.reservoir_type = 'radial_composite_inf'
        self.t_max_days = t_max_days
        self.size = size
        self.points_count = points_count
        self.param_gen = RadialCompositeModelParamGenerator(self.size)


    def generate(self):
        params, converter = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual radial composite infinite reservoir data"):
            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D']
            S = param['S']

            M1 = param['M1']
            M2 =  param['M2']
            omega1 = param['omega1']
            omega2 = param['omega2']
            r_fD = param['r_fD']

            model = InfiniteRadialCompositeReservoirModel(C_D=C_D, S=S, M1=M1, M2=M2, omega1=omega1, omega2=omega2, r_fD=r_fD)
            alg = ShtefestAlgorithm(N=16)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)
