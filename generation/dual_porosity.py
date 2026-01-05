from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.dual_porosity_converter import DualPorosityDimensionConverter
from generation.generator import ParamGenerator, DataGenerator
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.dualporosity.finite_dual_porosity_model import FiniteDualPorosityReservoirModel
from model.dualporosity.infinite_dual_porosity_model import InfiniteDualPorosityReservoirModel


class DualPorosityModelParamGenerator(ParamGenerator):
    """
    Генерация параметров для модели двойной пористости
    с использованием DualPorosityDimensionConverter
    """

    def generate(self) -> List[pd.DataFrame]:
        """
        Генерация реалистичных параметров для модели двойной пористости.
        Возвращает как размерные параметры, так и безразмерные.
        """
        base_params = {
            # Параметры трещин (fractures) - высокая проницаемость
            'k_f': 5e-13,  # проницаемость трещин, м²
            'phi_f': 0.02,  # пористость трещин (обычно 0.001-0.05)
            'c_tf': 1e-9,  # сжимаемость трещин, 1/Па

            # Параметры матрицы (matrix) - высокая пористость
            'k_m': 5e-15,  # проницаемость матрицы, м²
            'phi_m': 0.18,  # пористость матрицы (обычно 0.05-0.25)
            'c_tm': 1.5e-9,  # сжимаемость матрицы, 1/Па

            # Общие параметры
            'h': 10,  # толщина пласта, м
            'q': 2.31e-3,  # дебит, м³/с
            'mu': 1e-3,  # вязкость, Па·с
            'B': 1.2,  # объемный коэффициент
            'p_i': 25e6,  # начальное давление, Па
            'r_w': 0.1,  # радиус скважины, м

            # Геометрический фактор
            'alpha': 12.0,  # shape factor, 1/м²

            # Размерный коэффициент влияния ствола
            'C': 1e-8,  # м³/Па
        }

        correlation_groups = [
            ['k_f', 'phi_f'],  # параметры трещин коррелируют
            ['k_m', 'phi_m'],  # параметры матрицы коррелируют
            ['mu', 'B'],  # вязкость и объемный коэффициент
            ['p_i', 'c_tf', 'c_tm'],  # давление и сжимаемости
        ]

        varied_params_list = []

        for i in range(self.size):
            params = base_params.copy()

            # Вариация геометрических параметров
            params['h'] *= np.random.uniform(0.5, 2.0)  # толщина
            params['q'] *= np.random.uniform(0.2, 3.0)  # дебит
            params['r_w'] *= np.random.uniform(0.8, 1.2)  # радиус скважины
            params['r_e'] = np.random.uniform(50, 1000)  # внешний радиус

            # Вариация коэффициента влияния ствола
            params['C'] = 10 ** np.random.uniform(-9, -7)  # м³/Па

            # Геометрический фактор (случайный выбор)
            params['alpha'] = np.random.choice([4, 12, 32])  # 4: сферы, 12: слэбы, 32: кубы

            # Скин-фактор
            rand = np.random.random()
            if rand < 0.1:  # 10% - маленький скин
                params['S'] = np.random.uniform(0, 0.1)
            elif rand < 0.3:  # 20% - нулевой или близкий к нулю
                params['S'] = np.random.uniform(0.1, 0.5)
            elif rand < 0.8:  # 50% - положительный, но небольшой
                params['S'] = np.random.uniform(0.5, 10)
            else:  # 20% - высокий положительный скин
                params['S'] = np.random.uniform(10, 50)

            for group in correlation_groups:
                group_factor = np.random.uniform(0.5, 2.0)
                for param_name in group:
                    if param_name in params:
                        individual_factor = np.random.uniform(0.9, 1.1)
                        params[param_name] *= group_factor * individual_factor

            # Трещины
            params['phi_f'] = np.clip(params['phi_f'], 0.001, 0.05)  # 0.1-5%
            params['k_f'] = np.clip(params['k_f'], 1e-14, 1e-11)  # 10 мД - 10 Д
            params['c_tf'] = np.clip(params['c_tf'], 0.5e-10, 5e-9)  # сжимаемость трещин

            # Матрица
            params['phi_m'] = np.clip(params['phi_m'], 0.05, 0.35)  # 5-35%
            params['k_m'] = np.clip(params['k_m'], 1e-17, 1e-13)  # 0.01 мД - 100 мД
            params['c_tm'] = np.clip(params['c_tm'], 0.5e-9, 5e-9)  # сжимаемость матрицы

            # Флюидные свойства
            params['mu'] = np.clip(params['mu'], 0.5e-3, 50e-3)  # 0.5-50 мПа·с
            params['B'] = np.clip(params['B'], 1.0, 1.8)  # объемный коэффициент

            # Плотность скин-фактора
            params['S'] = np.clip(params['S'], 0, 100)

            converter = DualPorosityDimensionConverter(
                k_f=params['k_f'],
                k_m=params['k_m'],
                h=params['h'],
                q=params['q'],
                mu=params['mu'],
                B=params['B'],
                p_i=params['p_i'],
                phi_f=params['phi_f'],
                phi_m=params['phi_m'],
                c_tf=params['c_tf'],
                c_tm=params['c_tm'],
                r_w=params['r_w'],
                alpha=params['alpha']
            )

            # 1. C_D (безразмерный коэффициент влияния ствола)
            params['C_D'] = converter.wellbore_storage_from_dim_to_dimless(params['C'])

            # 2. ω (коэффициент ёмкости)
            params['omega'] = converter.omega

            # 3. λ (коэффициент перетока)
            params['lambda'] = converter.lambda_param

            # 4. κ (отношение проницаемостей)
            params['kappa'] = converter.kappa

            # 5. Отношение пьезопроводностей
            params['eta_ratio'] = converter.diffusivity_ratio()

            # 6. Безразмерный внешний радиус
            params['R_eD'] = converter.radius_from_dim_to_dimless(params['r_e'])

            result_df = pd.DataFrame([params])
            varied_params_list.append(result_df)

        return varied_params_list, converter


class InfiniteDualPorosityGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта модели двойной пористости
    """

    def __init__(self, t_max_days, points_count, size):
        self.reservoir_type = 'dual_porosity_inf'
        self.t_max_days = t_max_days
        self.size = size
        self.points_count = points_count
        self.param_gen = DualPorosityModelParamGenerator(self.size)


    def generate(self):
        params, converter = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual porosity infinite reservoir data"):
            param.drop('R_eD', axis=1, inplace=True) # убираем информацию о радиусе границы

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            S = param['S'][0]
            omega = param['omega'][0]
            lam = param['lambda'][0]

            model = InfiniteDualPorosityReservoirModel(C_D=C_D, S=S, omega=omega, lam=lam)
            alg = ShtefestAlgorithm(N=16)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)


class FiniteDualPorosityGenerator(DataGenerator):
    """
    Генерация данных для модели двойной пористости с круговой границей
    """

    def __init__(self, t_max_days, points_count, size):
        self.reservoir_type = 'dual_porosity_fin'
        self.t_max_days = t_max_days
        self.size = size
        self.points_count = points_count
        self.param_gen = DualPorosityModelParamGenerator(self.size)


    def generate(self):
        params, converter = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual porosity finite reservoir data"):
            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            r_D_e = param['R_eD'][0]
            omega = param['omega'][0]
            lam = param['lambda'][0]
            S = param['S'][0]

            model = FiniteDualPorosityReservoirModel(C_D=C_D, S=S, omega=omega, lam=lam, R_D_e=r_D_e)
            alg = ShtefestAlgorithm(N=16)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=5e-7) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)