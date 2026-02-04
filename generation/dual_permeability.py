from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.dual_permeability_coverter import DualPermeabilityDimensionConverter
from generation.generator import ParamGenerator, DataGenerator, GenerationParams
from generation.utils import extract_scalar
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.dualpermeability.finite_dual_permeability_model import FiniteDualPermeabilityReservoirModel
from model.dualpermeability.infinite_dual_permeability_model import InfiniteDualPermeabilityReservoirModel


class DualPermeabilityParamGenerator(ParamGenerator):
    """
    Генерация параметров для модели двойной проницаемости
    """

    def generate(self) -> List[pd.DataFrame]:
        base_params = {
            # Система 1 (обычно трещины/высокопроницаемая)
            'h1': 10,  # толщина пласта, м
            'k1': 5e-13,  # проницаемость системы 1, м²
            'phi1': 0.05,  # пористость системы 1 (1-15%)
            'c_t1': 1e-9,  # сжимаемость системы 1, 1/Па

            # Система 2 (обычно матрица/низкопроницаемая)
            'h2': 20,  # толщина пласта, м
            'k2': 1e-14,  # проницаемость системы 2, м²
            'phi2': 0.15,  # пористость системы 2 (5-35%)
            'c_t2': 1.5e-9,  # сжимаемость системы 2, 1/Па

            # Общие параметры
            'q1_frac': 0.8,  # доля дебита из системы 1
            'mu': 1e-3,  # вязкость, Па·с
            'B': 1.2,  # объемный коэффициент
            'p_i': 25e6,  # начальное давление, Па
            'r_w': 0.1  # радиус скважины, м
        }

        correlation_groups = [
            ['k2', 'phi2'],  # параметры системы 2 коррелируют
            ['mu', 'B'],  # вязкость и объемный коэффициент
            ['p_i', 'c_t1', 'c_t2']  # давление и сжимаемости
        ]

        varied_params_list = []

        for i in range(self.size):
            params = base_params.copy()

            # Вариация геометрических параметров
            params['h1'] *= np.random.uniform(0.5, 1.0)  # толщина
            params['h2'] *= np.random.uniform(0.5, 2.0)  # толщина
            params['r_e'] = np.random.uniform(50, 1000)  # внешний радиус

            # Распределение дебита
            params['q_total'] = np.random.uniform(5, 50)  # дебит
            params['q1_frac'] = np.random.uniform(0.1, 0.9)  # доля из системы 1
            params['q2_frac'] = 1.0 - params['q1_frac']  # доля из системы 2

            # Абсолютные дебиты
            params['q1'] = params['q_total'] * params['q1_frac']
            params['q2'] = params['q_total'] * params['q2_frac']

            # Коэффициент влияния ствола
            params['C'] = 10 ** np.random.uniform(-9, -7)  # м³/Па

            # Фактор формы
            params['alpha'] = np.random.uniform(0.001, 1)

            # Скин-факторы
            params['S1'] = np.random.uniform(0, 10)
            params['S2'] = np.random.uniform(0, 20)

            for group in correlation_groups:
                group_factor = np.random.uniform(0.5, 2.0)
                for param_name in group:
                    if param_name in params:
                        individual_factor = np.random.uniform(0.9, 1.1)
                        params[param_name] *= group_factor * individual_factor

            # Система 1
            params['phi1'] = np.clip(params['phi1'], 0.01, 0.15)  # 1-15%
            params['k1'] = np.clip(params['k1'], 1e-14, 1e-11)  # 10 мД - 10 Д
            params['c_t1'] = np.clip(params['c_t1'], 0.5e-10, 5e-9)  # сжимаемость

            # Система 2
            params['phi2'] = np.clip(params['phi2'], 0.05, 0.35)  # 5-35%
            params['k2'] = np.clip(params['k2'], 1e-16, 1e-13)  # 0.1 мД - 100 мД
            params['c_t2'] = np.clip(params['c_t2'], 0.5e-9, 5e-9)  # сжимаемость

            # Флюидные свойства
            params['mu'] = np.clip(params['mu'], 0.5e-3, 50e-3)  # 0.5-50 мПа·с
            params['B'] = np.clip(params['B'], 1.0, 1.8)  # объемный коэффициент
            
            # Корректировка параметров для получения характерных графиков двойной проницаемости
            kappa = np.random.uniform(0.7, 0.99)
            params['k1'] = params['k2'] * kappa * params['h2'] / (params['h1'] * (1 - kappa))

            converter = DualPermeabilityDimensionConverter(
                k1=params['k1'],
                k2=params['k2'],
                phi1=params['phi1'],
                phi2=params['phi2'],
                c_t1=params['c_t1'],
                c_t2=params['c_t2'],
                h1=params['h1'],
                h2=params['h2'],
                q1=params['q1'],
                q2=params['q2'],
                mu=params['mu'],
                B=params['B'],
                p_i=params['p_i'],
                r_w=params['r_w'],
                alpha=params['alpha'],
                S1=params['S1'],
                S2=params['S2']
            )

            # 4. Безразмерный внешний радиус
            params['R_eD'] = converter.radius_from_dim_to_dimless(params['r_e'])

            params['C_D'] = converter.wellbore_storage_from_dim_to_dimless(params['C'])
            params['omega'] = converter.calc_omega()
            params['lambda'] = converter.calc_lambda()

            # Отношения проницаемостей
            params['kappa'] = converter.kappa

            result_df = pd.DataFrame([params])
            varied_params_list.append(result_df)

        return varied_params_list


class InfiniteDualPermeabilityModelGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта модели двойной проницаемости
    """
    def __init__(self, params: GenerationParams):
        super().__init__('dual_permeability_inf', params)
        self.param_gen = DualPermeabilityParamGenerator(self.size)

    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual permeability infinite reservoir data"):
            param.drop('R_eD', axis=1, inplace=True) # убираем информацию о радиусе границы

            converter = DualPermeabilityDimensionConverter(
                k1=extract_scalar(param['k1']),
                k2=extract_scalar(param['k2']),
                phi1=extract_scalar(param['phi1']),
                phi2=extract_scalar(param['phi2']),
                c_t1=extract_scalar(param['c_t1']),
                c_t2=extract_scalar(param['c_t2']),
                h1=extract_scalar(param['h1']),
                h2=extract_scalar(param['h2']),
                q1=extract_scalar(param['q1']),
                q2=extract_scalar(param['q2']),
                mu=extract_scalar(param['mu']),
                B=extract_scalar(param['B']),
                p_i=extract_scalar(param['p_i']),
                r_w=extract_scalar(param['r_w']),
                alpha=extract_scalar(param['alpha']),
                S1=extract_scalar(param['S1']),
                S2=extract_scalar(param['S2'])
            )

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            S = param['S1'][0]
            omega = param['omega'][0]
            lam = param['lambda'][0]
            kappa = param['kappa'][0]

            model = InfiniteDualPermeabilityReservoirModel(C_D=C_D, S=S, omega=omega, lam=lam, kappa=kappa)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=self.sigma) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)


class FiniteDualPermeabilityGenerator(DataGenerator):
    """
    Генерация данных для модели двойной проницаемости для круговой границы
    """
    def __init__(self, params: GenerationParams):
        super().__init__('dual_permeability_fin', params)
        self.param_gen = DualPermeabilityParamGenerator(self.size)

    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual permeability finite reservoir data"):
            converter = DualPermeabilityDimensionConverter(
                k1=extract_scalar(param['k1']),
                k2=extract_scalar(param['k2']),
                phi1=extract_scalar(param['phi1']),
                phi2=extract_scalar(param['phi2']),
                c_t1=extract_scalar(param['c_t1']),
                c_t2=extract_scalar(param['c_t2']),
                h1=extract_scalar(param['h1']),
                h2=extract_scalar(param['h2']),
                q1=extract_scalar(param['q1']),
                q2=extract_scalar(param['q2']),
                mu=extract_scalar(param['mu']),
                B=extract_scalar(param['B']),
                p_i=extract_scalar(param['p_i']),
                r_w=extract_scalar(param['r_w']),
                alpha=extract_scalar(param['alpha']),
                S1=extract_scalar(param['S1']),
                S2=extract_scalar(param['S2'])
            )

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            S = param['S1'][0]
            r_D_e = param['R_eD'][0]
            omega = param['omega'][0]
            lam = param['lambda'][0]
            kappa = param['kappa'][0]

            model = FiniteDualPermeabilityReservoirModel(C_D=C_D, S=S, omega=omega, lam=lam, kappa=kappa, R_D_E=r_D_e)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma=self.sigma) \
                .derivative(smoothig_alg='regression', delta=0.3) \
                .get_pressure()

            self.save(curve, param)