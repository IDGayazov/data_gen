from typing import List, Dict

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from conversion.dual_porosity_converter import DualPorosityDimensionConverter
from generation.generator import ParamGenerator, DataGenerator, GenerationParams
from generation.utils import extract_scalar
from generation.split import make_param_train_val_lists
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.dualporosity.finite_dual_porosity_model import FiniteDualPorosityReservoirModel
from model.dualporosity.infinite_dual_porosity_model import InfiniteDualPorosityReservoirModel


class DualPorosityModelParamGenerator(ParamGenerator):
    """
    Генерация параметров для модели двойной пористости
    """

    def __init__(self, size, type='train', t_max_days=30):
        super().__init__(size, type)
        self.t_max_days = t_max_days

    def generate(self) -> List[pd.DataFrame]:

        # Разделение на подинтервалы для параметра lambda
        list1, list2 = make_param_train_val_lists(1e-8, 1e-5, 16)

        train_ranges = {
            # Размерные параметры (ключевые)
            'k_f': [(0.01, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9)],  # проницаемость трещин, 1e-12*м^2
            'k_m': [(1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)],  # проницаемость матрицы, 1e-12*м^2
            
            # Безразмерные параметры (ключевые)
            'C_D': [(200, 300), (400, 500), (600, 700), (800, 900)],  # безразмерный коэфф. ствола
            'omega': [(0.01, 0.05), (0.1, 0.15), (0.2, 0.24), (0.27, 0.3)],  # коэффициент ёмкости
            'lambda': list1,  # коэффициент перетока
            
            # Независимые параметры (остаются как есть)
            'h': [(5.0, 30.0)],  # толщина, м
            'mu': [(0.5e-3, 10e-3)],  # вязкость, Па*с
            'B': [(1.0, 1.5)],  # объемный коэфф.
            'p_i': [(15e6, 40e6)],  # нач. давление, Па
            'q': [(10, 100)],  # дебит, м3/сут
            'S': [(0.1, 1), (2, 3), (4, 5), (6, 7), (8, 9)],  # скин-фактор
            'r_e': [(200, 300), (400, 500), (600, 700), (800, 900)]
        }

        val_ranges = {
            'k_f': [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1)],
            'k_m': [(5e-5, 5e-4), (5e-4, 1e-2)],
            'C_D': [(300, 400), (500, 600), (700, 800), (900, 1000)],
            'omega': [(0.05, 0.1), (0.15, 0.2), (0.24, 0.27)],
            'lambda': list2, # коэффициент перетока
            'h': [(5.0, 30.0)],
            'mu': [(0.5e-3, 10e-3)],
            'B': [(1.0, 1.5)],
            'p_i': [(15e6, 40e6)],
            'q': [(10, 100)],
            'S': [(1, 2), (3, 4), (5, 6), (7, 8)],
            'r_e': [(300, 400), (500, 600), (700, 800), (900, 1000)]
        }

        test_ranges = {
            'k_f': [(0.1, 1)],
            'k_m': [(1e-5, 1e-2)],
            'C_D': [(200, 1000)],
            'omega': [(0.01, 0.3)],
            'lambda': [(1e-8, 1e-5)],
            'h': [(5.0, 30.0)],
            'mu': [(0.5e-3, 10e-3)],
            'B': [(1.0, 1.5)],
            'p_i': [(15e6, 40e6)],
            'q': [(10, 100)],
            'S': [(1, 8)],
            'r_e': [(200, 1000)]
        }

        # Выбираем диапазоны
        if self.type == 'val':
            ranges = val_ranges
        elif self.type == 'test':
            ranges = test_ranges
        else:
            ranges = train_ranges

        varied_params_list = []

        for _ in range(self.size):
            params = {}
            
            # 1. Генерация КЛЮЧЕВЫХ параметров из интервалов
            for param_name, param_intervals in ranges.items():
                low, high = random.choice(param_intervals)
                
                # Логарифмические параметры
                if param_name == 'k_f' or param_name == 'k_m':
                    params[param_name] = np.random.uniform(low, high) * 1e-12
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            # 2. Фиксируем r_w и пересчитываем остальные параметры
            params['r_w'] = 0.1
            
            # 2.1. Пересчитываем phi_f и phi_m через omega без клиппинга
            # omega = phi_f / (phi_f + phi_m)  (при c_tf = c_tm)
            c_tf_typical = 1e-9
            c_tm_typical = 1e-9

            # Выбираем phi_f так, чтобы phi_m = phi_f*(1/omega - 1) не превышало 0.35
            omega = params['omega']
            phi_f_max_no_clip = 0.35 * omega / (1.0 - omega)  # граница из phi_m ≤ 0.35
            phi_f_max = min(0.05, phi_f_max_no_clip)
            phi_f_min = 0.001
            params['phi_f'] = np.random.uniform(phi_f_min, max(phi_f_min, phi_f_max))
            params['phi_m'] = params['phi_f'] * (1.0 / omega - 1.0)  # клиппинг не нужен

            params['kappa'] = params['k_m'] / params['k_f']

            # 2.25. Гарантируем что переход двойной пористости виден в окне t_max_days
            # Переход виден если t_D_transition ~ 1/lambda < t_D_max
            t_max_seconds = self.t_max_days * 24 * 3600
            total_storage_val = params['phi_f'] * c_tf_typical + params['phi_m'] * c_tm_typical
            lambda_min = 5 * total_storage_val * params['mu'] * params['r_w']**2 / (params['k_f'] * t_max_seconds)
            params['lambda'] = max(params['lambda'], lambda_min)

            # 2.3. Через lambda пересчитываем alpha (без клиппинга — alpha определяется физикой)
            params['alpha'] = params['lambda'] / (params['kappa'] * params['r_w']**2)
            
            # 2.4. Через C_D пересчитываем реальный C (размерный коэффициент ствола)
            # C_D = C / (2 * pi * h * phi_f * c_tf * r_w^2)  (упрощённо)
            # Отсюда вычисляем C (если нужен)
            denominator = 2 * np.pi * params['h'] * params['phi_f'] * c_tf_typical * params['r_w']**2
            params['C'] = params['C_D'] * denominator
            
            # 2.5. Через C_D также можно скорректировать c_tf, если нужно согласование
            # c_tf = C_D / (C * 2 * pi * h * phi_f * r_w^2) - но C у нас неизвестен, оставляем как есть
            
            # 3. Задаём оставшиеся параметры (независимые)
            params['c_tf'] = c_tf_typical
            params['c_tm'] = c_tm_typical
            params['R_eD'] = params['r_e'] / params['r_w']  # безразмерный радиус
            
            # 5. Проверка согласованности через конвертер (опционально)
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
            
            # 6. Сохраняем результат
            params['data_type'] = self.type
            varied_params_list.append(pd.DataFrame([params]))
        
        return varied_params_list


class InfiniteDualPorosityGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта модели двойной пористости
    """
    def __init__(self, params: GenerationParams):
        super().__init__('dual_porosity_inf', params)
        self.param_gen = DualPorosityModelParamGenerator(self.size, self.type, self.t_max_days)


    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual porosity infinite reservoir data"):
            param.drop('R_eD', axis=1, inplace=True) # убираем информацию о радиусе границы

            converter = DualPorosityDimensionConverter(
                k_f=extract_scalar(param['k_f']),
                k_m=extract_scalar(param['k_m']),
                h=extract_scalar(param['h']),
                q=extract_scalar(param['q']),
                mu=extract_scalar(param['mu']),
                B=extract_scalar(param['B']),
                p_i=extract_scalar(param['p_i']),
                phi_f=extract_scalar(param['phi_f']),
                phi_m=extract_scalar(param['phi_m']),
                c_tf=extract_scalar(param['c_tf']),
                c_tm=extract_scalar(param['c_tm']),
                r_w=extract_scalar(param['r_w']),
                alpha=extract_scalar(param['alpha'])
            )

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            S = param['S'][0]
            omega = param['omega'][0]
            lam = param['lambda'][0]

            model = InfiniteDualPorosityReservoirModel(C_D=C_D, S=S, omega=omega, lam=lam)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma_mpa=self.sigma, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)


class FiniteDualPorosityGenerator(DataGenerator):
    """
    Генерация данных для модели двойной пористости с круговой границей
    """
    def __init__(self, params: GenerationParams):
        super().__init__('dual_porosity_fin', params)
        self.param_gen = DualPorosityModelParamGenerator(self.size, self.type, self.t_max_days)

    def generate(self):
        params = self.param_gen.generate()

        params_list = list(params)

        for param in tqdm(params_list, desc="Generating dual porosity finite reservoir data"):
            converter = DualPorosityDimensionConverter(
                k_f=extract_scalar(param['k_f']),
                k_m=extract_scalar(param['k_m']),
                h=extract_scalar(param['h']),
                q=extract_scalar(param['q']),
                mu=extract_scalar(param['mu']),
                B=extract_scalar(param['B']),
                p_i=extract_scalar(param['p_i']),
                phi_f=extract_scalar(param['phi_f']),
                phi_m=extract_scalar(param['phi_m']),
                c_tf=extract_scalar(param['c_tf']),
                c_tm=extract_scalar(param['c_tm']),
                r_w=extract_scalar(param['r_w']),
                alpha=extract_scalar(param['alpha'])
            )

            t_D_array = self.generate_search_time(converter)

            C_D = param['C_D'][0]
            r_D_e = param['R_eD'][0]
            omega = param['omega'][0]
            lam = param['lambda'][0]
            S = param['S'][0]

            model = FiniteDualPorosityReservoirModel(C_D=C_D, S=S, omega=omega, lam=lam, R_D_e=r_D_e)
            alg = ShtefestAlgorithm(N=12)

            curve = model.pressure(t_D_array, alg) \
                .gauss_noize(mu=0, sigma_mpa=self.sigma, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)