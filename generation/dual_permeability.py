from typing import List

import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from conversion.dual_permeability_coverter import DualPermeabilityDimensionConverter
from generation.generator import ParamGenerator, DataGenerator, GenerationParams
from generation.utils import extract_scalar
from inversion.shtefest_algorithm import ShtefestAlgorithm
from generation.split import make_param_train_val_lists
from model.dualpermeability.finite_dual_permeability_model import FiniteDualPermeabilityReservoirModel
from model.dualpermeability.infinite_dual_permeability_model import InfiniteDualPermeabilityReservoirModel


class DualPermeabilityParamGenerator(ParamGenerator):
    """
    Генерация параметров для модели двойной проницаемости
    """

    def __init__(self, size, type='train', t_max_days=30):
        super().__init__(size, type)
        self.t_max_days = t_max_days

    def generate(self) -> List[pd.DataFrame]:
        # Разделение на подинтервалы для параметров
        lambda_list1, lambda_list2 = make_param_train_val_lists(1e-7, 1e-5, 16)
        omega_list1, omega_list2 = make_param_train_val_lists(0.01, 0.5, 10, 'lin')
        kappa_list1, kappa_list2 = make_param_train_val_lists(0.3, 0.97, 10, 'lin')
        
        train_ranges = {
            # Система 1 (трещины/высокопроницаемая)
            'h1': [(5.0, 12.0)],  # толщина, м
            'k1': [(0.01, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9)],  # проницаемость, м²
            'c_t1': [(0.5e-10, 5e-9)],  # сжимаемость, 1/Па
            
            # Система 2 (матрица/низкопроницаемая)
            'h2': [(12.0, 15.0)],  # толщина, м
            'phi2': [(0.10, 0.15), (0.15, 0.25), (0.25, 0.32)],  # пористость 5-35%
            'c_t2': [(1e-9, 2e-9), (2e-9, 3e-9), (3e-9, 5e-9)],  # сжимаемость, 1/Па
            
            # Флюидные свойства
            'mu': [(0.5e-3, 2e-3), (2e-3, 10e-3), (10e-3, 30e-3)],  # вязкость, Па·с
            'B': [(1.0, 1.8)],  # объемный коэффициент
            
            # Режимные параметры
            'q_total': [(5, 15), (15, 30), (30, 50)],  # дебит, м³/сут
            'q1_frac': [(0.1, 0.3), (0.3, 0.6), (0.6, 0.9)],  # доля из системы 1
            
            # Безразмерные параметры (ключевые)
            'C_D': [(200, 300), (400, 500), (600, 700), (800, 900)],  # безразмерный коэфф. ствола
            'omega': omega_list1,  # коэффициент ёмкости
            'lambda': lambda_list1,  # коэффициент перетока
            'kappa': kappa_list1,

            # Геометрические и безразмерные
            'r_e': [(200, 300), (400, 500), (600, 700), (800, 900)],  # внешний радиус, м
            'S1': [(0.1, 1), (2, 3), (4, 5), (6, 7), (8, 9)],  # скин-фактор системы 1
            'S2': [(0, 3), (3, 6), (6, 10)],  # скин-фактор системы 2
        }
        
        val_ranges = {
            'h1': [(12.0, 15.0)], 
            'k1': [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1)],
            'c_t1': [(0.5e-10, 5e-9)],

            'h2': [(12.0, 15.0)],
            'phi2': [(0.15, 0.25)],
            'c_t2': [(2e-9, 3e-9)], 

            'C_D': [(300, 400), (500, 600), (700, 800), (900, 1000)],
            'omega': omega_list2,
            'lambda': lambda_list2,
            'kappa': kappa_list2,
            
            'mu': [(2e-3, 10e-3)],
            'B': [(1.2, 1.5)], 
            'q_total': [(15, 30)], 
            'q1_frac': [(0.3, 0.6)],
            'r_e': [(300, 400), (500, 600), (700, 800), (900, 1000)],
            'S1': [(1, 2), (3, 4), (5, 6), (7, 8)],
            'S2': [(3, 6)]
        }
        
        test_ranges = {
            'h1': [(5.0, 12.0)], 
            'k1': [(0.1, 1)],
            'c_t1': [(0.5e-9, 3e-9)],

            'h2': [(12.0, 20.0)],
            'phi2': [(0.15, 0.25)],
            'c_t2': [(1e-9, 5e-9)], 

            'mu': [(0.5e-3, 30e-3)],
            'B': [(1.0, 1.8)], 
            'q_total': [(5, 50)], 
            'q1_frac': [(0.1, 0.9)],
            'r_e': [(200, 800)],
            'S1': [(0, 8)],'S2': [(0, 10)],
            'C_D': [(200, 1000)],
            'lambda': [(1e-7, 1e-5)],
            'omega': [(0.01, 0.2)],
            'kappa': [(0.7, 0.99)]
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
            
            # 1. Генерация параметров из интервалов
            for param_name, param_intervals in ranges.items():
                low, high = random.choice(param_intervals)
                
                # Логарифмические параметры
                if param_name in ['k1']:
                    params[param_name] = np.random.uniform(low, high) * 1e-12
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            # 2. Фиксируем базовые параметры
            params['r_w'] = 0.1  # радиус скважины, м
            params['p_i'] = 25e6  # начальное давление, Па
            
            # 3. Пересчитываем дебиты
            params['q1'] = params['q_total'] * params['q1_frac']
            params['q2'] = params['q_total'] * (1.0 - params['q1_frac'])
            
            # 4. Корректировка k1 через k2 и kappa для характерных графиков двойной проницаемости
            kappa = params['kappa']
            params['k2'] = (params['k1'] * params['h1'] * (1.0 - kappa)) / (params['h2'] * kappa)

            denom_ratio = (1.0 / params['omega']) - 1.0
            params['phi1'] = (params['phi2'] * params['c_t2'] * params['h2'] * (1.0/denom_ratio)) / (params['c_t1'] * params['h1'])
            
            total_kh = params['k1'] * params['h1'] + params['k2'] * params['h2']
            total_storage = (params['phi1'] * params['c_t1'] * params['h1'] +
                             params['phi2'] * params['c_t2'] * params['h2'])

            # Гарантируем что переход двойной проницаемости виден в окне t_max_days
            t_max_seconds = self.t_max_days * 24 * 3600
            lambda_min = 5 * total_storage * params['mu'] * params['r_w']**2 / (total_kh * t_max_seconds)
            params['lambda'] = max(params['lambda'], lambda_min)

            params['alpha'] = (params['lambda'] * total_kh) / (params['r_w']**2 * params['k2'] * params['h2'])
            denominator = 2 * np.pi * total_storage * params['r_w']**2
            params['C'] = params['C_D'] * denominator

            # 6. Создание конвертера безразмерных величин
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
            
            # 7. Вычисление безразмерных параметров
            params['R_eD'] = converter.radius_from_dim_to_dimless(params['r_e'])

            # 8. Сохраняем результат
            params['data_type'] = self.type
            varied_params_list.append(pd.DataFrame([params]))
        
        return varied_params_list


class InfiniteDualPermeabilityModelGenerator(DataGenerator):
    """
    Генерация данных для бесконечного пласта модели двойной проницаемости
    """
    def __init__(self, params: GenerationParams):
        super().__init__('dual_permeability_inf', params)
        self.param_gen = DualPermeabilityParamGenerator(self.size, self.type, self.t_max_days)

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
                .gauss_noize(mu=0, sigma_mpa=0.00035, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)


class FiniteDualPermeabilityGenerator(DataGenerator):
    """
    Генерация данных для модели двойной проницаемости для круговой границы
    """
    def __init__(self, params: GenerationParams):
        super().__init__('dual_permeability_fin', params)
        self.param_gen = DualPermeabilityParamGenerator(self.size, self.type, self.t_max_days)

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
                .gauss_noize(mu=0, sigma_mpa=0.00035, converter=converter) \
                .derivative(smoothig_alg='regression', delta=0.175) \
                .get_pressure()

            self.save(curve, param)