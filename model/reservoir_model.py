import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from abc import ABC, abstractmethod

from derivate.bourdais_logarithmic_derivative import BourdaisLogarithmicDerivative
from inversion.laplas_inversion_method import LaplasInversionMethod


class ReservoirModel(ABC):
    """
    Абстрактный класс для модели пласта.

    Атрибут:
        df: DataFrame: хранит таблицу зависимости давления от времени
        Столбец t_D: безразмерное время
        Столбец P_wD : идеальная кривая давления без изменений
        Столбец dP_wD : градиент кривой давления
        Столбец P_wD_gauss : гауссов шум
    """

    def __init__(self):
        self.df = pd.DataFrame()

    @staticmethod
    def agarwal_filter(p_w_D, s, r_D, S, C_D):
        """
        Решение для PTA в бесконечно действующих пластах с учетом CWBS представлено в пространстве Лапласа Агарвалом

        Принимает:
            p_w_D : функция зависимости давления от времени для пласта
            s     : переменная Лапласа
            r_D   : безразмерный радиус
            S     : скин-фактор
            C_D   : безразмерное влияние ствола скважины

        Возвращает:
            давление
        """

        denominator = s * (s * C_D + 1 / (s * p_w_D(s, r_D) + S))

        return 1 / denominator

    @abstractmethod
    def F(self, s):
        """
        Решение уравнения диффузии в пространстве Лапласа для модели скважины.

        Замечание: решение уравнения диффузии зависит также от параметров пласта,
        но для численного обращения нужна зависимость только от переменной Лапласа,
        то есть необходимо зафиксировать остальные параметры.

        Получает:
            s: переменная Лапласа.

        Возвращает:
            Значение функции в конкретной точке.
        """
        pass

    def pressure(self, t_D, alg: LaplasInversionMethod):
        """
        Функция вычисления массива давления, по массиву безразмерного времени.
        С автоматическим восстановлением NaN/Inf значений через интерполяцию.с
        (в среднем nan значений в выборках немного, обнаруживаются в radial_composite для ранних периодов времени)
        """
        if np.any(t_D <= 0):
            raise ValueError("Time values must be positive")

        p_w_d_results = []
        nan_count = 0
        inf_count = 0

        for idx, t_D_val in enumerate(t_D):
            try:
                pwd_val = alg.inverse(self.F, t_D_val)

                if np.isnan(pwd_val):
                    nan_count += 1
                    pwd_val = np.nan
                elif np.isinf(pwd_val):
                    inf_count += 1
                    pwd_val = np.nan

                p_w_d_results.append(pwd_val)
            except Exception as e:
                print(f"Ошибка при t_D={t_D_val:.2e}: {e}")
                nan_count += 1
                p_w_d_results.append(np.nan)

        p_w_d_array = np.array(p_w_d_results)

        table_data = np.column_stack((t_D, p_w_d_array))
        self.df = pd.DataFrame(table_data, columns=['t_D', 'P_wD'])

        return self

    def gauss_noize(self, mu=0, sigma=0.1, relative=False, sigma_mpa=None, converter=None):
        """
        Добавление гауссова шума к кривой ГДИС.

        sigma      — стандартное отклонение в безразмерных единицах.
        relative   — если True, sigma задаётся как доля от P_wD.
        sigma_mpa  — стандартное отклонение в МПа; требует передачи converter.
                     Имеет приоритет над sigma.
        converter  — объект конвертера (HomogeneousConverter и т.п.) с методом
                     pressure_from_dim_to_dimless и атрибутом p_i.
        """
        if sigma_mpa is not None:
            if converter is None:
                raise ValueError("converter обязателен при задании sigma_mpa")
            # σ_Pa → σ_D: используем тот же масштабный множитель, что и для давления
            sigma = converter.pressure_from_dim_to_dimless(converter.p_i - sigma_mpa * 1e6)

        if relative:
            noise = np.random.normal(mu, sigma, len(self.df)) * np.abs(self.df['P_wD'].values)
        else:
            noise = np.random.normal(mu, sigma, len(self.df))
        
        self.df['P_wD_gauss'] = self.df['P_wD'] + noise

        self.df['P_wD_gauss'] = self.df['P_wD_gauss'].clip(lower=0)
        return self

    def derivative(self, smoothig_alg='regression', delta=1):
        """
        Добавление в датафрейм столбца с логарифмической производной
        """
        
        self.df['ln_t_D'] = np.log(self.df['t_D'])
        
        div = BourdaisLogarithmicDerivative(self.df)

        y_col = 'P_wD'
        if 'P_wD_gauss' in self.df.columns:
            y_col = 'P_wD_gauss'

        if smoothig_alg == 'regression':
            self.df['dP_wD'] = div.derivate_with_regression_smoothing(self.df['t_D'], delta, y_col_name=y_col)
        elif smoothig_alg == 'rolling_window':
            self.df['dP_wD'] = div.derivative_with_rolling_window_smoothing(self.df['t_D'], delta, y_col_name=y_col)
        else:
            print('Error smoothing algo can be only: regression or rolling_window')

        return self


    def visualize(self, title, horizontal_line=False, point_type='x'):
        """
        Визуализация, зависимости давления от времени в loglog графике.
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(self.df['t_D'], self.df['P_wD'], 'g' + point_type, label='Теоретическое давление')
        plt.loglog(self.df['t_D'], self.df['dP_wD'], 'b' + point_type, linewidth=2, label='dP_wD/dln_t_D')

        if 'P_wD_gauss' in self.df.columns:
            plt.loglog(self.df['t_D'], self.df['P_wD_gauss'], 'ro', markersize=2,
                       label='Имитация реальных замеров (Гауссов шум)', alpha=0.6)

        if horizontal_line:
            x_limits = plt.xlim()
            plt.hlines(y=0.5, xmin=x_limits[0], xmax=x_limits[1], colors='red', linestyles=':',
                       linewidth=2, label='y = 0.5')

        plt.xlabel('Безразмерное время, t_D')
        plt.ylabel('Безразмерное давление, P_wD')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()


    def write_in_file(self, filename):
        """
        Запись в данных в файл

        Получает:
            filename: название файла

        Возвращает:
            self
        """
        self.df.to_csv(filename + '.csv', index=False)
        return self

    def get_pressure(self):
        """
        Функция для получения датафрейма с давлением.

        Возвращает:
            Датафрейм зависимости давления/времени.
        """
        return self.df

    def load_model(self, filename: str):
        """
        Загрузка модели из файла
        """
        self.df = pd.read_csv(filename)
        return self