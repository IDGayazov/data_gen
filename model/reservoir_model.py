import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        Получает:
            F: функция решения уравнения диффузии в пространстве Лапласа.
            t_D: массив безразмерного времени.
            alg: численный метод обращения (LaplasInversionMethod).
            filename: файл для записи данных.

        Возвращает:
            self
        """
        # Проверка входных данных
        if np.any(np.isnan(t_D)) or np.any(np.isinf(t_D)):
            raise ValueError("Input time array contains NaN or Inf values")
        
        if np.any(t_D <= 0):
            raise ValueError("Time values must be positive")
        
        p_w_d_results = []
        for t_D_val in t_D:
            try:
                pwd_val = alg.inverse(self.F, t_D_val)
                # Проверка результата
                if np.isnan(pwd_val) or np.isinf(pwd_val):
                    raise ValueError(f"Pressure calculation returned NaN/Inf for t_D={t_D_val}")
                p_w_d_results.append(pwd_val)
            except Exception as e:
                raise ValueError(f"Error calculating pressure for t_D={t_D_val}: {e}")

        table_data = np.column_stack((t_D, p_w_d_results))
        self.df = pd.DataFrame(table_data, columns=['t_D', 'P_wD'])
        
        # Финальная проверка на NaN
        if self.df['P_wD'].isna().any() or self.df['t_D'].isna().any():
            raise ValueError("Resulting DataFrame contains NaN values")
        
        return self

    def gauss_noize(self, mu=0, sigma=5e-5):
        """
        Функция, добавляющая гауссов шум к данным.
        Данные с шумами добавляются в отдельный столбец P_wD_gauss в датафрейме.

        Получает:
            mu: матожидание.
            sigma: стандартное отклонение.

        Возвращает:
            self
        """
        # Проверка на NaN в исходных данных
        if self.df['P_wD'].isna().any():
            raise ValueError("Pressure data contains NaN values before adding noise")
        
        signal_amplitude = self.df['P_wD'].max() - self.df['P_wD'].min()
        
        # Проверка на валидность амплитуды
        if signal_amplitude <= 0 or np.isnan(signal_amplitude) or np.isinf(signal_amplitude):
            raise ValueError(f"Invalid signal amplitude: {signal_amplitude}")
        
        noise = np.random.normal(mu, sigma * signal_amplitude, len(self.df))
        self.df['P_wD_gauss'] = self.df['P_wD'] + noise
        
        # Проверка результата
        if self.df['P_wD_gauss'].isna().any():
            raise ValueError("Pressure data with noise contains NaN values")
        
        return self

    def derivative(self, smoothig_alg='regression', delta=1):
        """
        Добавление в датафрейм столбца с логарифмической производной
        """
        # Проверка на NaN в исходных данных
        if self.df['t_D'].isna().any() or self.df['P_wD'].isna().any():
            raise ValueError("Input data contains NaN values before derivative calculation")
        
        # Проверка на валидность времени
        if (self.df['t_D'] <= 0).any():
            raise ValueError("Time values must be positive")
        
        self.df['ln_t_D'] = np.log(self.df['t_D']) # необходимо для последующего сглаживания
        
        # Проверка на NaN после вычисления логарифма
        if self.df['ln_t_D'].isna().any():
            raise ValueError("NaN values after computing logarithm of time")
        
        div = BourdaisLogarithmicDerivative(self.df)

        y_col = 'P_wD'
        if 'P_wD_gauss' in self.df.columns:
            y_col = 'P_wD_gauss'
        
        # Проверка на NaN в данных давления
        if self.df[y_col].isna().any():
            raise ValueError(f"Pressure data ({y_col}) contains NaN values")

        if smoothig_alg == 'regression':
            self.df['dP_wD'] = div.derivate_with_regression_smoothing(self.df['t_D'], delta, y_col_name=y_col)
        elif smoothig_alg == 'rolling_window':
            self.df['dP_wD'] = div.derivative_with_rolling_window_smoothing(self.df['t_D'], delta, y_col_name=y_col)
        else:
            print('Error smoothing algo can be only: regression or rolling_window')

        # Проверка на NaN в результате производной
        nan_count = self.df['dP_wD'].isna().sum()
        if nan_count > 0:
            # Заменяем NaN на предыдущее значение или интерполируем
            self.df['dP_wD'] = self.df['dP_wD'].fillna(method='ffill').fillna(method='bfill')
            # Если все еще есть NaN, заменяем на 0
            self.df['dP_wD'] = self.df['dP_wD'].fillna(0.0)
            print(f"Warning: {nan_count} NaN values in derivative were replaced")

        return self


    def visualize(self, title, horizontal_line=True):
        """
        Визуализация, зависимости давления от времени в loglog графике.
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(self.df['t_D'], self.df['P_wD'], 'g-', label='Теоретическое давление')
        plt.loglog(self.df['t_D'], self.df['dP_wD'], 'b--', linewidth=2, label='dP_wD/dln_t_D')

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
