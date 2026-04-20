import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
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

        # if np.any(np.isnan(p_w_d_array)):
        #     p_w_d_array = self._interpolate_nan_values(t_D, p_w_d_array)

        table_data = np.column_stack((t_D, p_w_d_array))
        self.df = pd.DataFrame(table_data, columns=['t_D', 'P_wD'])

        return self

    def _interpolate_nan_values(self, t, p):
        """
        Интерполяция NaN значений в массиве давления.
        Использует линейную интерполяцию в логарифмической шкале.
        """
        p_interp = p.copy()

        nan_mask = np.isnan(p_interp)
        valid_mask = ~nan_mask

        if np.all(nan_mask) or np.sum(valid_mask) < 2:
            print("Недостаточно валидных точек для интерполяции. Возвращаю нули.")
            return np.zeros_like(p_interp)

        # Если нет NaN - возвращаем как есть
        if not np.any(nan_mask):
            return p_interp

        t_valid = t[valid_mask]
        p_valid = p_interp[valid_mask]

        try:
            log_interp = np.interp(
                np.log(t[nan_mask] + 1e-100),
                np.log(t_valid),
                np.log(p_valid + 1e-10)
            )

            p_interp[nan_mask] = np.exp(log_interp)

        except Exception as e:
            print(f"Ошибка при логарифмической интерполяции: {e}")
            print("Пробую линейную интерполяцию в обычной шкале...")

            linear_interp = np.interp(
                t[nan_mask],
                t_valid,
                p_valid
            )
            p_interp[nan_mask] = linear_interp

        # Проверка результата интерполяции
        if np.any(np.isnan(p_interp)) or np.any(np.isinf(p_interp)):
            for i in np.where(np.isnan(p_interp) | np.isinf(p_interp))[0]:
                valid_indices = np.where(valid_mask)[0]
                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                p_interp[i] = p_valid[np.where(valid_indices == nearest_idx)[0][0]]

        return p_interp

    def resample_to_fixed_points(self, n_points=128):
        """
        Приведение кривой к фиксированному числу точек
        с равномерным шагом по логарифму времени
        """
        if len(self.df) == 0:
            return self
        
        # Получаем логарифмические границы
        log_t_min = np.log10(self.df['t_D'].min())
        log_t_max = np.log10(self.df['t_D'].max())
        
        # Равномерная сетка по log(t)
        log_t_new = np.linspace(log_t_min, log_t_max, n_points)
        t_new = 10 ** log_t_new
        
        # Линейная интерполяция
        p_new = np.interp(np.log(t_new), np.log(self.df['t_D']), self.df['P_wD'])
        
        # Обновляем датафрейм
        self.df = pd.DataFrame({
            't_D': t_new,
            'P_wD': p_new
        })
        
        return self

    def gauss_noize(self, mu=0, sigma=0.1):
        """
        Добавление гауссова шума с фиксированной дисперсией
        sigma^2 = 0.01 (стандартное отклонение = 0.1)
        """
        noise = np.random.normal(mu, sigma, len(self.df))
        self.df['P_wD_gauss'] = self.df['P_wD'] + noise
        return self

    # def gauss_noize(self, mu=0, sigma=5e-5):
    #     """
    #     Функция, добавляющая гауссов шум к данным.
    #     Данные с шумами добавляются в отдельный столбец P_wD_gauss в датафрейме.

    #     Получает:
    #         mu: матожидание.
    #         sigma: стандартное отклонение.

    #     Возвращает:
    #         self
    #     """
        
    #     signal_amplitude = self.df['P_wD'].max() - self.df['P_wD'].min()
        
    #     noise = np.random.normal(mu, sigma * signal_amplitude, len(self.df))
    #     self.df['P_wD_gauss'] = self.df['P_wD'] + noise
        
    #     return self

    def derivative(self, smoothig_alg='regression', delta=1):
        """
        Добавление в датафрейм столбца с логарифмической производной
        """
        
        self.df['ln_t_D'] = np.log(self.df['t_D']) # необходимо для последующего сглаживания

        
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


    def visualize(self, title, horizontal_line=True, point_type='x'):
        """
        Визуализация, зависимости давления от времени в loglog графике.
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(self.df['t_D'], self.df['P_wD_gauss'], 'g' + point_type, label='Теоретическое давление')
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

    def load_and_preprocess(self, filename: str, n_points: int = 128, 
                            t_min: float = None, t_max: float = None):
        """
        Загрузка и предобработка кривой ГДИС
        
        Parameters:
        -----------
        filename : str
            Путь к файлу с колонками t_D, dP_wD
        n_points : int
            Количество точек после интерполяции (по умолчанию 128)
        t_min, t_max : float, optional
            Границы для обрезания кривой (в логарифмическом масштабе)
            Если None - берем по данным
        """
        # Загрузка данных
        self.df = pd.read_csv(filename)
        
        # Убедимся, что данные отсортированы по времени
        self.df = self.df.sort_values('t_D').reset_index(drop=True)
        
        # Логарифмическое обрезание кривой
        self.df = self._crop_curve(t_min, t_max)
        
        # Логарифмическая интерполяция на n_points точек
        self.df_interpolated = self._log_interpolate(n_points)
        
        return self

    def _crop_curve(self, t_min: float = None, t_max: float = None):
        """
        Обрезание кривой по времени в логарифмическом масштабе
        """
        df_cropped = self.df.copy()
        
        if t_min is not None:
            df_cropped = df_cropped[df_cropped['t_D'] >= t_min]
        
        if t_max is not None:
            df_cropped = df_cropped[df_cropped['t_D'] <= t_max]
        
        # Логируем информацию об обрезании
        print(f"Кривая обрезана: {len(self.df)} -> {len(df_cropped)} точек")
        print(f"Диапазон t_D: [{df_cropped['t_D'].min():.2e}, {df_cropped['t_D'].max():.2e}]")
        
        return df_cropped.reset_index(drop=True)

    def _log_interpolate(self, n_points: int = 128):
        """
        Логарифмическая интерполяция на равномерную сетку в log пространстве
        """
        # Защита от дубликатов
        df_clean = self.df.drop_duplicates(subset=['t_D']).copy()
        
        # Создаем равномерную сетку в логарифмическом пространстве
        log_t_min = np.log10(df_clean['t_D'].min())
        log_t_max = np.log10(df_clean['t_D'].max())
        
        # Генерируем точки на равномерной логарифмической сетке
        log_t_interp = np.linspace(log_t_min, log_t_max, n_points)
        t_interp = 10 ** log_t_interp
        
        # Интерполяция производной давления
        # Используем кубический сплайн для гладкости
        f_interp = interpolate.interp1d(
            df_clean['t_D'], 
            df_clean['dP_wD'],
            kind='cubic',      # Кубическая интерполяция
            bounds_error=False,
            fill_value='extrapolate'  # Экстраполяция на краях (если нужно)
        )
        
        dP_interp = f_interp(t_interp)
        
        # Создаем DataFrame с интерполированными данными
        df_result = pd.DataFrame({
            't_D': t_interp,
            'dP_wD': dP_interp,
            'log_t_D': log_t_interp  # Сохраняем логарифм для удобства
        })
        
        print(f"Интерполяция завершена: {n_points} точек в логарифмическом масштабе")
        
        return df_result