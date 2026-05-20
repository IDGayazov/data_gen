import numpy as np
import pandas as pd


class BourdaisLogarithmicDerivative:
    """
    Логарифмическая производная Бурде.

    Поддерживает два типа сглаживания c параметром интервала сглаживания delta:
        - многоточечная регрессия
        - скользящее окно

    Атрибут:
        f: DataFrame, таблично заданная функция
    """

    def __init__(self, f):
        self.f = f


    def derivate_with_regression_smoothing(self, t, delta, x_col_name='ln_t_D', y_col_name='P_wD'):
        """
        Вычисление логарифмической производной в точке t с помощью сглаживания многоточечной регрессией

        Принимает:
            t: точка или массив для вычисления производной
            delta: интервал сглаживания
            x_col_name: область определения f
            y_col_name: область значений f
        Возвращает:
            Скалярное значение или вектор с вычисленными производными
        """
        if np.isscalar(t):
            if t <= 0:
                return np.nan
            
            ln_t = np.log(t)

            ln_delta = delta * np.log(10) 

            t_local_points = self.f[self.f[x_col_name].between(ln_t - ln_delta, ln_t + ln_delta)]

            if len(t_local_points) < 2:
                t_local_points = self.f[self.f[x_col_name].between(ln_t - 2*ln_delta, ln_t + 2*ln_delta)]

            if len(t_local_points) < 2:
                return np.nan
            
            x_data = t_local_points[x_col_name].to_numpy(dtype=float)
            y_data = t_local_points[y_col_name].to_numpy(dtype=float)
            
            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data = x_data[valid_mask]
                y_data = y_data[valid_mask]
                
                if len(x_data) < 2:
                    return np.nan
            
            if len(np.unique(x_data)) < 2:
                return np.nan
            
            try:
                slope, _ = np.polyfit(x_data, y_data, 1)
                if np.isnan(slope) or np.isinf(slope):
                    return np.nan
                return slope
            except (np.linalg.LinAlgError, ValueError):
                return np.nan
        elif isinstance(t, pd.Series):
            return t.apply(lambda x: self.derivate_with_regression_smoothing(x, delta, x_col_name, y_col_name))
        else:
            t_series = pd.Series(t)
            return t_series.apply(lambda x: self.derivate_with_regression_smoothing(x, delta, x_col_name, y_col_name))


    def derivative_with_rolling_window_smoothing(self, t, delta, x_col_name='ln_t_D', y_col_name='P_wD'):
        """
        Вычисление логарифмической производной в точке t с помощью сглаживания скользящим окном

        Принимает:
            t: точка или массив для вычисления производной
            delta: интервал сглаживания
            x_col_name: область определения f
            y_col_name: область значений f
        Возвращает:
            Скалярное значение или вектор с вычисленными производными
        """
        if np.isscalar(t):
            if t <= 0:
                return np.nan

            ln_t = np.log(t)
            ln_delta = delta * np.log(10)
            t_local_points = self.f[self.f[x_col_name].between(ln_t - ln_delta, ln_t + ln_delta)]

            if len(t_local_points) < 2:
                return np.nan

            first_idx = t_local_points.index[0]
            last_idx = t_local_points.index[-1]

            # Ближайшая к ln_t точка как центральная (вместо float ==)
            center_idx = (self.f[x_col_name] - ln_t).abs().idxmin()

            left_point  = (t_local_points.loc[first_idx, x_col_name],  t_local_points.loc[first_idx, y_col_name])
            center_point = (self.f.loc[center_idx, x_col_name], self.f.loc[center_idx, y_col_name])
            right_point = (t_local_points.loc[last_idx, x_col_name],  t_local_points.loc[last_idx, y_col_name])

            m_left  = self.slope_from_two_points(left_point, center_point)
            m_right = self.slope_from_two_points(center_point, right_point)

            if np.isnan(m_left) or np.isnan(m_right):
                return np.nan

            l_left  = center_point[0] - left_point[0]
            l_right = right_point[0] - center_point[0]

            if l_left + l_right == 0:
                return np.nan

            # Формула Бурде: левый наклон взвешивается на правый интервал и наоборот
            return (m_left * l_right + m_right * l_left) / (l_left + l_right)
        elif isinstance(t, pd.Series):
            return t.apply(lambda x: self.derivative_with_rolling_window_smoothing(x, delta, x_col_name, y_col_name))
        else:
            t_series = pd.Series(t)
            return t_series.apply(
                lambda x: self.derivative_with_rolling_window_smoothing(x, delta, x_col_name, y_col_name))


    def slope_from_two_points(self, point1, point2):
        """
        Вычисление угла наклона прямой между двумя точками

        Принимает:
            point1, point2: кортежи (x, y)

        Возвращает:
            Танегенс угла наклона прямой
        """
        x1, y1 = point1
        x2, y2 = point2

        if x1 == x2:
            return np.nan

        slope_tangent = (y2 - y1) / (x2 - x1)

        return slope_tangent