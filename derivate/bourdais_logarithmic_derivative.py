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
            t_local_points = self.f[self.f[x_col_name].between(ln_t - delta, ln_t + delta)]
            
            # Если недостаточно точек, расширяем окно
            if len(t_local_points) < 2:
                # Пробуем расширить окно в 2 раза
                t_local_points = self.f[self.f[x_col_name].between(ln_t - 2*delta, ln_t + 2*delta)]
                if len(t_local_points) < 2:
                    # Если все еще недостаточно, используем все доступные точки
                    t_local_points = self.f
                    if len(t_local_points) < 2:
                        return np.nan
            
            # Проверка на NaN в данных
            x_data = t_local_points[x_col_name].to_numpy(dtype=float)
            y_data = t_local_points[y_col_name].to_numpy(dtype=float)
            
            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
                # Удаляем NaN значения
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data = x_data[valid_mask]
                y_data = y_data[valid_mask]
                
                if len(x_data) < 2:
                    return np.nan
            
            # Проверка на одинаковые значения x (деление на ноль)
            if len(np.unique(x_data)) < 2:
                return np.nan
            
            try:
                slope, _ = np.polyfit(x_data, y_data, 1)
                # Проверка на валидность результата
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
            ln_t = np.log(t)
            t_local_points = self.f[self.f[x_col_name].between(ln_t - delta, ln_t + delta)]

            first_idx = t_local_points.index[0]
            last_idx = t_local_points.index[-1]

            first_point = (t_local_points.loc[first_idx, x_col_name], t_local_points.loc[first_idx, y_col_name])
            t_point = (ln_t, self.f[self.f[x_col_name] == ln_t][y_col_name].values[0])
            last_point = (t_local_points.loc[last_idx, x_col_name], t_local_points.loc[last_idx, y_col_name])

            m1 = self.slope_from_two_points(first_point, t_point)
            m2 = self.slope_from_two_points(t_point, last_point)

            l1 = t_point[0] - first_point[0]
            l2 = last_point[0] - t_point[0]

            return (l1 * m1 + l2 * m2) / (l1 + l2)
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