import math
import numpy as np

from inversion.laplas_inversion_method import LaplasInversionMethod


class ShtefestAlgorithm(LaplasInversionMethod):
    """
    Алгоритм Штефеста для численного обращения Лапласа

    Атрибуты:
        N: порядок алгоритма (должен быть четным). Обычно 8, 10, 16, 18.
           Более высокий N дает точность для гладких функций, но может стать неустойчивым
           (вывод из статьи https://dl.acm.org/doi/pdf/10.1145/361953.361969 (Алгоритм 368)).
    """

    def __init__(self, N):
        self.N = N
        self._fact_cache = {}  # Кеш для факториалов
        self.V = self._calculate_coefficients()

    def _factorial(self, n):
        """Мемоизированный факториал"""
        if n in self._fact_cache:
            return self._fact_cache[n]

        result = math.factorial(n)
        self._fact_cache[n] = result
        return result

    def _calculate_coefficients(self):
        N_half = self.N // 2
        V = np.zeros(self.N)

        # Предвычисляем факториалы для часто используемых значений
        max_fact = 2 * N_half
        for i in range(max_fact + 1):
            self._factorial(i)  # Заполняем кеш

        for i in range(1, self.N + 1):
            start = (i + 1) // 2
            end = min(i, N_half) + 1

            sum_val = 0.0
            for j in range(start, end):
                numerator = j ** N_half * self._factorial(2 * j)
                denominator = (self._factorial(N_half - j) *
                               self._factorial(j) *
                               self._factorial(j - 1) *
                               self._factorial(i - j) *
                               self._factorial(2 * j - i))
                sum_val += numerator / denominator

            V[i - 1] = (-1) ** (i + N_half) * sum_val

        return V

    def inverse(self, F, t) -> float:
        """Численное обращение преобразования Лапласа по алгоритму Штефеста"""
        if self.N % 2 != 0:
            raise ValueError("Порядок алгоритма N должен быть четным.")

        ln2_t = math.log(2.0) / t
        result = 0.0

        for i in range(1, self.N + 1):
            s_val = i * ln2_t
            result += self.V[i - 1] * F(s_val)

        return ln2_t * result

    def inverse_vectorized(self, F, t_array) -> np.ndarray:
        """Обращение для массива времен t"""
        if not isinstance(t_array, np.ndarray):
            t_array = np.array(t_array)

        results = np.zeros_like(t_array)

        for idx, t in enumerate(t_array):
            results[idx] = self.inverse(F, t)

        return results