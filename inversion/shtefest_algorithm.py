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

    def inverse(self, F, t) -> float:
        if self.N % 2 != 0:
            raise ValueError("Порядок алгоритма N должен быть четным.")

        ln2_t = math.log(2.0) / t
        N_half = self.N // 2

        V = np.zeros(self.N)
        for i in range(1, self.N + 1):
            k = min(i, N_half)
            start = (i + 1) // 2
            end = min(i, N_half) + 1
            sum_val = 0.0
            for j in range(start, end):
                numerator = j ** N_half * math.factorial(2 * j)
                denominator = (math.factorial(N_half - j) *
                               math.factorial(j) *
                               math.factorial(j - 1) *
                               math.factorial(i - j) *
                               math.factorial(2 * j - i))
                sum_val += numerator / denominator
            V[i - 1] = (-1) ** (i + N_half) * sum_val

        result = 0.0
        for i in range(1, self.N + 1):
            s_val = i * ln2_t
            result += V[i - 1] * F(s_val)

        return ln2_t * result