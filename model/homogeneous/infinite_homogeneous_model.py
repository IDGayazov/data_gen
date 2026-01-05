import warnings

import numpy as np

from scipy.special import k0, k1

from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.reservoir_model import ReservoirModel


class InfiniteHomogeneousReservoirModel(ReservoirModel):
    """
    Класс для однородной модели пласта.

    Атрибуты:
        C_D: безразмерная емкость
        S:   скин-фактор
    """

    def __init__(self, C_D, S):
        super().__init__()
        self.C_D = C_D
        self.S = S


    def P_wD_laplace(self, u, C_D, S):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            C_D:     безразмерная емкость
            S:       скин-фактор
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        sqrt_u = np.sqrt(u)
        numerator = k0(sqrt_u) + S * sqrt_u * k1(sqrt_u)
        denominator = u * (sqrt_u * k1(sqrt_u) + C_D * u * (k0(sqrt_u) + S * sqrt_u * k1(sqrt_u)))

        return numerator / denominator


    def F(self, s):
        return self.P_wD_laplace(s, self.C_D, self.S)

if __name__ == "__main__":
    model = InfiniteHomogeneousReservoirModel(C_D=100, S=1.0)

    t_D_array = np.logspace(0, 10, 500)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-7) \
        .derivative(smoothig_alg='regression', delta=0.3) \
        .visualize('Кривые давления (однородный пласт)')