import numpy as np

from scipy.special import k0, k1, i0, i1

from model.reservoir_model import ReservoirModel
from inversion.shtefest_algorithm import ShtefestAlgorithm


class FiniteHomogeneousReservoirModel(ReservoirModel):
    """
    Класс для однородной модели скважины с круговой границей, решение получено при помощи уравнения Агарвала.

    Атрибуты:
        C_D: безразмерная емкость
        S  : скин-фактор
    """

    def __init__(self, C_D, S, R_D_E):
        super().__init__()
        self.C_D = C_D
        self.S = S
        self.R_D_E = R_D_E


    def P_wD_laplace_with_bound(self, u, r_D, r_D_e):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            r_D:     безразмерный радиус
            r_D_e:   безразмерный радиус коллектора
            C_D:     безразмерная емкость
            S:       скин-фактор
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        sqrt_u = np.sqrt(u)
        numerator = k0(r_D * sqrt_u) * i1(r_D_e * sqrt_u) + k1(r_D_e * sqrt_u) * i0(r_D * sqrt_u)
        denominator = u * sqrt_u * (k1(sqrt_u) * i1(r_D_e * sqrt_u) - k1(r_D_e * sqrt_u) * i1(sqrt_u))

        return numerator / denominator


    def P_wD_laplace(self, u, r_D):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            r_D:     безразмерный радиус
            C_D:     безразмерная емкость
            S:       скин-фактор
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        return self.P_wD_laplace_with_bound(u, r_D, self.R_D_E)


    def F(self, s):
        """
        Забойное давление в скважине (r_D = 1)
        """
        return self.agarwal_filter(self.P_wD_laplace, s, 1, self.S, self.C_D)

if __name__ == "__main__":
    model = FiniteHomogeneousReservoirModel(C_D=100, S=3.0, R_D_E=500)

    t_D_array = np.logspace(0, 7, 1000)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg)\
         .gauss_noize(mu=0, sigma=5e-9)\
         .derivative(smoothig_alg='regression', delta=0.3)\
         .visualize('Кривые давления (однородный пласт c круговой границей)', False)