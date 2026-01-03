import numpy as np
import pandas as pd

from scipy.special import k0, k1, i0, i1

from conversion.homogeneous_converter import DimensionConverter
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.reservoir_model import ReservoirModel


class FiniteDualPorosityReservoirModel(ReservoirModel):
    """
    Класс для модели двойной пористости.

    Атрибуты:
        C_D:   безразмерная емкость
        S:     скин-фактор
        omega: коэффициент ёмкости
        lam:   коэффициент межпорового перетока
        R_D_e: радиус границ
    """

    def __init__(self, C_D, S, omega, lam, R_D_e):
        super().__init__()
        self.C_D = C_D
        self.S = S
        self.omega = omega
        self.lam = lam
        self.R_D_e = R_D_e


    def f(self, u, omega, lam):
        return (omega * (1 - omega) * u + lam) / ((1 - omega) * u + lam)


    def P_wD_laplace_rd_bound(self, u, r_D, r_D_e, omega, lam):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            C_D:     безразмерная емкость
            omega:   коэффициент ёмкости
            lam:     коэффициент межпорового перетока
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        f_u = self.f(u, omega, lam)
        sqrt_u_fu = np.sqrt(u * f_u)
        r_sqrt_u_fu = r_D_e * sqrt_u_fu

        K0_sqrt = k0(sqrt_u_fu)
        I1_r_sqrt = i1(r_sqrt_u_fu)
        K1_r_sqrt = k1(r_sqrt_u_fu)
        I0_sqrt = i0(sqrt_u_fu)
        K1_sqrt = k1(sqrt_u_fu)
        I1_sqrt = i1(sqrt_u_fu)

        numerator = K0_sqrt * I1_r_sqrt + K1_r_sqrt * I0_sqrt
        denominator = u * sqrt_u_fu * (K1_sqrt * I1_r_sqrt - K1_r_sqrt * I1_sqrt)

        return numerator / denominator


    def P_wD_laplace(self, u, r_D):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """

        return self.P_wD_laplace_rd_bound(u, r_D, self.R_D_e, self.omega, self.lam)


    def F(self, s):
        return self.agarwal_filter(self.P_wD_laplace, s, 1, self.S, self.C_D)


if __name__ == "__main__":
    model = FiniteDualPorosityReservoirModel(C_D=20, S=0.03, omega=0.14155166176665768, lam=3.2408607033195305e-06, R_D_e=200)

    t_D_array = np.logspace(0, 7, 500)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-8) \
        .derivative(smoothig_alg='regression', delta=0.5) \
        .visualize('Кривые давления для модели двойной пористости c круговой непроницаемой границей', False)