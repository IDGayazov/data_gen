import numpy as np
import pandas as pd

from scipy.special import k0, k1
from model.reservoir_model import ReservoirModel

from inversion.shtefest_algorithm import ShtefestAlgorithm

class InfiniteDualPorosityReservoirModel(ReservoirModel):
    """
    Класс для модели двухпорового пласта.

    Атрибуты:
        C_D:   безразмерная емкость
        S:     скин-фактор
        omega: коэффициент ёмкости
        lam:   коэффициент межпорового перетока
    """

    def __init__(self, C_D, S, omega, lam):
        super().__init__()
        self.C_D = C_D
        self.S = S
        self.omega = omega
        self.lam = lam


    def f(self, u, omega, lam):
        return (omega * (1 - omega) * u + lam) / ((1 - omega) * u + lam)


    def P_wD_laplace_rd(self, u, r_D):
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
        f_sqrt_u = np.sqrt(self.f(u, self.omega, self.lam) * u)
        numerator = k0(r_D * f_sqrt_u)
        denominator = u * f_sqrt_u * k1(f_sqrt_u)

        return numerator / denominator


    def P_wD_laplace(self, u, r_D):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u      : параметр преобразования Лапласа
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """

        return self.P_wD_laplace_rd(u, r_D)


    def F(self, s):
        return self.agarwal_filter(self.P_wD_laplace, s, 1, self.S, self.C_D)


if __name__ == "__main__":
    df = pd.read_csv('./dataset/params/1300.csv')

    model = InfiniteDualPorosityReservoirModel(C_D=df['C_D'][0],
                                               S=df['S'][0],
                                               omega=df['omega'][0],
                                               lam=df['lambda'][0])

    t_D_array = np.logspace(0, 7, 1000)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-8) \
        .derivative(smoothig_alg='regression', delta=0.5) \
        .visualize('Кривые давления (модель двойной пористости)')