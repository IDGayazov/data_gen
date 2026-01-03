import numpy as np

from scipy.special import k0, k1, i0, i1

from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.reservoir_model import ReservoirModel


class InfiniteRadialCompositeReservoirModel(ReservoirModel):
    """
    Класс для радиально-композитной модели пласта с бесконечной границей.

    Атрибуты:
        C_D:     безразмерная емкость
        S:       скин-фактор
        M12:     отношение подвижностей
        omega12: отношение ёмкостей
    """

    def __init__(self, C_D, S, M1, M2, omega1, omega2, r_fD):
        super().__init__()
        self.C_D = C_D
        self.S = S
        self.M1 = M1
        self.M2 = M2
        self.omega1 = omega1
        self.omega2 = omega2
        self.r_fD = r_fD


    def a(self, s):
        return self.r_fD * np.sqrt(s)


    def b(self, s):
        M12 = self.M1 / self.M2
        omega12 = self.omega1 / self.omega2
        x21 = M12 / omega12
        return self.r_fD * np.sqrt(x21 * s)


    def KI(self, s):
        a_val = self.a(s)
        b_val = self.b(s)
        M12 = self.M1 / self.M2
        omega12 = self.omega1 / self.omega2
        x21 = M12 / omega12
        coeff = np.sqrt(x21) / M12

        numerator = i1(a_val) * k0(b_val) + coeff * i0(a_val) * k1(b_val)
        denominator = k1(a_val) * k0(b_val) - coeff * k0(a_val) * k1(b_val)

        return numerator / denominator


    def B(self, s):
        sqrt_s = np.sqrt(s)
        return s * sqrt_s * (self.KI(s) * k1(sqrt_s) - i1(sqrt_s))


    def P_wD_laplace(self, u, r_D):
        """
        Вычисляет решение для пластового давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            C_D:     безразмерная емкость
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        sqrt_u = np.sqrt(u)
        numerator = self.KI(u) * k0(r_D * sqrt_u) + i0(r_D * sqrt_u)

        return numerator / self.B(u)


    def F(self, s):
        return self.agarwal_filter(self.P_wD_laplace, s, 1, self.S, self.C_D)


if __name__ == "__main__":
    model = InfiniteRadialCompositeReservoirModel(C_D=20, S=1, M1=5, M2=3, omega1=0.5, omega2=0.3, r_fD=200)

    t_D_array = np.logspace(0, 10, 3000)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-9) \
        .derivative(smoothig_alg='regression', delta=0.3) \
        .visualize('Кривые давления для радиально-композитной модели с бесконечной границей', False)