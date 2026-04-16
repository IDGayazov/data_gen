import warnings
from functools import lru_cache

import numpy as np

from scipy.special import k0, k1, i0, i1

from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.reservoir_model import ReservoirModel
# from scipy.special import kn, in_  
# from scipy.special import kve, ive


class FiniteDualPermeabilityReservoirModel(ReservoirModel):
    """
    Класс для модели двойной проницаемости.

    Атрибуты:
        - C_D:   безразмерная емкость
        - S:     скин-фактор
        - omega: коэффициент ёмкости
        - lam:   коэффициент межпорового перетока
        - kappa: отношение kh
    """

    def __init__(self, C_D, S, omega, lam, kappa, R_D_E):
        super().__init__()
        self.C_D = C_D
        self.S = S
        self.omega = omega
        self.lam = lam
        self.kappa = kappa
        self.R_D_E = R_D_E

    @lru_cache(maxsize=1000)
    def delta(self, s):
        """
        Вычисляет функцию Δ(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            Δ(s): значение функции
        """
        omega = self.omega
        lam = self.lam
        kappa = self.kappa

        term1 = ((1 - omega) * s + lam) / (1 - kappa)
        term2 = (omega * s + lam) / kappa
        term3 = (4 * lam ** 2) / (kappa * (1 - kappa))

        delta_s = np.sqrt((term1 - term2) ** 2 + term3)

        return delta_s

    @lru_cache(maxsize=1000)
    def sigma1_squared(self, s):
        """
        Вычисляет функцию σ₁²(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            σ₁²(s): значение функции
        """
        omega, lam, kappa = self.omega, self.lam, self.kappa

        sum_terms = ((1 - omega) * s + lam) / (1 - kappa) + (omega * s + lam) / kappa
        delta_s = self.delta(s)
        sigma1_sq = 0.5 * (sum_terms + delta_s)

        return sigma1_sq

    @lru_cache(maxsize=1000)
    def sigma2_squared(self, s):
        """
        Вычисляет функцию σ₂²(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            σ₂²(s): значение функции
        """
        omega, lam, kappa = self.omega, self.lam, self.kappa

        sum_terms = ((1 - omega) * s + lam) / (1 - kappa) + (omega * s + lam) / kappa
        delta_s = self.delta(s)
        sigma2_sq = 0.5 * (sum_terms - delta_s)

        return sigma2_sq

    @lru_cache(maxsize=1000)
    def a1(self, s):
        """
        Вычисляет функцию a₁(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            a₁(s): значение функции
        """
        omega, lam, kappa = self.omega, self.lam, self.kappa

        sigma1_sq = self.sigma1_squared(s)

        a1_value = 1 + (1 / lam) * ((1 - omega) * s - (1 - kappa) * sigma1_sq)

        return a1_value

    def a2(self, s):
        """
        Вычисляет функцию a₂(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            a₂(s): значение функции
        """
        omega, lam, kappa = self.omega, self.lam, self.kappa

        sigma2_sq = self.sigma2_squared(s)

        a2_value = 1 + (1 / lam) * ((1 - omega) * s - (1 - kappa) * sigma2_sq)

        return a2_value

    @lru_cache(maxsize=1000)
    def b(self, s):
        """
        Вычисляет функцию b(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            b(s): значение функции
        """
        omega, lam, kappa = self.omega, self.lam, self.kappa

        sigma1_sq = self.sigma1_squared(s)
        sigma2_sq = self.sigma2_squared(s)

        a1_val = self.a1(s)
        a2_val = self.a2(s)

        sigma1 = np.sqrt(sigma1_sq)
        sigma2 = np.sqrt(sigma2_sq)

        term1 = s * (1 - a1_val) * (kappa * a2_val + 1 - kappa) * sigma2 * k0(sigma1) * k1(sigma2)
        term2 = s * (1 - a2_val) * (kappa * a1_val + 1 - kappa) * sigma1 * k0(sigma2) * k1(sigma1)

        return term1 - term2

    @lru_cache(maxsize=1000)
    def B1(self, s):
        """
        Вычисляет функцию B_1(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            B_1(s): значение функции
        """
        sigma2_sq = self.sigma2_squared(s)
        a2_val = self.a2(s)
        sigma2 = np.sqrt(sigma2_sq)

        return -(1 - a2_val) * k0(sigma2) / self.b(s)

    @lru_cache(maxsize=1000)
    def B2(self, s):
        """
        Вычисляет функцию B_2(s) согласно уравнению.

        Аргументы:
            s: параметр преобразования Лапласа

        Возвращает:
            B_2(s): значение функции
        """
        sigma1_sq = self.sigma1_squared(s)
        a1_val = self.a1(s)
        sigma1 = np.sqrt(sigma1_sq)

        return (1 - a1_val) * k0(sigma1) / self.b(s)


    def P_wD_laplace_rD(self, u, r_D, r_D_e, omega, lam, kappa):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            C_D:     безразмерная емкость
            r_D:     безразмерный радиус
            omega:   коэффициент ёмкости
            lam:     коэффициент межпорового перетока
            kappa:   отношение kh
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        sigma1_sq = self.sigma1_squared(u)
        a1_val = self.a1(u)
        sigma1 = np.sqrt(sigma1_sq)

        sigma2_sq = self.sigma2_squared(u)
        a2_val = self.a2(u)
        sigma2 = np.sqrt(sigma2_sq)

        p1_1 = a1_val * self.B1(u) * (k0(r_D * sigma1) + (k1(sigma1 * r_D_e) / i1(sigma1 * r_D_e)) * i0(sigma1 * r_D))
        p1_2 = a2_val * self.B2(u) * (k0(sigma2 * r_D) + (k1(sigma2 * r_D_e) / i1(sigma2 * r_D_e)) * i0(sigma2 * r_D))

        return p1_1 + p1_2

    def P_wD_laplace(self, u, r_D):
        """
        Вычисляет решение для забойного давления в пространстве Лапласа.
        Аргументы:
            u:       параметр преобразования Лапласа
            C_D:     безразмерная емкость
            r_D:     безразмерный радиус
            omega:   коэффициент ёмкости
            lam:     коэффициент межпорового перетока
            kappa:   отношение kh
        Возвращает:
            P_wD(u): значение безразмерного давления в пространстве Лапласа
        """
        return self.P_wD_laplace_rD(u, r_D, self.R_D_E, self.omega, self.lam, self.kappa)

    def F(self, s):
        """
        Забойное давление в скважине (r_D = 1)
        """
        return self.agarwal_filter(self.P_wD_laplace, s, 1, self.S, self.C_D)


if __name__ == "__main__":
    model = FiniteDualPermeabilityReservoirModel(C_D=20, S=1, R_D_E=200, omega=0.5, lam=7e-6, kappa=0.5)

    t_D_array = np.logspace(0, 6, 128)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-9) \
        .derivative(smoothig_alg='regression', delta=0.3) \
        .visualize('Кривые давления для двойной проницаемости с круговой непроницаемой границей')