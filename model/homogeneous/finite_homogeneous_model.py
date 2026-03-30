import warnings

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


    def P_wD_laplace_with_bound(self, u, r_D, r_D_e, eps=1e-10):
        from scipy.special import k0, k1, i0, i1, ive, kve

        u = np.atleast_1d(u)
        sqrt_u = np.sqrt(u)
        
        z = sqrt_u
        z_rd = r_D * sqrt_u
        z_re = r_D_e * sqrt_u

        # Чтобы избежать inf, используем отношение функций или масштабированные функции
        # Для больших z_re: K1(z_re)/I1(z_re) практически равно 0
        # Используем ive(n, z) = i_n(z) * exp(-z)
        
        # Считаем отношение R = K1(z_re) / I1(z_re)
        # kve(1, z_re) = k1 * exp(z_re), ive(1, z_re) = i1 * exp(-z_re)
        # R = (kve * exp(-z_re)) / (ive * exp(z_re)) = (kve/ive) * exp(-2*z_re)
        
        # Безопасный расчет отношения:
        ratio = (kve(1, z_re) / ive(1, z_re)) * np.exp(-2 * z_re)
        
        # Теперь переписываем формулу, разделив всё на I1(z_re)
        # num = K0(z_rd) + ratio * I0(z_rd)
        # den = u * sqrt_u * (K1(z) - ratio * I1(z))
        
        num = k0(z_rd) + ratio * i0(z_rd)
        den = u * sqrt_u * (k1(z) - ratio * i1(z))
        
        return num / den


    # def P_wD_laplace_with_bound(self, u, r_D, r_D_e, eps=1e-10):
    #     """
    #     Вычисляет решение для забойного давления в пространстве Лапласа.
    #     Аргументы:
    #         u:       параметр преобразования Лапласа
    #         r_D:     безразмерный радиус
    #         r_D_e:   безразмерный радиус коллектора
    #         C_D:     безразмерная емкость
    #         S:       скин-фактор
    #     Возвращает:
    #         P_wD(u): значение безразмерного давления в пространстве Лапласа
    #     """
    #     warnings.filterwarnings('ignore', category=RuntimeWarning)

    #     u = np.asarray(u, dtype=np.float64)
    #     u_safe = np.where(np.abs(u) < eps, np.sign(u) * eps, u)

    #     sqrt_u = np.sqrt(u_safe)

    #     k0_rd = k0(r_D * sqrt_u)
    #     i0_rd = i0(r_D * sqrt_u)
    #     k1_rd_e = k1(r_D_e * sqrt_u)
    #     i1_rd_e = i1(r_D_e * sqrt_u)
    #     k1_sqrt = k1(sqrt_u)
    #     i1_sqrt = i1(sqrt_u)

    #     numerator = k0_rd * i1_rd_e + k1_rd_e * i0_rd
    #     denominator = u_safe * sqrt_u * (k1_sqrt * i1_rd_e - k1_rd_e * i1_sqrt)

    #     denominator_safe = np.where(
    #         np.abs(denominator) < eps,
    #         np.sign(denominator) * eps,
    #         denominator
    #     )

    #     print("k0_rd = ", k0_rd)
    #     print("i0_rd = ", i0_rd)
    #     print("k1_rd_e = ", k1_rd_e)
    #     print("i1_rd_e = ", i1_rd_e)
    #     print("k1_sqrt = ", k1_sqrt)
    #     print("i1_sqrt = ", i1_sqrt)
    #     print("u = ", u)
    #     print("sqrt_u = ", sqrt_u)
    #     print("num = ", numerator)
    #     print("denom = ", denominator_safe)

    #     result = numerator / denominator_safe

    #     return result


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
        p_d = self.P_wD_laplace_with_bound(u, r_D, self.R_D_E)
        return p_d


    def F(self, s):
        """
        Забойное давление в скважине (r_D = 1)
        """
        return self.agarwal_filter(self.P_wD_laplace, s, 1, self.S, self.C_D)

if __name__ == "__main__":
    model = FiniteHomogeneousReservoirModel(C_D=300, S=5, R_D_E=1000)

    t_D_array = np.logspace(-2, 7, 128)
    alg = ShtefestAlgorithm(N=12)

    model.pressure(t_D_array, alg)\
         .gauss_noize(mu=0, sigma=5e-4)\
         .derivative(smoothig_alg='regression', delta=0.3)\
         .visualize('Кривые давления (однородный пласт c круговой границей)', False)

    df = model.get_pressure()['P_wD']

    print(df.isna().sum())