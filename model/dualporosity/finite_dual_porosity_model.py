import warnings

import numpy as np
import pandas as pd

from scipy.special import k0, k1, i0, i1
from scipy.special import k0, k1, i0, i1, kve, ive

from conversion.homogeneous_converter import HomogeneousConverter
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
        u = np.atleast_1d(u)
        f_u = self.f(u, omega, lam)
        
        # Аргумент теперь включает влияние f_u
        z = np.sqrt(u * f_u)
        z_re = r_D_e * z
        z_rd = 1.0 * z # обычно r_D на стенке скважины = 1

        # Ключевой момент: считаем отношение K1/I1 так, чтобы не было переполнения
        # ratio = K1(z_re) / I1(z_re)
        # Используем формулу: (kve(1, z_re) * exp(-z_re)) / (ive(1, z_re) * exp(z_re))
        # Это дает exp(-2 * z_re), что стремится к 0 при больших u
        
        # Ограничиваем аргумент экспоненты, чтобы не получить overflow в саму другую сторону
        exp_term = np.exp(-2 * np.clip(z_re, 0, 700))
        ratio = (kve(1, z_re) / ive(1, z_re)) * exp_term

        # Используем стандартные k0, i0, но теперь они защищены тем, 
        # что мы делим всё выражение на I1(z_re)
        # Формула превращается в:
        # (K0(z) + ratio * I0(z)) / (u * z * (K1(z) - ratio * I1(z)))
        
        num = k0(z) + ratio * i0(z)
        den = u * z * (k1(z) - ratio * i1(z))

        res = num / den
        
        # Если z слишком велик (очень малые времена), ratio станет 0, 
        # и формула выродится в решение для бесконечного пласта: k0(z) / (u * z * k1(z))
        return np.where(np.isfinite(res), res, 1.0 / (u * z))


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