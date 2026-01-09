import warnings

import numpy as np

from scipy.special import k0, k1, i0, i1

from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.reservoir_model import ReservoirModel


class InfiniteRadialCompositeReservoirModel(ReservoirModel):
    """
    Класс для радиально-композитной модели пласта с бесконечной границей.
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

        self.M12 = self.M1 / self.M2
        self.omega12 = self.omega1 / self.omega2
        self.M21 = 1.0 / self.M12
        self.x21 = self.M12 / self.omega12

        # print(f"M12 (M1/M2): {self.M12:.6f}")
        # print(f"M21 (M2/M1): {self.M21:.6f}")
        # print(f"ω12 (ω1/ω2): {self.omega12:.6f}")
        # print(f"x21 (M12/ω12): {self.x21:.6f}")
        # print(f"r_fD: {self.r_fD:.1f}")

    def a(self, s):
        """
        a(s) = r_fD * √s
        """
        return self.r_fD * np.sqrt(s)

    def b(self, s):
        """
        b(s) = r_fD * √(x21 * s)
        где x21 = M12/ω12
        """
        return self.r_fD * np.sqrt(self.x21 * s)

    def KI(self, s):
        """
        Функция KI(s):

        KI(s) = [I₁(a)·K₀(b) + (√x21/M21)·I₀(a)·K₁(b)] /
                [K₁(a)·K₀(b) - (√x21/M21)·K₀(a)·K₁(b)]

        где:
        - a = r_fD·√s
        - b = r_fD·√(x21·s)
        - x21 = M12/ω12
        - M21 = 1/M12
        """
        a_val = self.a(s)
        b_val = self.b(s)

        coeff = np.sqrt(self.x21) / self.M21  # √x21 / M21

        # Числитель: I₁(a)·K₀(b) + coeff·I₀(a)·K₁(b)
        numerator = i1(a_val) * k0(b_val) + coeff * i0(a_val) * k1(b_val)

        # Знаменатель: K₁(a)·K₀(b) - coeff·K₀(a)·K₁(b)
        denominator = k1(a_val) * k0(b_val) - coeff * k0(a_val) * k1(b_val)

        return numerator / denominator

    def B(self, s):
        """
        Функция B(s) согласно формуле (90) из статьи:

        B(s) = s·√s·[KI(s)·K₁(√s) - I₁(√s)]
        """
        sqrt_s = np.sqrt(s)
        ki_val = self.KI(s)

        return s * sqrt_s * (ki_val * k1(sqrt_s) - i1(sqrt_s))

    def P_wD_laplace(self, u, r_D=1.0):
        """
        Решение для безразмерного давления в пространстве Лапласа.

        P_wD(u) = [KI(u)·K₀(r_D·√u) + I₀(r_D·√u)] / B(u)

        Аргументы:
            u: параметр преобразования Лапласа
            r_D: безразмерный радиус (по умолчанию 1.0 - скважина)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            sqrt_u = np.sqrt(u)
            numerator = self.KI(u) * k0(r_D * sqrt_u) + i0(r_D * sqrt_u)

            return numerator / self.B(u)

    def F(self, s):
        """
        Функция фильтра Агарвала для учета скин-фактора и влияния ствола.
        """
        return self.agarwal_filter(self.P_wD_laplace, s, 1.0, self.S, self.C_D)


if __name__ == "__main__":
    # M12(M1 / M2): 0.863334
    # M21(M2 / M1): 1.158300
    # ω12(ω1 / ω2): 2.724104
    # x21(M12 / ω12): 0.316924
    # r_fD: 50.0
    C_D = 20
    S = 2
    M1 = np.random.uniform(0.5, 2)
    M2 = np.random.uniform(0.2, 5)
    omega1 = np.random.uniform(0.05, 0.3)
    omega2 = np.random.uniform(0.05, 0.3)
    r_fD = 100
    model = InfiniteRadialCompositeReservoirModel(C_D=C_D,
                                                  S=S,
                                                  M1=M1,
                                                  M2=M2,
                                                  omega1=omega1,
                                                  omega2=omega2,
                                                  r_fD=r_fD)

    t_D_array = np.logspace(0, 10, 3000)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-9) \
        .derivative(smoothig_alg='regression', delta=0.3) \
        .visualize('Кривые давления для радиально-композитной модели с бесконечной границей', False)