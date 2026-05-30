import numpy as np
import pandas as pd

from scipy.special import i0e, i1e, k0e, k1e

from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.reservoir_model import ReservoirModel


class InfiniteRadialCompositeReservoirModel(ReservoirModel):
    """
    Класс для радиально-композитной модели пласта с бесконечной границей.

    Нормировка времени — по зоне 1: t_D = k1·t / (μ·φ1·ct1·rw²).
    Аргументы Бесселя: a(s) = r_fD·√s (зона 1), b(s) = r_fD·√(x21·s) (зона 2).
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

    def a(self, s):
        return self.r_fD * np.sqrt(s)

    def b(self, s):
        return self.r_fD * np.sqrt(self.x21 * s)

    def get_R_I(self, x):
        return i1e(x) / i0e(x)

    def get_R_K(self, x):
        return k1e(x) / k0e(x)

    def KI_stable(self, s):
        """
        Численно стабильное вычисление KI(s).

        Исходная формула:
            KI = [I₁(a)·K₀(b) + coeff·I₀(a)·K₁(b)] /
                 [K₁(a)·K₀(b) - coeff·K₀(a)·K₁(b)]
        где coeff = √x21 / M21.

        Разделив числитель и знаменатель на I₀(a)·K₀(b), получаем
        (num, den) такие, что KI = num / den, без переполнения.
        """
        a, b = self.a(s), self.b(s)
        coeff = np.sqrt(self.x21) / self.M21

        r_ia = self.get_R_I(a)
        r_kb = self.get_R_K(b)

        k0_i0_a = (k0e(a) / i0e(a)) * np.exp(-2 * a)   # K₀(a)/I₀(a), стабильно

        num = r_ia + coeff * r_kb
        den = (self.get_R_K(a) - coeff * r_kb) * k0_i0_a

        return num, den

    def P_wD_laplace(self, u, r_D=1.0):
        """
        Безразмерное давление в пространстве Лапласа.

        При a_val > 600 (float64 нестабилен) — асимптотика чистой зоны 1:
            P_wD = K₀(rD·√s) / (s·√s·K₁(√s))
        """
        s = u
        sqrt_s = np.sqrt(s)
        a_val = self.a(s)

        num_ki, den_ki = self.KI_stable(s)

        if a_val > 600:
            return k0e(r_D * sqrt_s) / (s * sqrt_s * k1e(sqrt_s))

        r_ia = self.get_R_I(sqrt_s)
        r_ka = self.get_R_K(sqrt_s)
        k0_i0_s = (k0e(sqrt_s) / i0e(sqrt_s)) * np.exp(-2 * sqrt_s)

        # Подстановка KI = num_ki/den_ki в формулу
        # P_wD = (KI·K₀(√s) + I₀(√s)) / (s·√s·(KI·K₁(√s) - I₁(√s)))
        # делим числитель и знаменатель на I₀(√s), den_ki сокращается
        numerator = num_ki * k0_i0_s + den_ki
        denominator = s * sqrt_s * (num_ki * r_ka * k0_i0_s - den_ki * r_ia)

        return numerator / denominator

    def F(self, s):
        """Фильтр Агарвала: учёт скин-фактора и влияния ствола скважины."""
        return self.agarwal_filter(self.P_wD_laplace, s, 1.0, self.S, self.C_D)


class InfiniteRadialCompositeReservoirModelV2(InfiniteRadialCompositeReservoirModel):
    """
    Радиально-композитная модель с бесконечной границей.
    Принимает уже вычисленные отношения M12 = M1/M2, omega12 = omega1/omega2
    вместо отдельных параметров зон.
    """

    def __init__(self, C_D, S, M12, omega12, r_fD):
        self.C_D = C_D
        self.S = S
        self.r_fD = r_fD

        self.M12 = M12
        self.omega12 = omega12
        self.M21 = 1.0 / M12
        self.x21 = M12 / omega12


if __name__ == "__main__":

    params = pd.read_csv('./datasets/dataset04/params/26.csv')

    model = InfiniteRadialCompositeReservoirModel(
        C_D=params['C_D'][0],
        S=params['S'][0],
        M1=params['M1'][0],
        M2=params['M2'][0],
        omega1=params['omega1'][0],
        omega2=params['omega2'][0],
        r_fD=params['r_fD'][0],
    )

    t_D_array = np.logspace(0, 10, 3000)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-9) \
        .derivative(smoothig_alg='regression', delta=0.3) \
        .visualize('Кривые давления для радиально-композитной модели с бесконечной границей', False)
