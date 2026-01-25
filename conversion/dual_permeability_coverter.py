import numpy as np


class DualPermeabilityDimensionConverter:
    '''
    Преобразование данных из размерных в безразмерные единицы
    для модели двойной проницаемости (dual permeability)

    Атрибуты:
       - k1 - проницаемость системы 1 (обычно трещины), м^2
       - k2 - проницаемость системы 2 (обычно матрица), м^2
       - phi1 - пористость системы 1, безразмерно
       - phi2 - пористость системы 2, безразмерно
       - c_t1 - сжимаемость системы 1, 1 / Па
       - c_t2 - сжимаемость системы 2, 1 / Па
       - h1 - толщина пласта системы 1, м
       - h2 - толщина пласта системы 2, м
       - q1 - дебит из системы 1, м^3 / сут
       - q2 - дебит из системы 2, м^3 / сут
       - mu - вязкость флюида, Па * с (одинакова для обеих систем)
       - B - объемный коэффициент, безразмерно
       - p_i - начальное давление, Па
       - r_w - радиус скважины, м
       - alpha - фактор формы, безразмерно
       - S1 - скин-фактор системы 1
       - S2 - скин-фактор системы 2
    '''

    def __init__(self, k1, k2, phi1, phi2, c_t1, c_t2, h1, h2,
                 q1, q2, mu, B, p_i, r_w, alpha, S1=0.0, S2=0.0):
        # Параметры системы 1 (обычно трещины/высокопроницаемая)
        self.k1 = k1
        self.phi1 = phi1
        self.c_t1 = c_t1
        self.q1 = self.convert_debit_on_m3_by_seconds(q1)

        # Параметры системы 2 (обычно матрица/низкопроницаемая)
        self.k2 = k2
        self.phi2 = phi2
        self.c_t2 = c_t2
        self.q2 = self.convert_debit_on_m3_by_seconds(q2)

        # Общие параметры
        self.h1 = h1
        self.h2 = h2
        self.mu = mu
        self.B = B
        self.p_i = p_i
        self.r_w = r_w

        # Фактор формы
        self.alpha = alpha  

        self.S1 = S1  # скин системы 1
        self.S2 = S2  # скин системы 2

        # Вычисление производных параметров
        self.total_q = q1 + q2
        self.q1_frac = q1 / self.total_q if self.total_q > 0 else 0.5
        self.q2_frac = q2 / self.total_q if self.total_q > 0 else 0.5

        self.storage1 = phi1 * c_t1 * h1
        self.storage2 = phi2 * c_t2 * h2
        self.total_storage = self.storage1 + self.storage2

        self.phi_total = phi1 + phi2

        # Вычисление безразмерных параметров
        self.omega1 = self.calc_omega1()
        self.omega2 = self.calc_omega2()
        self.lambda_param = self.calc_lambda()
        self.kappa = self.calc_kappa()

    def calc_omega1(self):
        """Коэффициент ёмкости системы 1"""
        return self.storage1 / self.total_storage if self.total_storage > 0 else 0.5

    def calc_omega2(self):
        """Коэффициент ёмкости системы 2"""
        return self.storage2 / self.total_storage if self.total_storage > 0 else 0.5

    def calc_omega(self):
        return self.phi1 * self.c_t1 * self.h1 / self.total_storage if self.total_storage > 0 else 0.5

    def calc_lambda(self):
        """Безразмерный коэффициент перетока между системами"""
        return self.alpha * self.r_w**2 * (self.k2 * self.h2) / (self.k1 * self.h1 + self.k2 * self.h2)

    def calc_kappa(self):
        """Отношение проницаемостей"""
        return self.k1 * self.h1 / (self.k1 * self.h1 + self.k2 * self.h2)

    def pressure_from_dim_to_dimless(self, p, system=1):
        """
        Преобразование давления в безразмерное
        system: 1 - для системы 1, 2 - для системы 2
        """
        if system == 1:
            k = self.k1
            h = self.h1
        elif system == 2:
            k = self.k2
            h = self.h2
        else:
            raise ValueError("system must be 1 or 2")

        return 2 * np.pi * k * h * (self.p_i - p) / (self.mu * self.B * self.total_q)

    def pressure_from_dimless_to_dim(self, p_D, system=1):
        """
        Преобразование безразмерного давления в размерное
        system: 1 - для системы 1, 2 - для системы 2
        """
        if system == 1:
            k = self.k1
            h = self.h1
        elif system == 2:
            k = self.k2
            h = self.h2
        else:
            raise ValueError("system must be 1 or 2")

        return self.p_i - self.mu * self.B * self.total_q * p_D / (2 * np.pi * k * h)

    def time_from_dim_to_dimless(self, t, system='total'):
        """
        Преобразование времени в безразмерное
        system: 'total' - общая ёмкость, '1' - система 1, '2' - система 2
        """
        if system == 'total':
            phi_c_t = self.total_storage
            k = self.k1 * self.h1 + self.k2 * self.h2
        elif system == '1':
            phi_c_t = self.storage1
            k = self.k1
        elif system == '2':
            phi_c_t = self.storage2
            k = self.k2
        else:
            raise ValueError("system must be 'total', '1', or '2'")

        return k * t / (self.mu * phi_c_t * self.r_w ** 2)

    def time_from_dimless_to_dim(self, t_D, system='total'):
        """
        Преобразование безразмерного времени в размерное
        system: 'total' - общая ёмкость, '1' - система 1, '2' - система 2
        """
        if system == 'total':
            phi_c_t = self.total_storage
            k = self.k1
        elif system == '1':
            phi_c_t = self.storage1
            k = self.k1
        elif system == '2':
            phi_c_t = self.storage2
            k = self.k2
        else:
            raise ValueError("system must be 'total', '1', or '2'")

        return t_D * self.mu * phi_c_t * self.r_w ** 2 / k

    def wellbore_storage_from_dim_to_dimless(self, C, system='total'):
        """
        Преобразование коэффициента влияния ствола скважины в безразмерное
        """
        return C / (2 * np.pi * self.total_storage * self.r_w ** 2)

    def wellbore_storage_from_dimless_to_dim(self, C_D, system='total'):
        """
        Преобразование безразмерного коэффициента в размерное
        system: 'total' - общая ёмкость, '1' - система 1, '2' - система 2
        """
        if system == 'total':
            phi_c_t = self.total_storage
        elif system == '1':
            phi_c_t = self.storage1
        elif system == '2':
            phi_c_t = self.storage2
        else:
            raise ValueError("system must be 'total', '1', or '2'")

        return C_D * 2 * np.pi * phi_c_t * self.r_w ** 2

    def radius_from_dim_to_dimless(self, r):
        """Преобразование радиуса в безразмерное"""
        return r / self.r_w

    def radius_from_dimless_to_dim(self, r_D):
        """Преобразование безразмерного радиуса в размерное"""
        return r_D * self.r_w

    def diffusivity_system1(self):
        """Коэффициент пьезопроводности системы 1"""
        return self.k1 / (self.phi1 * self.mu * self.c_t1)

    def diffusivity_system2(self):
        """Коэффициент пьезопроводности системы 2"""
        return self.k2 / (self.phi2 * self.mu * self.c_t2)

    def diffusivity_ratio(self):
        """Отношение пьезопроводностей (система 2 / система 1)"""
        eta1 = self.diffusivity_system1()
        eta2 = self.diffusivity_system2()
        return eta2 / eta1 if eta1 > 0 else 0.0

    def convert_debit_on_m3_by_seconds(self, q):
        """
        Перевод из м^3 / сут в м^3 / с
        """
        return q / 86400