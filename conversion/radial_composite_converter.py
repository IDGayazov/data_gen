import numpy as np


class RadialCompositeDualPorosityConverter:
    '''
    Преобразование данных для радиально-композитной модели
    с двойной пористостью в обеих зонах

    В этой модели:
    1. Две зоны: внутренняя (1) и внешняя (2)
    2. Каждая зона имеет двойную пористость (трещины + матрица)
    3. Проницаемости трещин: k_f1 ≠ k_f2
    4. Проницаемости матрицы: k_m1 ≠ k_m2

    Атрибуты:
       # Общие параметры
       - h - толщина пласта, м
       - q - дебит скважины, м^3 / с
       - mu - вязкость флюида, Па * с
       - B - объемный коэффициент, безразмерно
       - p_i - начальное давление, Па

       # Зона 1 (внутренняя)
       - k_f1 - проницаемость трещин зоны 1, м^2
       - k_m1 - проницаемость матрицы зоны 1, м^2
       - phi_f1 - пористость трещин зоны 1
       - phi_m1 - пористость матрицы зоны 1
       - c_tf1 - сжимаемость трещин зоны 1, 1 / Па
       - c_tm1 - сжимаемость матрицы зоны 1, 1 / Па
       - alpha1 - геометрический фактор зоны 1, 1/м^2

       # Зона 2 (внешняя)
       - k_f2 - проницаемость трещин зоны 2, м^2
       - k_m2 - проницаемость матрицы зоны 2, м^2
       - phi_f2 - пористость трещин зоны 2
       - phi_m2 - пористость матрицы зоны 2
       - c_tf2 - сжимаемость трещин зоны 2, 1 / Па
       - c_tm2 - сжимаемость матрицы зоны 2, 1 / Па
       - alpha2 - геометрический фактор зоны 2, 1/м^2

       # Геометрические параметры
       - r_w - радиус скважины, м
       - R_i - радиус интерфейса между зонами, м
       - r_e - внешний радиус, м
    '''

    def __init__(self, h, q, mu, B, p_i,
                 # Зона 1
                 k_f1, k_m1, phi_f1, phi_m1, c_tf1, c_tm1, alpha1,
                 # Зона 2
                 k_f2, k_m2, phi_f2, phi_m2, c_tf2, c_tm2, alpha2,
                 # Геометрия
                 r_w, R_i, r_e=None):

        # Общие параметры
        self.h = h
        self.q = q
        self.mu = mu
        self.B = B
        self.p_i = p_i
        self.r_w = r_w
        self.R_i = R_i
        self.r_e = r_e if r_e is not None else 1000 * R_i  # по умолчанию

        # Зона 1 (внутренняя)
        self.k_f1 = k_f1
        self.k_m1 = k_m1
        self.phi_f1 = phi_f1
        self.phi_m1 = phi_m1
        self.c_tf1 = c_tf1
        self.c_tm1 = c_tm1
        self.alpha1 = alpha1

        # Зона 2 (внешняя)
        self.k_f2 = k_f2
        self.k_m2 = k_m2
        self.phi_f2 = phi_f2
        self.phi_m2 = phi_m2
        self.c_tf2 = c_tf2
        self.c_tm2 = c_tm2
        self.alpha2 = alpha2

        # Вычисление производных параметров
        self.calc_derived_parameters()

    def calc_derived_parameters(self):
        """Вычисление всех производных и безразмерных параметров"""

        # 1. Расчет коэффициентов ёмкости (ω)
        storage_f1 = self.phi_f1 * self.c_tf1
        storage_m1 = self.phi_m1 * self.c_tm1
        storage_total1 = storage_f1 + storage_m1
        self.omega1 = storage_f1 / storage_total1 if storage_total1 > 0 else 0.0

        storage_f2 = self.phi_f2 * self.c_tf2
        storage_m2 = self.phi_m2 * self.c_tm2
        storage_total2 = storage_f2 + storage_m2
        self.omega2 = storage_f2 / storage_total2 if storage_total2 > 0 else 0.0

        # 2. Расчет коэффициентов перетока (λ)
        self.lambda1 = self.alpha1 * self.k_m1 / self.k_f1 * self.r_w ** 2
        self.lambda2 = self.alpha2 * self.k_m2 / self.k_f2 * self.r_w ** 2

        # 3. Отношения M (k_m/k_f) для каждой зоны
        self.M1 = self.k_m1 / self.k_f1 if self.k_f1 > 0 else 0.0
        self.M2 = self.k_m2 / self.k_f2 if self.k_f2 > 0 else 0.0

        # 4. Безразмерный радиус интерфейса
        self.r_fD = self.R_i / self.r_w

        # 5. κ (отношение проницаемостей) для каждой зоны
        self.kappa1 = self.M1  # k_m1/k_f1
        self.kappa2 = self.M2  # k_m2/k_f2

        # 6. Эффективные параметры для каждой зоны
        self.phi_eff1 = self.phi_f1 + self.phi_m1
        self.c_t_eff1 = storage_total1 / self.phi_eff1 if self.phi_eff1 > 0 else self.c_tf1

        self.phi_eff2 = self.phi_f2 + self.phi_m2
        self.c_t_eff2 = storage_total2 / self.phi_eff2 if self.phi_eff2 > 0 else self.c_tf2

        # 7. Отношение емкостей между зонами (D)
        self.D_comp = storage_total2 / storage_total1 if storage_total1 > 0 else 1.0

        # 8. Отношение подвижностей между зонами (M_comp)
        self.M_comp = (self.k_f2 / self.mu) / (self.k_f1 / self.mu)  # (k_f2/μ) / (k_f1/μ)

        # 9. Отношение общих подвижностей (с учетом матрицы)
        mobility1 = self.k_f1 / self.mu + self.k_m1 / self.mu
        mobility2 = self.k_f2 / self.mu + self.k_m2 / self.mu
        self.M_total_comp = mobility2 / mobility1 if mobility1 > 0 else 1.0

        # 10. Суммарные параметры (взвешенные по объему)
        volume1 = np.pi * self.R_i ** 2 * self.h
        volume2 = np.pi * (self.r_e ** 2 - self.R_i ** 2) * self.h

        # Средневзвешенная проницаемость трещин
        self.k_f_avg = (self.k_f1 * volume1 + self.k_f2 * volume2) / (volume1 + volume2) if (
                                                                                                        volume1 + volume2) > 0 else self.k_f1

        # Суммарные емкости
        self.total_storage = storage_total1 + storage_total2
        self.total_phi = self.phi_eff1 + self.phi_eff2
        self.avg_c_t = self.total_storage / self.total_phi if self.total_phi > 0 else self.c_tf1

    def pressure_from_dim_to_dimless(self, p, zone=1, use_fracture=True):
        """
        Преобразование давления в безразмерное
        zone: 1 - внутренняя зона, 2 - внешняя зона
        use_fracture: если True - используем k_f, если False - используем k_m
        """
        if zone == 1:
            k = self.k_f1 if use_fracture else self.k_m1
        else:  # zone == 2
            k = self.k_f2 if use_fracture else self.k_m2

        return 2 * np.pi * k * self.h * (self.p_i - p) / (self.mu * self.B * self.q)

    def pressure_from_dimless_to_dim(self, p_D, zone=1, use_fracture=True):
        """
        Преобразование безразмерного давления в размерное
        """
        if zone == 1:
            k = self.k_f1 if use_fracture else self.k_m1
        else:  # zone == 2
            k = self.k_f2 if use_fracture else self.k_m2

        return self.p_i - self.mu * self.B * self.q * p_D / (2 * np.pi * k * self.h)

    def time_from_dim_to_dimless(self, t, zone=1, system='fracture'):
        """
        Преобразование времени в безразмерное
        zone: 1 - внутренняя зона, 2 - внешняя зона
        system: 'fracture' - трещинная система, 'matrix' - матричная, 'effective' - эффективная
        """
        if zone == 1:
            if system == 'fracture':
                phi_c_t = self.phi_f1 * self.c_tf1
                k = self.k_f1
            elif system == 'matrix':
                phi_c_t = self.phi_m1 * self.c_tm1
                k = self.k_m1
            else:  # 'effective'
                phi_c_t = self.phi_eff1 * self.c_t_eff1
                k = self.k_f1  # для времени обычно используем k_f
        else:  # zone == 2
            if system == 'fracture':
                phi_c_t = self.phi_f2 * self.c_tf2
                k = self.k_f2
            elif system == 'matrix':
                phi_c_t = self.phi_m2 * self.c_tm2
                k = self.k_m2
            else:  # 'effective'
                phi_c_t = self.phi_eff2 * self.c_t_eff2
                k = self.k_f2

        return k * t / (self.mu * phi_c_t * self.r_w ** 2)

    def time_from_dimless_to_dim(self, t_D, zone=1, system='fracture'):
        """
        Преобразование безразмерного времени в размерное
        """
        if zone == 1:
            if system == 'fracture':
                phi_c_t = self.phi_f1 * self.c_tf1
                k = self.k_f1
            elif system == 'matrix':
                phi_c_t = self.phi_m1 * self.c_tm1
                k = self.k_m1
            else:  # 'effective'
                phi_c_t = self.phi_eff1 * self.c_t_eff1
                k = self.k_f1
        else:  # zone == 2
            if system == 'fracture':
                phi_c_t = self.phi_f2 * self.c_tf2
                k = self.k_f2
            elif system == 'matrix':
                phi_c_t = self.phi_m2 * self.c_tm2
                k = self.k_m2
            else:  # 'effective'
                phi_c_t = self.phi_eff2 * self.c_t_eff2
                k = self.k_f2

        return t_D * self.mu * phi_c_t * self.r_w ** 2 / k

    def wellbore_storage_from_dim_to_dimless(self, C, zone=1):
        """
        Преобразование коэффициента влияния ствола скважины в безразмерное
        zone: 1 - используем параметры внутренней зоны
        """
        if zone == 1:
            phi_c_t = self.phi_eff1 * self.c_t_eff1
        else:
            phi_c_t = self.phi_eff2 * self.c_t_eff2

        return C / (2 * np.pi * self.h * phi_c_t * self.r_w ** 2)

    def wellbore_storage_from_dimless_to_dim(self, C_D, zone=1):
        """
        Преобразование безразмерного коэффициента в размерное
        """
        if zone == 1:
            phi_c_t = self.phi_eff1 * self.c_t_eff1
        else:
            phi_c_t = self.phi_eff2 * self.c_t_eff2

        return C_D * 2 * np.pi * self.h * phi_c_t * self.r_w ** 2

    def radius_from_dim_to_dimless(self, r):
        """Преобразование радиуса в безразмерное"""
        return r / self.r_w

    def radius_from_dimless_to_dim(self, r_D):
        """Преобразование безразмерного радиуса в размерное"""
        return r_D * self.r_w


# Пример использования
if __name__ == "__main__":
    # Пример параметров с РАЗНЫМИ проницаемостями
    converter = RadialCompositeDualPorosityConverter(
        # Общие параметры
        h=10,
        q=2.31e-3,
        mu=1e-3,
        B=1.2,
        p_i=25e6,

        # Зона 1 (внутренняя) - например, зона повреждения
        k_f1=1e-13,  # низкая проницаемость трещин
        k_m1=1e-15,  # низкая проницаемость матрицы
        phi_f1=0.02,
        phi_m1=0.18,
        c_tf1=1e-9,
        c_tm1=1.5e-9,
        alpha1=12.0,

        # Зона 2 (внешняя) - например, нетронутый коллектор
        k_f2=5e-13,  # высокая проницаемость трещин
        k_m2=5e-15,  # высокая проницаемость матрицы
        phi_f2=0.015,
        phi_m2=0.15,
        c_tf2=1e-9,
        c_tm2=1.5e-9,
        alpha2=12.0,

        # Геометрия
        r_w=0.1,
        R_i=50.0,
        r_e=1000
    )