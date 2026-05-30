import numpy as np


class RadialCompositeConverter:
    """
    Конвертер для простой радиально-композитной модели
    (две зоны с разными k, phi, c_t)
    """

    def __init__(self, h, q, mu, B, p_i, r_w,
                 k1, phi1, c_t1,
                 k2, phi2, c_t2,
                 R_i, r_e=None):

        # Общие параметры
        self.h = h
        self.q = self.convert_debit_on_m3_by_seconds(q)
        self.mu = mu
        self.B = B
        self.p_i = p_i
        self.r_w = r_w
        self.R_i = R_i

        # Зона 1 (внутренняя)
        self.k1 = k1
        self.phi1 = phi1
        self.c_t1 = c_t1

        # Зона 2 (внешняя)
        self.k2 = k2
        self.phi2 = phi2
        self.c_t2 = c_t2

    @property
    def storage1(self):
        """Емкость зоны 1"""
        return self.phi1 * self.c_t1

    @property
    def storage2(self):
        """Емкость зоны 2"""
        return self.phi2 * self.c_t2

    @property
    def M1(self):
        return self.k1 * self.h / self.mu

    @property
    def M2(self):
        return self.k2 * self.h / self.mu

    @property
    def r_fD(self):
        """Безразмерный радиус интерфейса"""
        return self.R_i / self.r_w

    # ============ МЕТОДЫ ПРЕОБРАЗОВАНИЯ ДАВЛЕНИЯ ============

    def pressure_from_dim_to_dimless(self, p, use_zone=2):
        if use_zone == 1:
            k = self.k1
        elif use_zone == 2:
            k = self.k2
        else:
            raise ValueError("use_zone должно быть 1 или 2")

        return 2 * np.pi * k * self.h * (self.p_i - p) / (self.q * self.mu * self.B)

    def pressure_from_dimless_to_dim(self, p_D, use_zone=2):
        if use_zone == 1:
            k = self.k1
        elif use_zone == 2:
            k = self.k2
        else:
            raise ValueError("use_zone должно быть 1 или 2")

        return self.p_i - self.q * self.mu * self.B * p_D / (2 * np.pi * k * self.h)

    # ============ МЕТОДЫ ПРЕОБРАЗОВАНИЯ ВРЕМЕНИ ============

    def time_from_dim_to_dimless(self, t, use_zone=2):
        if use_zone == 1:
            k = self.k1
            phi_c_t = self.storage1
        elif use_zone == 2:
            k = self.k2
            phi_c_t = self.storage2
        else:
            raise ValueError("use_zone должно быть 1 или 2")

        return k * t / (self.mu * phi_c_t * self.r_w ** 2)

    def time_from_dimless_to_dim(self, t_D, use_zone=2):
        if use_zone == 1:
            k = self.k1
            phi_c_t = self.storage1
        elif use_zone == 2:
            k = self.k2
            phi_c_t = self.storage2
        else:
            raise ValueError("use_zone должно быть 1 или 2")

        return t_D * self.mu * phi_c_t * self.r_w ** 2 / k

    # ============ МЕТОДЫ ПРЕОБРАЗОВАНИЯ КОЭФФИЦИЕНТА ВЛИЯНИЯ СТВОЛА ============

    def wellbore_storage_from_dim_to_dimless(self, C, use_zone=2):
        if use_zone == 1:
            phi_c_t = self.storage1
        elif use_zone == 2:
            phi_c_t = self.storage2
        else:
            raise ValueError("use_zone должно быть 1 или 2")

        return C / (2 * np.pi * self.h * phi_c_t * self.r_w ** 2)

    def wellbore_storage_from_dimless_to_dim(self, C_D, use_zone=2):
        if use_zone == 1:
            phi_c_t = self.storage1
        elif use_zone == 2:
            phi_c_t = self.storage2
        else:
            raise ValueError("use_zone должно быть 1 или 2")

        return C_D * 2 * np.pi * self.h * phi_c_t * self.r_w ** 2

    # ============ МЕТОДЫ ПРЕОБРАЗОВАНИЯ РАДИУСА ============

    def radius_from_dim_to_dimless(self, r):
        return r / self.r_w

    def radius_from_dimless_to_dim(self, r_D):
        return r_D * self.r_w

    def convert_debit_on_m3_by_seconds(self, q):
        """
        Перевод из м^3 / сут в м^3 / с
        """
        return q / 86400


