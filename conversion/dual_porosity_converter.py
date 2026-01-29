import numpy as np


class DualPorosityDimensionConverter:
    '''
    Преобразование данных из размерных в безразмерные единицы
    для модели двойной проницаемости

    Атрибуты:
       - k_f - проницаемость трещин, м^2
       - k_m - проницаемость матрицы, м^2
       - h - толщина пласта, м
       - q - дебит скважины, м^3 / сут
       - mu - вязкость флюида, Па * с
       - B - объемный коэффициент, безразмерно
       - p_i - начальное давление, Па
       - phi_f - пористость трещин, безразмерно
       - phi_m - пористость матрицы, безразмерно
       - c_tf - сжимаемость трещин, 1 / Па
       - c_tm - сжимаемость матрицы, 1 / Па
       - r_w - радиус скважины, м
       - alpha - геометрический фактор, 1/м^2 (shape factor)
    '''

    def __init__(self, k_f, k_m, h, q, mu, B, p_i, phi_f, phi_m, c_tf, c_tm, r_w, alpha=12.0):
        # Параметры трещин
        self.k_f = k_f
        self.phi_f = phi_f
        self.c_tf = c_tf

        # Параметры матрицы
        self.k_m = k_m
        self.phi_m = phi_m
        self.c_tm = c_tm

        # Общие параметры
        self.h = h
        self.q = self.convert_debit_on_m3_by_seconds(q)
        self.mu = mu
        self.B = B
        self.p_i = p_i
        self.r_w = r_w
        self.alpha = alpha

        # Вычисление производных параметров
        self.total_storage = phi_f * c_tf + phi_m * c_tm

        # Безразмерные параметры двойной пористости
        self.omega = self.calc_omega()  # коэффициент ёмкости
        self.lambda_param = self.calc_lambda()  # коэффициент перетока
        self.kappa = self.calc_kappa()  # отношение проницаемостей

    def calc_omega(self):
        """Вычисление коэффициента ёмкости (storativity ratio)"""
        storage_f = self.phi_f * self.c_tf
        storage_m = self.phi_m * self.c_tm
        total_storage = storage_f + storage_m

        return storage_f / total_storage if total_storage > 0 else 0.0

    def calc_lambda(self):
        """Вычисление коэффициента перетока (interporosity flow coefficient)"""
        return self.alpha * (self.k_m / self.k_f) * self.r_w ** 2

    def calc_kappa(self):
        """Вычисление отношения проницаемостей"""
        return self.k_m / self.k_f

    def pressure_from_dim_to_dimless(self, p, use_fracture=True):
        """
        Преобразование давления в безразмерное
        use_fracture: если True - используем k_f, если False - используем k_m
        """
        k = self.k_f if use_fracture else self.k_m
        return 2 * np.pi * k * self.h * (self.p_i - p) / (self.q * self.mu * self.B)

    def pressure_from_dimless_to_dim(self, p_D, use_fracture=True):
        """
        Преобразование безразмерного давления в размерное
        use_fracture: если True - используем k_f, если False - используем k_m
        """
        k = self.k_f if use_fracture else self.k_m
        return self.p_i - self.q * self.mu * self.B * p_D / (2 * np.pi * k * self.h)

    def time_from_dim_to_dimless(self, t, use_total_storage=True):
        """
        Преобразование времени в безразмерное
        use_total_storage: если True - используем общую ёмкость, если False - только трещины
        """
        if use_total_storage:
            # Используем общую ёмкость (трещины + матрица)
            phi_c_t = self.total_storage
            k = self.k_f 
        else:
            # Используем только ёмкость трещин (раннее время)
            phi_c_t = self.phi_f * self.c_tf
            k = self.k_f

        return k * t / (self.mu * phi_c_t * self.r_w ** 2)

    def time_from_dimless_to_dim(self, t_D, use_total_storage=True):
        """
        Преобразование безразмерного времени в размерное
        use_total_storage: если True - используем общую ёмкость, если False - только трещины
        """
        if use_total_storage:
            phi_c_t = self.total_storage
            k = self.k_f
        else:
            phi_c_t = self.phi_f * self.c_tf
            k = self.k_f

        return t_D * self.mu * phi_c_t * self.r_w ** 2 / k

    def wellbore_storage_from_dim_to_dimless(self, C, use_total_storage=True):
        """
        Преобразование коэффициента влияния ствола скважины в безразмерное
        use_total_storage: если True - используем общую ёмкость, если False - только трещины
        """
        if use_total_storage:
            phi_c_t = self.total_storage
        else:
            phi_c_t = self.phi_f * self.c_tf

        return C / (2 * np.pi * self.h * phi_c_t * self.r_w ** 2)

    def wellbore_storage_from_dimless_to_dim(self, C_D, use_total_storage=True):
        """
        Преобразование безразмерного коэффициента влияния ствола скважины в размерное
        """
        if use_total_storage:
            phi_c_t = self.total_storage
        else:
            phi_c_t = self.phi_f * self.c_tf

        return C_D * 2 * np.pi * self.h * phi_c_t * self.r_w ** 2

    def radius_from_dim_to_dimless(self, r):
        """Преобразование радиуса в безразмерное"""
        return r / self.r_w

    def radius_from_dimless_to_dim(self, r_D):
        """Преобразование безразмерного радиуса в размерное"""
        return r_D * self.r_w

    def convert_debit_on_m3_by_seconds(self, q):
        """
        Перевод из м^3 / сут в м^3 / с
        """
        return q / 86400