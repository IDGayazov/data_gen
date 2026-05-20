import numpy as np

class HomogeneousConverter:
    '''
    Преобразование данных из размерных в безразмерные единицы и наоборот

    Атрибуты:
       - k - проницаемость, м^2
       - h - толщина пласта, м
       - q - дебит скважины, м^3 / сут
       - mu - вязкость флюида, Па * с
       - B - объемный коэффициент, безразмерно
       - p_i - начальное давление, Па
       - p - текущее давление, Па
       - phi - пористость, безразмерно
       - c_t - общая сжимаемость, 1 / Па
       - r_w - радиус скважины, м
    '''
    def __init__(self, k, h, q, mu, B, p_i, phi, c_t, r_w):
        self.k = k
        self.h = h
        self.q = self.convert_debit_on_m3_by_seconds(q)
        self.mu = mu
        self.B = B
        self.p_i = p_i
        self.phi = phi
        self.c_t = c_t
        self.r_w = r_w

    def pressure_from_dim_to_dimless(self, p):
        return 2 * np.pi * self.k * self.h * (self.p_i - p) / (self.q * self.mu * self.B)

    def pressure_from_dimless_to_dim(self, p_D):
        return self.p_i - self.q * self.mu * self.B * p_D / (2 * np.pi * self.k * self.h)

    def time_from_dim_to_dimless(self, t):
        return self.k * t / (self.phi * self.mu * self.c_t * self.r_w**2)

    def time_from_dimless_to_dim(self, t_D):
        return (t_D * self.phi * self.mu * self.c_t * self.r_w**2) / self.k

    def wellbore_storage_from_dim_to_dimless(self, C):
        return C / (2 * np.pi * self.h * self.phi * self.c_t * self.r_w ** 2)

    def wellbore_storage_from_dimless_to_dim(self, C_D):
        return C_D * 2 * np.pi * self.h * self.phi * self.c_t * self.r_w ** 2

    def reservoir_radius_from_dim_to_dimless(self, r_e):
        return r_e / self.r_w

    def reservoir_radius_from_dimless_to_dim(self, R_eD):
        return R_eD * self.r_w

    def convert_debit_on_m3_by_seconds(self, q):
        """
        Перевод из м^3 / сут в м^3 / с
        """
        return q / 86400