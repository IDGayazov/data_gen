import numpy as np

from conversion.homogeneous_converter import HomogeneousConverter
from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.homogeneous.infinite_homogeneous_model import InfiniteHomogeneousReservoirModel


def main():
    k = 5e-13
    h = 10
    q = 2.31e-3
    mu = 1e-3
    B = 1.2
    p_i = 25e6
    phi = 0.18
    c_t = 1.5e-9
    r_w = 0.1

    converter = HomogeneousConverter(k, h, q, mu, B, p_i, phi, c_t, r_w)

    t_max_days = 30  # 30 суток
    t_max_seconds = t_max_days * 24 * 3600  # 30 дней в секундах

    t_D_max = converter.time_from_dim_to_dimless(t_max_seconds)

    print(f"Максимальное безразмерное время t_D_max: {t_D_max:.2e}")

    t_D_min = 1e-2
    t_D_array = np.logspace(np.log10(t_D_min), np.log10(t_D_max), 500)

    t_physical = (t_D_array * phi * mu * c_t * r_w ** 2) / k
    t_days = t_physical / (24 * 3600)  # в сутках

    model = InfiniteHomogeneousReservoirModel(C_D=100, S=1.0)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-7) \
        .derivative(smoothig_alg='regression', delta=0.3) \
        .visualize('Кривые давления (однородный пласт)')

if __name__ == "__main__":
    main()
