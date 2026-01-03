import matplotlib
import numpy as np

from inversion.shtefest_algorithm import ShtefestAlgorithm
from model.dualporosity.infinite_dual_porosity_model import InfiniteDualPorosityReservoirModel

matplotlib.use('TkAgg')

if __name__ == "__main__":
    model = InfiniteDualPorosityReservoirModel(C_D=20, S=2, omega=0.1, lam=7e-6)

    t_D_array = np.logspace(0, 7, 1000)
    alg = ShtefestAlgorithm(N=16)

    model.pressure(t_D_array, alg) \
        .gauss_noize(mu=0, sigma=5e-8) \
        .derivative(smoothig_alg='regression', delta=0.5) \
        .visualize('Кривые давления (модель двойной пористости)')