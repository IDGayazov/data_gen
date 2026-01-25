from typing import Final

from generation.dual_permeability import InfiniteDualPermeabilityModelGenerator, FiniteDualPermeabilityGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from generation.radial_composite import InfiniteRadialCompositeGenerator
from generation.utils import clear_folder


def main():
    T_MAX_DAYS: Final = 30
    POINTS_COUNT: Final = 128
    SIZE: Final = 5
    OUTPUT_PATH: Final = '../datasets/dataset5'

    clear_folder(OUTPUT_PATH)

    generators = [
        InfiniteHomogeneousGenerator(T_MAX_DAYS, POINTS_COUNT,  SIZE, OUTPUT_PATH),
        FiniteHomogeneousGenerator(T_MAX_DAYS, POINTS_COUNT, SIZE, OUTPUT_PATH),
        InfiniteDualPorosityGenerator(T_MAX_DAYS, POINTS_COUNT, SIZE, OUTPUT_PATH),
        FiniteDualPorosityGenerator(T_MAX_DAYS, POINTS_COUNT, SIZE, OUTPUT_PATH),
        InfiniteDualPermeabilityModelGenerator(T_MAX_DAYS, POINTS_COUNT, SIZE, OUTPUT_PATH),
        FiniteDualPermeabilityGenerator(T_MAX_DAYS, POINTS_COUNT, SIZE, OUTPUT_PATH),
        InfiniteRadialCompositeGenerator(T_MAX_DAYS, POINTS_COUNT, SIZE, OUTPUT_PATH)
    ]

    for generator in generators:
        generator.generate()

if __name__ == "__main__":
    main()