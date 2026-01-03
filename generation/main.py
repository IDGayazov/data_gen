from generation.dual_permeability import InfiniteDualPermeabilityModelGenerator, FiniteDualPermeabilityGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from generation.radial_composite import InfiniteRadialCompositeGenerator


def main():
    generator1 = InfiniteHomogeneousGenerator(30, 500, 2)
    generator2 = FiniteHomogeneousGenerator(30, 500, 2)
    generator3 = InfiniteDualPorosityGenerator(30, 500, 2)
    generator4 = FiniteDualPorosityGenerator(30, 500, 2)
    generator5 = InfiniteDualPermeabilityModelGenerator(30, 100, 2)
    generator6 = FiniteDualPermeabilityGenerator(30, 100, 2)
    generator7 = InfiniteRadialCompositeGenerator(30, 500, 2)

    # generator1.generate()
    # generator2.generate()
    # generator3.generate()
    # generator4.generate()
    # generator5.generate()
    # generator6.generate()
    generator7.generate()


if __name__ == "__main__":
    main()