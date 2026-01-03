import random

from generation.dual_permeability import FiniteDualPermeabilityGenerator, InfiniteDualPermeabilityModelGenerator
from generation.dual_porosity import InfiniteDualPorosityGenerator, FiniteDualPorosityGenerator
from generation.homogeneous import InfiniteHomogeneousGenerator, FiniteHomogeneousGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed

from generation.radial_composite import InfiniteRadialCompositeGenerator


def run_generator(generator):
    """Функция для запуска генератора"""
    print(f"Запускается {generator.__class__.__name__}")
    generator.generate()
    return generator.__class__.__name__


def main():
    rng = random.Random()

    size1 = rng.randint(1000, 2000)
    size2 = rng.randint(1000, 2000)
    size3 = rng.randint(1000, 2000)
    size4 = rng.randint(1000, 2000)
    size5 = rng.randint(1000, 2000)
    size6 = rng.randint(1000, 2000)
    size7 = rng.randint(1000, 2000)

    print("InfiniteHomogeneousGenerator size:", size1)
    print("FiniteHomogeneousGenerator size:", size2)
    print("InfiniteDualPorosityGenerator size:", size3)
    print("FiniteDualPorosityGenerator size:", size4)
    print("InfiniteDualPermeabilityModelGenerator size:", size5)
    print("FiniteDualPermeabilityGenerator size:", size6)
    print("InfiniteRadialCompositeGenerator size:", size7)

    generators = [
        InfiniteHomogeneousGenerator(30, 128, 3),
        FiniteHomogeneousGenerator(30, 128, 3),
        InfiniteDualPorosityGenerator(30, 128, 3),
        FiniteDualPorosityGenerator(30, 128, 3),
        InfiniteDualPermeabilityModelGenerator(30, 128, 3),
        FiniteDualPermeabilityGenerator(30, 128, 3),
        InfiniteRadialCompositeGenerator(30, 128, 3)
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_generator, gen): gen.__class__.__name__
            for gen in generators
        }

        for future in as_completed(futures):
            generator_name = futures[future]
            try:
                result = future.result()
                print(f"Генератор {result} успешно завершил работу")
            except Exception as e:
                print(f"Ошибка в генераторе {generator_name}: {e}")


if __name__ == "__main__":
    main()