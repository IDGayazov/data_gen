import math

from inversion.shtefest_algorithm import ShtefestAlgorithm

alg = ShtefestAlgorithm(N=16)

def test1():
    def F(s):
        return 1 / (s + 1)

    t_val = 2.0
    f_t_approx = alg.inverse(F, t_val)
    print(f"Приближенное значение f({t_val}) = {f_t_approx}")
    print(f"Точное значение f({t_val}) = {math.exp(-t_val)}")
    print(f"Ошибка: {abs(math.exp(-t_val) - f_t_approx)}")


def test2():
    def F_exp(s):
        """Изображение для f(t) = e^{-at}"""
        a = 2.0  # Параметр
        return 1 / (s + a)

    # Оригинал: f(t) = e^{-2t}
    t_val = 1.0
    f_approx = alg.inverse(F_exp, t_val)
    exact_val = math.exp(-2 * t_val)
    print(f"e^(-2t) в t={t_val}: Прибл.={f_approx}, Точн.={exact_val}, Ошибка={abs(exact_val - f_approx)}")


def test3():
    def F_t_sin(s):
        """Изображение для f(t) = tsin(ωt)"""
        omega = 3.0
        return 2 * s * omega / (s ** 2 + omega ** 2) ** 2

    # Оригинал: f(t) = tsin(3t)
    t_val = 0.5
    f_approx = alg.inverse(F_t_sin, t_val)
    exact_val = t_val * math.sin(3 * t_val)
    print(f"sin(3t) в t={t_val}: Прибл.={f_approx}, Точн.={exact_val}, Ошибка={abs(exact_val - f_approx)}")


if __name__ == "__main__":
    test1()
    test2()
    test3()