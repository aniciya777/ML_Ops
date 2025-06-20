import os

from .utils import transport_files


def main() -> None:
    os.chdir('data')
    transport_files("input/ML/Relax", "output/ML/Relax")
    transport_files("input/ML/Catch", "output/ML/Catch")
    transport_files("input/ML/Gun", "output/ML/Gun")
    transport_files("input/ML/Index", "output/ML/Index")
    transport_files("input/ML/Like", "output/ML/Like")
    transport_files("input/ML/Rock", "output/ML/Rock")

    transport_files("input/ML/Watsapp/Catch", "output/ML/Catch")
    transport_files("input/ML/Watsapp/Gun", "output/ML/Gun")
    transport_files("input/ML/Watsapp/Relax", "output/ML/Relax")
    transport_files("input/ML/Watsapp/Index", "output/ML/Index")
    transport_files("input/ML/Watsapp/Like", "output/ML/Like")
    transport_files("input/ML/Watsapp/Rock", "output/ML/Rock")

    transport_files("input/ML/Запись Дин")


if __name__ == '__main__':
    main()
