import numpy as np
import scipy
import matplotlib
from scipy.ndimage import map_coordinates


def main():
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("Matplotlib:", matplotlib.__version__)

    dem = np.array([[10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90]], dtype=float)
    h = map_coordinates(dem, [[1.5], [1.5]], order=1)[0]
    print(f"Интерполированная высота в точке (row=1.5, col=1.5): {h:.2f}")
    print("Ожидаемое значение: 70.00")


if __name__ == "__main__":
    main()
