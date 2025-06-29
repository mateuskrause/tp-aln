import numpy as np
import matplotlib.pyplot as plt
from periodic_bspline import make_interp_spline

def closed_bspline_through_points(points, num_points=200):
    """
    Dados n pontos 2D, retorna uma curva B-spline fechada passando por todos os pontos.
    """
    points = np.asarray(points)
    points = np.vstack([points, points[0]])     # fecha a curva repetindo o primeiro ponto no fim

    n = len(points)
    print(f"Number of points: {n}")

    # parametriza os pontos uniformemente no intervalo [0, 1]
    t = np.linspace(0, 1, n)
    # print(f"t: {t}")

    # interpola os pontos separadamente
    spl_x = make_interp_spline(t, points[:, 0], k=3, bc_type='periodic')
    spl_y = make_interp_spline(t, points[:, 1], k=3, bc_type='periodic')

    t_dense = np.linspace(0, 1, num_points)
    x_dense = spl_x(t_dense)
    y_dense = spl_y(t_dense)

    return x_dense, y_dense


if __name__ == "__main__":
    # # pontos de exemplo
    points = np.array([
        [5114, 7717],
        [3945, 8831],
        [2070, 9300],
        [1977, 8258],
        [2297, 7445],
        [1202, 7088],
        [ 343, 5265],
        [1724, 4626],
        [3192, 3791],
        [4961,  343],
        [7049, 4442],
        [9204, 5425],
        [8552, 6917],
        [7668, 7679],
        [7823, 9082],
        [5812, 8665]
    ])
    points[:, 1] = 10000 - points[:, 1]

    # pontos de exemplo
    # points = np.array([
    #     [0, 0],
    #     [0, 1],
    #     [1, 1],
    #     [1, 0]
    # ])
    # points[:, 1] = 10000 - points[:, 1]

    # criar e obter pontos da curva
    x, y = closed_bspline_through_points(points)

    # plotar os pontos e a curva
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label='Closed B-spline')
    plt.plot(points[:, 0], points[:, 1], 'ro', label='Given Points')
    plt.axis('equal')
    plt.legend()
    plt.title('Closed B-spline Curve Through Given Points')
    plt.show()