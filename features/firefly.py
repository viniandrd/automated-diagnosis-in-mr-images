import math

from metrics.iou_metric import *

DECIMAL = 2


def firefly(d, n, gamma, alpha, beta, maxGeneration):
    """
    :param d: number of pipelines (weights)
    :param n: number of agents
    :param gamma: absorption coefficient
    :param alpha: random
    :param beta: attraction factor
    :param maxGeneration: number of max generation
    :return: best firefly
    """
    t = 0
    alpha_t = 1.0
    random.seed(0)
    metric = IoU()
    bests = [0.0] * d

    fireflies = []
    z = [0.0] * n  # Result metric vector
    ff_dis = []  # Distance vector

    # Initial solution
    for i in range(n):
        ff = create_firefly(d)
        fireflies.append(ff)

        lin = [0.0] * n
        ff_dis.append(lin)

    print(f'Conjunto inicial: {fireflies}')
    while t < maxGeneration:
        for i in range(n):
            # Metric results
            z[i] = -metric.iou_result(fireflies[i])

        # Index by results
        indice = np.argsort(z)

        # Sort results
        z.sort()
        z = [-x for x in z]

        # Rank firefly by intensity
        rank = [0.0] * n
        for i in range(n):
            rank[i] = fireflies[indice[i]]

        fireflies = rank

        for i in range(n):
            for j in range(n):
                ff_dis[i][j] = dist(fireflies[i], fireflies[j])

        alpha_t = alpha * alpha_t
        for i in range(n):
            for j in range(n):
                if z[i] < z[j]:
                    ff = create_firefly(d)
                    beta_t = beta * math.exp(-gamma * ((ff_dis[i][j]) ** 2))
                    if i != n - 1:
                        for k in range(d):
                            fireflies[i][k] = round(
                                ((1 - beta_t) * fireflies[i][k] + beta_t * fireflies[j][k] + alpha_t * ff[k]) / (
                                        1 + alpha_t), DECIMAL)
        bests = fireflies[0]
        t += 1
        print(f'{t} - Solucao de pesos: {fireflies}')
        print(f'IoU: {z}')

    return bests


# Create a firefly with random weights
def create_firefly(n):
    ff = [random.random() for i in range(n)]
    ssum = sum(ff)
    ff = [i / ssum for i in ff]
    ff = [round(i, DECIMAL) for i in ff]
    return ff


# Calculate distance
def dist(a, b):
    d = 0
    for k in range(len(a)):
        d += (a[k] - b[k]) ** 2
    d = math.sqrt(d)
    return d
