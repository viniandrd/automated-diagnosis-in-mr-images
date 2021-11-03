import random, math
from custom_objects.custom_metrics import * 

from config import config as cfg
import numpy as np

class FireflyOptimization():
    def __init__(self, d, n, gamma, alpha, beta, generations):
        """
        :param d: number of pipelines (weights)
        :param n: number of agents
        :param gamma: absorption coefficient
        :param alpha: random
        :param beta: attraction factor
        :param maxGeneration: number of max generation
        :return: best firefly
        """
        self.d = d
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.generations = generations
        self.DECIMAL = 5
        self.t = 0
        self.alpha_t = 1.0
        self.bests = [0.0] * self.d

        self.metric = IoU()
        self.fireflies = []
        self.z = [0.0] * self.n
        self.ff_dis = []

        self.__first_solution()
        self.__optimize()

    def __optimize(self):
        random.seed(0)
        while self.t <= self.generations:
            for i in range(self.n):
                self.z[i] = - self.metric.iou_result(self.fireflies[i])

            index = np.argsort(self.z)

            self.z.sort()
            self.z = [-x for x in self.z]

            rank = [0.0] * self.n
            for i in range(self.n):
                rank[i] = self.fireflies[index[i]]

            self.fireflies = rank

            for i in range(self.n):
                for j in range(self.n):
                    self.ff_dis[i][j] = self.__dist(self.fireflies[i], self.fireflies[j])

            self.alpha_t = self.alpha * self.alpha_t

            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        if self.z[i] <= self.z[j]:
                            ff = self.__create_firefly(self.d)
                            beta_t = self.beta * math.exp(-self.gamma * ((self.ff_dis[i][j]) ** 2))

                            if i != self.n - 1:
                                for k in range(self.d):
                                    self.fireflies[i][k] = round( ((1 - beta_t) * self.fireflies[i][k] + beta_t *
                                                                self.fireflies[j][k] + self.alpha_t * ff[k]) /
                                                                (1 + self.alpha_t), self.DECIMAL)

            self.bests = self.fireflies[0]
            print(f'Optimized weights [0] solution for gen. = {self.t}: {self.fireflies[0]}')
            self.t += 1
            print(f'Metric avg: {self.z[0]}')
            print('--\n')
        return self.bests

    def __dist(self, a, b):
        d = 0
        for k in range(len(a)):
            d += (a[k] - b[k]) ** 2
        
        d = math.sqrt(d)

        return d

    def __first_solution(self):
        for i in range(self.n):
            firefly = self.__create_firefly(self.d)
            self.fireflies.append(firefly)

            lin = [0.0] * self.n
            self.ff_dis.append(lin)
        
        print(f'First solution [0]: {self.fireflies[0]}')
    
    def __create_firefly(self, n):
        ff = [random.random() for i in range(n)]
        ssum = sum(ff)
        ff = [i / ssum for i in ff]
        ff = [round(i, self.DECIMAL) for i in ff]
        return ff