from features.firefly import *
from config import config as cfg


for i in range(260, 621):
    FireflyOptimization(3, 50, 1, 0.99, 1, 100, seg_index=i)