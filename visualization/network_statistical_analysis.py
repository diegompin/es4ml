import os
import random
import zipfile
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import compress
from abc import ABC, abstractmethod
from tqdm import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt


class NetworkStatisticalAnalysis:
    def __init__(self, data, real_data_point, alpha=.05):
        self.data = data
        self.alpha = alpha
        self.mean = data.mean()
        self.std = data.std()
        self._z_i = st.norm.ppf(1 - alpha/2)
        self.real_data_point = real_data_point

    def calculate_lower_bound(self):
         return (self.mean - self._z_i * self.std) 
    
    
    def calculate_upper_bound(self):
         return (self.mean + self._z_i * self.std) 
    

    def plot_confidence_interval(self,x, horizontal_line_width=0.25):
        left = x - horizontal_line_width / 2
        top = self.calculate_lower_bound()
        right = x + horizontal_line_width / 2
        bottom = self.calculate_upper_bound()
        plt.xticks([1], ['year'])
        plt.plot([x, x], [top, bottom], color='#2187bb')
        plt.plot([left, right], [top, top], color='#2187bb')
        plt.plot([left, right], [bottom, bottom], color='#2187bb')
        plt.plot(x, self.mean, 'o', color='#2187bb')
        plt.plot(x, self.real_data_point, 'ro', label='Real Data')
        plt.title('Confidence Interval')
        plt.text(x, self.real_data_point+.8, "%d" %self.real_data_point, ha="center")

        plt.xlim(0,10)
        plt.ylim(0,60)
        plt.legend()
        plt.show()

