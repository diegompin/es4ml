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
    def __init__(self, data_coefficients, real_data_points,year_dict, alpha=.05):
        self.data_coefficients = data_coefficients
        self.alpha = alpha
        self._z_i = st.norm.ppf(1 - alpha/2)
        self.real_data_points = real_data_points
        self.year_dict = year_dict

    def calculate_lower_bound(self,coefficients):
         return (coefficients.mean() - self._z_i * coefficients.std()) 
    
    
    def calculate_upper_bound(self,coefficients):
         return (coefficients.mean() + self._z_i * coefficients.std()) 
    

    def plot_confidence_interval(self,horizontal_line_width=0.25):
        positions = [i+1 for i, _ in enumerate(self.year_dict)]
        plt.xticks(positions, self.year_dict.values())
        for index,_ in enumerate(self.year_dict):
            left = positions[index] - horizontal_line_width / 2
            top = self.calculate_lower_bound(coefficients=self.data_coefficients[index])
            right = positions[index] + horizontal_line_width / 2
            bottom = self.calculate_upper_bound(coefficients=self.data_coefficients[index])
            plt.plot([positions[index], positions[index]], [top, bottom], color='#2187bb')
            plt.plot([left, right], [top, top], color='#2187bb')
            plt.plot([left, right], [bottom, bottom], color='#2187bb')
            plt.plot(positions[index], self.data_coefficients[index].mean(), 'o', color='#2187bb')
            plt.hlines(top, left, right, color='#2187bb', linestyle='--')  
            plt.text(right + 0.1, top, f"{top:.3f}", va='center', ha='left', color='#2187bb')  
            plt.hlines(bottom, left, right, color='#2187bb', linestyle='--') 
            plt.text(right + 0.1, bottom, f"{bottom:.3f}", va='center', ha='left', color='#2187bb')  
            #plt.plot(positions[index], self.real_data_points[index], 'ro')
            plt.title('Confidence Interval')
            plt.text(positions[index], self.real_data_points[index]-.04, "%f" %self.real_data_points[index], ha="center")
            print(f'{self.year_dict[index]} Lower bound: {top} Upper bound {bottom}')
            
        plt.plot(positions, self.real_data_points, 'ro', label="Real Network Metric")
        plt.xlim(0,len(self.data_coefficients) +1)
        plt.ylim(-.4,.2)
        plt.legend()
        plt.xlabel("Years")
        plt.ylabel("Attribute Assortativity")
        plt.savefig("confidence_intervals.png")
        plt.show()



if __name__ == '__main__':
    year_dict = {0: "2015", 1:"2016", 2: "2017"}
    assortativity_values = [np.array([12, 14, 23, 13, 29]),np.array([24, 34, 32, 29, 40]),np.array([42, 40, 39, 47, 48])]
    real_values=[37,48,60]
    statistical_analysis = NetworkStatisticalAnalysis(year_dict=year_dict,data_coefficients=assortativity_values,real_data_points=real_values)
    statistical_analysis.plot_confidence_interval()

