import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlex import PastFutureSplit
from mlex import SimplePipeline



class testSimpleRnn():

    def __init__(self) -> None:
        super().__init__()
        self.testing()


    def testing():

        path = "/data/pcpe_01.csv"
        obj = PastFutureSplit()
        df = obj.csv_extraction(path=path)

        columns_num = [
            'DIA_LANCAMENTO', 
            'MES_LANCAMENTO',
            'VALOR_TRANSACAO',
            'VALOR_SALDO',
        ]

        columns_cat = [
            'TIPO',
            'CNAB',
            'NATUREZA_SALDO'
        ]
                
        target = ['I-d']
        y = df[target].values
        y = np.nan_to_num(y)
        

        pipe = SimplePipeline(
            numeric_features = columns_num, 
            categorical_features = columns_cat,
            df = df,
            y = y,
        )
        
        