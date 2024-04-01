import pandas as pd

class LacciAnalysis:

    def __init__(self, df : pd.DataFrame) -> None:
        self.df = df

    def get_results_descriptive(self):
        index = [
        'NUMERO_BANCO',
        'NUMERO_AGENCIA',
        'NUMERO_CONTA',
        'CPF_CNPJ_TITULAR'
        ]

        COL_TYPOLOGY = 'Typology'
        COL_ACCOUNT = 'Accounts'
        COL_TRANSACTIONS = 'Transactions'
        COL_INDIVIDUALS = 'Individuals/Companies'

        df_descriptive = self.df.copy()
        df_descriptive[COL_TYPOLOGY] = 'None'
        df_descriptive[COL_ACCOUNT] = df_descriptive['NUMERO_BANCO'].apply(lambda x: str(x)) + df_descriptive['NUMERO_AGENCIA'].apply(lambda x: str(x)) + df_descriptive['NUMERO_CONTA'].apply(lambda x: str(x))
        df_descriptive[COL_TRANSACTIONS] = range(len(df_descriptive))
            
        df_descriptive = df_descriptive.rename(columns={
            'CPF_CNPJ_TITULAR': COL_INDIVIDUALS
        })

        df_descriptive.loc[df_descriptive['I-a'].notna() & df_descriptive['I-d'].isna(), COL_TYPOLOGY] = 'I-a'
        df_descriptive.loc[df_descriptive['I-d'].notna() & df_descriptive['I-a'].isna(), COL_TYPOLOGY] = 'I-d'
        df_descriptive.loc[df_descriptive['I-d'].notna() & df_descriptive['I-a'].notna(), COL_TYPOLOGY] = 'I-a AND I-d'

        df_descriptive = df_descriptive.pivot_table(index=None, columns=COL_TYPOLOGY, values=[COL_TRANSACTIONS, COL_ACCOUNT, COL_INDIVIDUALS], aggfunc=pd.Series.nunique)
        return df_descriptive