import pandas as pd

def decompose_dataframe(df):
    pd.set_option('display.width', 200)  
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("General DataFrame Information:")
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nFirst Rows of DataFrame:")
    print(df.head())
    print("\nFormat (rows, columns):")
    print(df.shape)