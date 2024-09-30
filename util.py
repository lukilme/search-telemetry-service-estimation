import pandas as pd
import numpy as np
from scipy import stats

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

def get_data_frames():
    # telemetry data
    data_log = pd.read_csv("datasets/log_INT_sinusoid_8h.txt", delimiter=",")
    data_log.columns = data_log.columns.str.replace(" ", "")
    # QoS data
    data_dash = pd.read_csv("datasets/dash_sinusoid_8h.log", sep=",")
    return data_log, data_dash

def get_data_concated(data_log, data_dash):
  #removing all columns with std equal to 0.0, because they are constant 
  #variables and did not help in creating the model
  for colum in data_log.columns:
      if(data_log[colum].std()==0.0):
          data_log = data_log.drop(columns=[colum])
      
  #merging dataframes to correlate QoS results with telemetry
  data_log = data_log.groupby("timestamp").mean().reset_index()
  result_total = pd.concat(
      [data_log.set_index("timestamp"), data_dash.set_index("timestamp")],
      axis=1,
      join="outer",
  )

  #fill in the empty space with the column measurement
  for index in result_total.columns:
    media = result_total[index].mean()
    result_total[index] = result_total[index].fillna(media)
  result_total = result_total.reset_index()

  #drop the outliers
  z_scores = np.abs(stats.zscore(result_total['framesDisplayedCalc']))
  threshold = 3
  result_total = result_total[(z_scores < threshold)]

  #get the labels
  labels = result_total['framesDisplayedCalc']

  #and drop the rest
  columns_to_remove = list(data_dash.columns)
  columns_to_remove.remove('timestamp')
  result_total = result_total.drop(columns=columns_to_remove)
  return result_total, labels

def alert_end():
    from plyer import notification

    notification.notify(
        title='Alerta Importante!',
        message='Terminou o teste, cabeção',
        app_icon='alert.ico',
        timeout=5,
        ticker='Novo e-mail',
        toast=True,
    )