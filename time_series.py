import pandas as pd
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy.fft import fft
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

uploaded_file = st.file_uploader('upload your time series')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    date_column = st.selectbox("Выберите столбец с датами", options=data.columns)
    for i in data.columns:
        if i != date_column:
            values_column = i
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)

    period_val = {'День': 'D', 'Неделя': 'W', 'Месяц': 'M', 'Год': 'Y'}
    agg_val = {'Сумма': 'sum', 'Среднее': 'mean', 'Медиана': 'median'}
    
    # period = st.selectbox("Выберите группировку", options=['День', 'Неделя', 'Месяц', 'Год'])
    period = st.select_slider('Выберите период', options=['День', 'Неделя', 'Месяц', 'Год'], value='Неделя')
    agg = st.selectbox("Выберите агрегирующую функцию", options=['Сумма', 'Среднее', 'Медиана'])
    
    if period and agg:

        data = data.resample(period_val[period]).agg({data.columns[0]: agg_val[agg]})
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Sales'])
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Применяем метод декомпозиции временного ряда
        decomposition = sm.tsa.seasonal_decompose(data['Sales'], model='additive')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Выводим полученные компоненты
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Построение графиков и задание названий
        axes[0].plot(trend)
        axes[0].set_title('Тренд')
        
        axes[1].plot(seasonal)
        axes[1].set_title('Сезонность')
        
        axes[2].plot(residual)
        axes[2].set_title('Остатки')
        
        # Отображение графика
        plt.tight_layout()
        st.pyplot(fig)

        fft_seasonal = fft(seasonal.values)
        amplitudes = np.abs(fft_seasonal)
        period_index = np.argmax(amplitudes)
        # Период сезонности в количестве точек данных
        seasonality_period = max(len(data['Sales']) // period_index, 1)



        prophet_df = data.reset_index().rename(columns={date_column: 'ds', values_column: 'y'})
        model = Prophet()

        split_index = -seasonality_period

        # Разделение данных на обучающий и тестовый наборы
        data_train = prophet_df.iloc[:split_index]
        data_test = prophet_df.iloc[split_index:]

        # st.write(data_train.head())
        
        model.fit(data_train)
        
        number_of_future_predicted_points = 2 * seasonality_period 
        
        future = model.make_future_dataframe(periods=number_of_future_predicted_points, freq=period_val[period])
        forecast = model.predict(future)


        forecast_train = forecast[:-number_of_future_predicted_points] # Трейновый период
        forecast_test = forecast[-number_of_future_predicted_points: - number_of_future_predicted_points + len(data_test)] # Тестовый
        forecast_future = forecast[-number_of_future_predicted_points + len(data_test):] # Будущий период
        
        
        prophet_mae_train = np.round(mean_absolute_error(data_train['y'], forecast_train['yhat']), 1)
        prophet_mae_test = np.round(mean_absolute_error(data_test['y'], forecast_test['yhat']), 1)

        prophet_r2_train = np.round(r2_score(data_train['y'], forecast_train['yhat']), 2)
        prophet_r2_test = np.round(r2_score(data_test['y'], forecast_test['yhat']), 2)
        
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(data['Sales'], label='true_data', marker='o')
        
        ax.plot(forecast_train['ds'], forecast_train['yhat'], marker='v', linestyle=':', label='forecast_train')
        ax.plot(forecast_test['ds'], forecast_test['yhat'], marker='v', linestyle=':', label='forecast_test')
        ax.plot(forecast_future['ds'], forecast_future['yhat'], marker='v', linestyle=':', label='forecast_future', color='b')
        
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.15)
        plt.xticks(rotation=45)
        
        ax.set_title('Prophet')
        plt.legend()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.metric("**MAE train**", prophet_mae_train)
        col2.metric("**R2 train**", prophet_r2_train)
        
        col3.metric("**MAE test**", prophet_mae_test)
        col4.metric("**R2 test**", prophet_r2_test)

        
            

    

        
            
