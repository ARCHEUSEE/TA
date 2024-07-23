import streamlit as st
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers


# Muat dataset dari file CSV  # menggunakan cache untuk meningkatkan kinerja
def load_data():
    data = pd.read_csv("data_penjualan_skincare.csv")  # Ganti "nama_file.csv" dengan nama file Anda
    return data

# Panggil fungsi untuk memuat dataset
data = load_data()

def main():
    # Judul halaman utama
    st.title("KAYRA BEAUTY")

    # Pilihan navigasi ke halaman lain
    page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Azarine Sunscreen Gel SPF 45", "The Originote Hyalucera Moisturizer", "Skintific 5x Cramide Moisturizer", "Wardah UV Shild SPF 30"])

    # Menampilkan konten berdasarkan halaman yang dipilih
    if page == "Home":
        show_homepage()
    elif page == "Azarine Sunscreen Gel SPF 45":
        show_secondpage()
    elif page == "The Originote Hyalucera Moisturizer":
        show_thirdpage()
    elif page == "Skintific 5x Cramide Moisturizer":
        show_forthpage()
    elif page == "Wardah UV Shild SPF 30":
        show_fifthpage()

def show_homepage():
    # Konten halaman utama
    st.header("Tentang Kami")
    st.write("""
        Kayra Beauty adalah sebuah toko reseller yang menghadirkan beragam jenis produk perawatan kulit. 
        Kami menawarkan pilihan skincare yang luas untuk memenuhi kebutuhan perawatan kulit pelanggan kami.
    """)

    st.header("Data Penjualan Skincare")
    st.write("Berikut adalah data penjualan produk skincare kami:")

    # Tampilkan dataset dalam tabel
    st.write("Dataframe:", data)

    # Visualisasi data secara umum
    st.subheader("Ringkasan Penjualan Produk")
    fig = go.Figure()

    # Buat visualisasi untuk masing-masing produk
    for product in data.columns[1:]:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[product], mode='lines', name=product))

    fig.update_layout(title='Tren Penjualan Produk Skincare', xaxis_title='Tanggal', yaxis_title='Jumlah Penjualan')
    st.plotly_chart(fig)

    # Deskripsi dataset
    st.write("""
        Dataset ini berisi informasi penjualan harian dari berbagai produk skincare yang kami tawarkan. 
        Anda dapat memilih produk tertentu dari sidebar untuk melihat analisis lebih lanjut dan prediksi penjualan.""")

#halaman prediksi produk azarine
def show_secondpage():
    # Konten halaman kedua
    st.header("Azarine Sunscreen Gel SPF 45")

    # Set the random seeds for reproducibility
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Fungsi untuk mengonversi string menjadi objek datetime
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    # Data yang akan digunakan
    data1 = data[['Date', 'Azarine Sunscreen Gel SPF 45']]

    # Mengonversi kolom 'Date' menjadi tipe datetime
    data1['Date'] = data1['Date'].apply(str_to_datetime)

    # Mengatur 'Date' sebagai indeks
    data1.index = data1.pop('Date')

    # Visualisasi data menggunakan Plotly Express
    fig = px.line(data1, x=data1.index, y='Azarine Sunscreen Gel SPF 45', title='Data Penjualan Azarine Sunscreen Gel SPF 45')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Azarine Sunscreen Gel SPF 45')
    st.plotly_chart(fig)

    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data1)

    # Menambahkan kolom baru 'scaled' ke dalam dataframe
    data1['scaled'] = scaled_data.flatten()

    # Fungsi untuk mengonversi dataframe berjendela menjadi array tanggal, X, dan y
    def create_windowed_data(dataframe, n=3):
        X, y = [], []
        for i in range(len(dataframe) - n):
            X.append(dataframe['scaled'].iloc[i:i+n].values)
            y.append(dataframe['scaled'].iloc[i+n])
        return np.array(X), np.array(y)

    # Mengonversi dataframe menjadi data berjendela
    X, y = create_windowed_data(data1)

    # Mengubah bentuk data agar sesuai dengan input LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Pembagian data menjadi data latih dan data uji
    q_80 = int(len(X) * .8)
    q_90 = int(len(X) * .9)
    dates_train, X_train, y_train = data1.index[:q_80], X[:q_80], y[:q_80]
    dates_test, X_test, y_test = data1.index[q_80:len(X)], X[q_80:], y[q_80:]

    # Membuat dan melatih model
    model = Sequential([
        layers.Input((3, 1)),
        layers.LSTM(100, return_sequences=True),
        layers.LSTM(100),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.001),
                metrics=['mean_absolute_error'])

    st.write("Training Model...")
    model.fit(X_train, y_train, epochs=400, verbose=0)

    # Membuat prediksi dan mengonversi kembali ke skala asli
    train_predictions = model.predict(X_train).flatten()
    test_predictions = model.predict(X_test).flatten()

    # Mengubah dimensi prediksi menjadi (n_samples, n_features)
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)

    # Mengembalikan prediksi ke skala asli
    train_predictions_i = scaler.inverse_transform(train_predictions)
    y_train_i = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions_i = scaler.inverse_transform(test_predictions)
    y_test_i = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Memastikan dimensi sesuai untuk visualisasi
    dates_train = dates_train[:len(train_predictions_i)]
    dates_test = dates_test[:len(test_predictions_i)]

    # Visualisasi prediksi dan data asli menggunakan Plotly Express
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_i.flatten(), mode='lines', name='Actual Train'))
    fig.add_trace(go.Scatter(x=dates_train, y=train_predictions_i.flatten(), mode='lines', name='Predicted Train'))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_i.flatten(), mode='lines', name='Actual Test'))
    fig.add_trace(go.Scatter(x=dates_test, y=test_predictions_i.flatten(), mode='lines', name='Predicted Test'))
    fig.update_layout(title='Predictions vs Actual', xaxis_title='Date', yaxis_title='Azarine Sunscreen Gel SPF 45')
    st.plotly_chart(fig)

    # Menambahkan input tanggal untuk prediksi
    st.write("Masukkan tanggal untuk prediksi:")
    input_date = st.date_input("Pilih tanggal", datetime.date(2024, 1, 1))
    input_date_str = input_date.strftime('%Y-%m-%d')

    if input_date_str in data1.index.strftime('%Y-%m-%d'):
        new_data_point = data1.loc[str_to_datetime(input_date_str)]['scaled']
        new_data_point = np.array(new_data_point).reshape((1, 1, 1))
        prediction = model.predict(new_data_point)
        prediction_i = scaler.inverse_transform(prediction.reshape(-1, 1))

        actual_value = data1.loc[str_to_datetime(input_date_str)]['Azarine Sunscreen Gel SPF 45']
        st.write("Prediksi untuk", input_date_str, ":", prediction_i[0][0])
        st.write("Data aktual untuk", input_date_str, ":", actual_value)

        # Visualisasi perbandingan aktual dan prediksi menggunakan Plotly Express
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=['Actual'], y=[actual_value]),
            go.Bar(name='Predicted', x=['Predicted'], y=[prediction_i[0][0]])
        ])
        fig.update_layout(title=f'Actual vs Predicted for {input_date_str}', xaxis_title='Type', yaxis_title='Azarine Sunscreen Gel SPF 45')
        st.plotly_chart(fig)
    else:
        st.write("Tanggal tidak ditemukan dalam data. Mohon pilih tanggal lain.")

    new_data_point = test_predictions_i[-1]  # Data terbaru (sampai H hari terakhir yang tersedia)
    new_data_point = new_data_point.reshape((1, 1, 1))
    prediction = model.predict(new_data_point)
    st.write("Prediksi H+1:", prediction[0][0]) 

    prediction_value = prediction[0][0]  # Mendapatkan nilai prediksi
    if prediction_value < -0:
        rounded_prediction = round(prediction_value)*0
    else:
        rounded_prediction = round(prediction_value)

    st.write("Prediksi H+1 yang dibulatkan:", rounded_prediction)

    # Menghitung MAE
    mae = mean_absolute_error(y_test, test_predictions_i)

    # Menampilkan hasil
    st.write("Mean Absolute Error (MAE) :", mae)

def show_thirdpage():
    # Konten halaman Ketiga
    st.header("The Originote Hyalucera Moisturizer")

    # Fungsi untuk mengonversi string menjadi objek datetime
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    # Set the random seeds for reproducibility
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Data yang akan digunakan
    data2 = data[['Date', 'The Originote Hyalucera Moisturizer']]

    # Mengonversi kolom 'Date' menjadi tipe datetime
    data2['Date'] = data2['Date'].apply(str_to_datetime)

    # Mengatur 'Date' sebagai indeks
    data2.index = data2.pop('Date')

    # Visualisasi data menggunakan Plotly Express
    fig = px.line(data2, x=data2.index, y='The Originote Hyalucera Moisturizer', title='Data Penjualan The Originote Hyalucera Moisturizer')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='The Originote Hyalucera Moisturizer')
    st.plotly_chart(fig)

    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data2)

    # Menambahkan kolom baru 'scaled' ke dalam dataframe
    data2['scaled'] = scaled_data.flatten()

    # Fungsi untuk mengonversi dataframe berjendela menjadi array tanggal, X, dan y
    def create_windowed_data(dataframe, n=3):
        X, y = [], []
        for i in range(len(dataframe) - n):
            X.append(dataframe['scaled'].iloc[i:i+n].values)
            y.append(dataframe['scaled'].iloc[i+n])
        return np.array(X), np.array(y)

    # Mengonversi dataframe menjadi data berjendela
    X, y = create_windowed_data(data2)

    # Mengubah bentuk data agar sesuai dengan input LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Pembagian data menjadi data latih dan data uji
    q_80 = int(len(X) * .8)
    q_90 = int(len(X) * .9)
    dates_train, X_train, y_train = data2.index[:q_80], X[:q_80], y[:q_80]
    dates_test, X_test, y_test = data2.index[q_80:len(X)], X[q_80:], y[q_80:]

    # Membuat dan melatih model
    model = Sequential([
        layers.Input((3, 1)),
        layers.LSTM(25, return_sequences=True),
        layers.LSTM(25),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.001),
                metrics=['mean_absolute_error'])

    st.write("Training Model...")
    model.fit(X_train, y_train, epochs=400, verbose=0)

    # Membuat prediksi dan mengonversi kembali ke skala asli
    train_predictions = model.predict(X_train).flatten()
    test_predictions = model.predict(X_test).flatten()

    # Mengubah dimensi prediksi menjadi (n_samples, n_features)
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)

    # Mengembalikan prediksi ke skala asli
    train_predictions_i = scaler.inverse_transform(train_predictions)
    y_train_i = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions_i = scaler.inverse_transform(test_predictions)
    y_test_i = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Memastikan dimensi sesuai untuk visualisasi
    dates_train = dates_train[:len(train_predictions_i)]
    dates_test = dates_test[:len(test_predictions_i)]

    # Visualisasi prediksi dan data asli menggunakan Plotly Express
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_i.flatten(), mode='lines', name='Actual Train'))
    fig.add_trace(go.Scatter(x=dates_train, y=train_predictions_i.flatten(), mode='lines', name='Predicted Train'))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_i.flatten(), mode='lines', name='Actual Test'))
    fig.add_trace(go.Scatter(x=dates_test, y=test_predictions_i.flatten(), mode='lines', name='Predicted Test'))
    fig.update_layout(title='Predictions vs Actual', xaxis_title='Date', yaxis_title='The Originote Hyalucera Moisturizer')
    st.plotly_chart(fig)

    # Menambahkan input tanggal untuk prediksi
    st.write("Masukkan tanggal untuk prediksi:")
    input_date = st.date_input("Pilih tanggal", datetime.date(2024, 1, 1))
    input_date_str = input_date.strftime('%Y-%m-%d')

    if input_date_str in data2.index.strftime('%Y-%m-%d'):
        new_data_point = data2.loc[str_to_datetime(input_date_str)]['scaled']
        new_data_point = np.array(new_data_point).reshape((1, 1, 1))
        prediction = model.predict(new_data_point)
        prediction_i = scaler.inverse_transform(prediction.reshape(-1, 1))

        actual_value = data2.loc[str_to_datetime(input_date_str)]['The Originote Hyalucera Moisturizer']
        st.write("Prediksi untuk", input_date_str, ":", prediction_i[0][0])
        st.write("Data aktual untuk", input_date_str, ":", actual_value)

        # Visualisasi perbandingan aktual dan prediksi menggunakan Plotly Express
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=['Actual'], y=[actual_value]),
            go.Bar(name='Predicted', x=['Predicted'], y=[prediction_i[0][0]])
        ])
        fig.update_layout(title=f'Actual vs Predicted for {input_date_str}', xaxis_title='Type', yaxis_title='The Originote Hyalucera Moisturizer')
        st.plotly_chart(fig)
    else:
        st.write("Tanggal tidak ditemukan dalam data. Mohon pilih tanggal lain.")

    new_data_point = test_predictions_i[-1]  # Data terbaru (sampai H hari terakhir yang tersedia)
    new_data_point = new_data_point.reshape((1, 1, 1))
    prediction = model.predict(new_data_point)
    st.write("Prediksi H+1:", prediction[0][0])

    prediction_value = prediction[0][0]  # Mendapatkan nilai prediksi
    if prediction_value < -0:
        rounded_prediction = round(prediction_value)*0
    else:
        rounded_prediction = round(prediction_value)

    st.write("Prediksi H+1 yang dibulatkan:", rounded_prediction)

        # Menghitung MAE
    mae = mean_absolute_error(y_test, test_predictions_i)

    # Menampilkan hasil
    st.write("Mean Absolute Error (MAE):", mae)

def show_forthpage():
    # Konten halaman Keempat
    st.header("Skintific 5x Cramide Moisturizer")

    # Fungsi untuk mengonversi string menjadi objek datetime
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    # Set the random seeds for reproducibility
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Data yang akan digunakan
    data3 = data[['Date', 'Skintific 5x Cramide Moisturizer']]

    # Mengonversi kolom 'Date' menjadi tipe datetime
    data3['Date'] = data3['Date'].apply(str_to_datetime)

    # Mengatur 'Date' sebagai indeks
    data3.index = data3.pop('Date')

    # Visualisasi data menggunakan Plotly Express
    fig = px.line(data3, x=data3.index, y='Skintific 5x Cramide Moisturizer', title='Data Penjualan Skintific 5x Cramide Moisturizer')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Skintific 5x Cramide Moisturizer')
    st.plotly_chart(fig)

    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data3)

    # Menambahkan kolom baru 'scaled' ke dalam dataframe
    data3['scaled'] = scaled_data.flatten()

    # Fungsi untuk mengonversi dataframe berjendela menjadi array tanggal, X, dan y
    def create_windowed_data(dataframe, n=3):
        X, y = [], []
        for i in range(len(dataframe) - n):
            X.append(dataframe['scaled'].iloc[i:i+n].values)
            y.append(dataframe['scaled'].iloc[i+n])
        return np.array(X), np.array(y)

    # Mengonversi dataframe menjadi data berjendela
    X, y = create_windowed_data(data3)

    # Mengubah bentuk data agar sesuai dengan input LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Pembagian data menjadi data latih dan data uji
    q_80 = int(len(X) * .8)
    q_90 = int(len(X) * .9)
    dates_train, X_train, y_train = data3.index[:q_80], X[:q_80], y[:q_80]
    dates_test, X_test, y_test = data3.index[q_80:len(X)], X[q_80:], y[q_80:]

    # Membuat dan melatih model
    model = Sequential([
        layers.Input((3, 1)),
        layers.LSTM(25, return_sequences=True),
        layers.LSTM(25),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.001),
                metrics=['mean_absolute_error'])

    st.write("Training Model...")
    model.fit(X_train, y_train, epochs=400, verbose=0)

    # Membuat prediksi dan mengonversi kembali ke skala asli
    train_predictions = model.predict(X_train).flatten()
    test_predictions = model.predict(X_test).flatten()

    # Mengubah dimensi prediksi menjadi (n_samples, n_features)
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)

    # Mengembalikan prediksi ke skala asli
    train_predictions_i = scaler.inverse_transform(train_predictions)
    y_train_i = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions_i = scaler.inverse_transform(test_predictions)
    y_test_i = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Memastikan dimensi sesuai untuk visualisasi
    dates_train = dates_train[:len(train_predictions_i)]
    dates_test = dates_test[:len(test_predictions_i)]

    # Visualisasi prediksi dan data asli menggunakan Plotly Express
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_i.flatten(), mode='lines', name='Actual Train'))
    fig.add_trace(go.Scatter(x=dates_train, y=train_predictions_i.flatten(), mode='lines', name='Predicted Train'))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_i.flatten(), mode='lines', name='Actual Test'))
    fig.add_trace(go.Scatter(x=dates_test, y=test_predictions_i.flatten(), mode='lines', name='Predicted Test'))
    fig.update_layout(title='Predictions vs Actual', xaxis_title='Date', yaxis_title='Skintific 5x Cramide Moisturizer')
    st.plotly_chart(fig)

    # Menambahkan input tanggal untuk prediksi
    st.write("Masukkan tanggal untuk prediksi:")
    input_date = st.date_input("Pilih tanggal", datetime.date(2024, 1, 1))
    input_date_str = input_date.strftime('%Y-%m-%d')

    if input_date_str in data3.index.strftime('%Y-%m-%d'):
        new_data_point = data3.loc[str_to_datetime(input_date_str)]['scaled']
        new_data_point = np.array(new_data_point).reshape((1, 1, 1))
        prediction = model.predict(new_data_point)
        prediction_i = scaler.inverse_transform(prediction.reshape(-1, 1))

        actual_value = data3.loc[str_to_datetime(input_date_str)]['Skintific 5x Cramide Moisturizer']
        st.write("Prediksi untuk", input_date_str, ":", prediction_i[0][0])
        st.write("Data aktual untuk", input_date_str, ":", actual_value)

        # Visualisasi perbandingan aktual dan prediksi menggunakan Plotly Express
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=['Actual'], y=[actual_value]),
            go.Bar(name='Predicted', x=['Predicted'], y=[prediction_i[0][0]])
        ])
        fig.update_layout(title=f'Actual vs Predicted for {input_date_str}', xaxis_title='Type', yaxis_title='Skintific 5x Cramide Moisturizer')
        st.plotly_chart(fig)
    else:
        st.write("Tanggal tidak ditemukan dalam data. Mohon pilih tanggal lain.")

    new_data_point = test_predictions_i[-1]  # Data terbaru (sampai H hari terakhir yang tersedia)
    new_data_point = new_data_point.reshape((1, 1, 1))
    prediction = model.predict(new_data_point)
    st.write("Prediksi H+1:", prediction[0][0])

    prediction_value = prediction[0][0]  # Mendapatkan nilai prediksi
    if prediction_value < -0:
        rounded_prediction = round(prediction_value)*0
    else:
        rounded_prediction = round(prediction_value)

    st.write("Prediksi H+1 yang dibulatkan:", rounded_prediction)

     # Menghitung MAE
    mae = mean_absolute_error(y_test, test_predictions_i)

    # Menampilkan hasil
    st.write("Mean Absolute Error (MAE):", mae)

def show_fifthpage():
    # Konten halaman Kelima
    st.header("Wardah UV Shild SPF 30")

    # Fungsi untuk mengonversi string menjadi objek datetime
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    # Set the random seeds for reproducibility
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Data yang akan digunakan
    data4 = data[['Date', 'Wardah UV Shild SPF 30']]

    # Mengonversi kolom 'Date' menjadi tipe datetime
    data4['Date'] = data4['Date'].apply(str_to_datetime)

    # Mengatur 'Date' sebagai indeks
    data4.index = data4.pop('Date')

    # Visualisasi data menggunakan Plotly Express
    fig = px.line(data4, x=data4.index, y='Wardah UV Shild SPF 30', title='Data Penjualan Wardah UV Shild SPF 30')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Wardah UV Shild SPF 30')
    st.plotly_chart(fig)

    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data4)

    # Menambahkan kolom baru 'scaled' ke dalam dataframe
    data4['scaled'] = scaled_data.flatten()

    # Fungsi untuk mengonversi dataframe berjendela menjadi array tanggal, X, dan y
    def create_windowed_data(dataframe, n=3):
        X, y = [], []
        for i in range(len(dataframe) - n):
            X.append(dataframe['scaled'].iloc[i:i+n].values)
            y.append(dataframe['scaled'].iloc[i+n])
        return np.array(X), np.array(y)

    # Mengonversi dataframe menjadi data berjendela
    X, y = create_windowed_data(data4)

    # Mengubah bentuk data agar sesuai dengan input LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Pembagian data menjadi data latih dan data uji
    q_80 = int(len(X) * .8)
    q_90 = int(len(X) * .9)
    dates_train, X_train, y_train = data4.index[:q_80], X[:q_80], y[:q_80]
    dates_test, X_test, y_test = data4.index[q_80:len(X)], X[q_80:], y[q_80:]

    # Membuat dan melatih model
    model = Sequential([
        layers.Input((3, 1)),
        layers.LSTM(50, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.001),
                metrics=['mean_absolute_error'])

    st.write("Training Model...")
    model.fit(X_train, y_train, epochs=400, verbose=0)

    # Membuat prediksi dan mengonversi kembali ke skala asli
    train_predictions = model.predict(X_train).flatten()
    test_predictions = model.predict(X_test).flatten()

    # Mengubah dimensi prediksi menjadi (n_samples, n_features)
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)

    # Mengembalikan prediksi ke skala asli
    train_predictions_i = scaler.inverse_transform(train_predictions)
    y_train_i = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions_i = scaler.inverse_transform(test_predictions)
    y_test_i = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Memastikan dimensi sesuai untuk visualisasi
    dates_train = dates_train[:len(train_predictions_i)]
    dates_test = dates_test[:len(test_predictions_i)]

    # Visualisasi prediksi dan data asli menggunakan Plotly Express
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_i.flatten(), mode='lines', name='Actual Train'))
    fig.add_trace(go.Scatter(x=dates_train, y=train_predictions_i.flatten(), mode='lines', name='Predicted Train'))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_i.flatten(), mode='lines', name='Actual Test'))
    fig.add_trace(go.Scatter(x=dates_test, y=test_predictions_i.flatten(), mode='lines', name='Predicted Test'))
    fig.update_layout(title='Predictions vs Actual', xaxis_title='Date', yaxis_title='Wardah UV Shild SPF 30')
    st.plotly_chart(fig)

    # Menambahkan input tanggal untuk prediksi
    st.write("Masukkan tanggal untuk prediksi:")
    input_date = st.date_input("Pilih tanggal", datetime.date(2024, 1, 1))
    input_date_str = input_date.strftime('%Y-%m-%d')

    if input_date_str in data4.index.strftime('%Y-%m-%d'):
        new_data_point = data4.loc[str_to_datetime(input_date_str)]['scaled']
        new_data_point = np.array(new_data_point).reshape((1, 1, 1))
        prediction = model.predict(new_data_point)
        prediction_i = scaler.inverse_transform(prediction.reshape(-1, 1))

        actual_value = data4.loc[str_to_datetime(input_date_str)]['Wardah UV Shild SPF 30']
        st.write("Prediksi untuk", input_date_str, ":", prediction_i[0][0])
        st.write("Data aktual untuk", input_date_str, ":", actual_value)

        # Visualisasi perbandingan aktual dan prediksi menggunakan Plotly Express
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=['Actual'], y=[actual_value]),
            go.Bar(name='Predicted', x=['Predicted'], y=[prediction_i[0][0]])
        ])
        fig.update_layout(title=f'Actual vs Predicted for {input_date_str}', xaxis_title='Type', yaxis_title='Wardah UV Shild SPF 30')
        st.plotly_chart(fig)
    else:
        st.write("Tanggal tidak ditemukan dalam data. Mohon pilih tanggal lain.")

    new_data_point = test_predictions_i[-1]  # Data terbaru (sampai H hari terakhir yang tersedia)
    new_data_point = new_data_point.reshape((1, 1, 1))
    prediction = model.predict(new_data_point)
    st.write("Prediksi H+1:", prediction[0][0])

    prediction_value = prediction[0][0]  # Mendapatkan nilai prediksi
    if prediction_value < -0:
        rounded_prediction = round(prediction_value)*0
    else:
        rounded_prediction = round(prediction_value)

    st.write("Prediksi H+1 yang dibulatkan:", rounded_prediction)

    # Menghitung MAE
    mae = mean_absolute_error(y_test, test_predictions_i)

    # Menampilkan hasil
    st.write("Mean Absolute Error (MAE):", mae)

if __name__ == "__main__":
    main()