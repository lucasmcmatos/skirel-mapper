import os
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from selenium import webdriver
import math
import time
from joblib import load
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MODEL_PATH = './modelo/modelo_otimizado.pkl'

# Verifica se o arquivo é permitido
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# Valida o dataset
def validate_dataset(data, model_expected_features):
    # Colunas adicionais necessárias para o mapeamento
    mapping_expected_features = ['latitude', 'longitude', 'ELEVATION', 'AZIMUTH']

    # Verificar colunas ausentes
    missing_columns = set(model_expected_features + mapping_expected_features) - set(data.columns)
    if missing_columns:
        raise ValueError(f"As seguintes colunas estão ausentes no dataset: {missing_columns}")

    # Garantir que as colunas estejam preenchidas e sejam numéricas
    for col in set(model_expected_features + mapping_expected_features):
        if data[col].isnull().any():
            raise ValueError(f"A coluna {col} contém valores ausentes.")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"A coluna {col} contém valores não numéricos.")

    # Retornar as colunas esperadas pelo modelo e as necessárias para o mapeamento
    return data

# Calcula o ponto ionosférico
def calc_ipp(lat_estacao,long_estacao,elevacao,azimute):
    try:
        R_EARTH = 6371.0 # Raio médio da terra
        H_IONO = 350.0  # Altura média do pico da camada ionosférica em km
        elevacao_rad = math.radians(elevacao)
        azimute_rad = math.radians(azimute)
        psi = math.acos((R_EARTH / (R_EARTH + H_IONO)) * math.cos(elevacao_rad)) - elevacao_rad
        lat_ipp = lat_estacao + math.degrees(psi * math.cos(azimute_rad))
        long_ipp = long_estacao + math.degrees(psi * math.sin(azimute_rad) / math.cos(math.radians(lat_ipp)))
        return lat_ipp, long_ipp
    except Exception as e:
        print(f"Erro ao calcular IPP para: lat={lat_estacao}, long={long_estacao}, elev={elevacao}, az={azimute}")
        raise e

# Processa o dataset para gerar os mapas de calor
def mapping_process(filepath, output_dir):

    # Verifica se o arquivo é permitido
    if not allowed_file(filepath):
        raise ValueError("Tipo de arquivo não permitido. Envie apenas arquivos '.xlsx' ou '.csv'")

    
    model = load(MODEL_PATH) # Carrega o modelo de treinamento
    dataset = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath) # Carrega o dataset recebido
    model_expected_features = model.get_booster().feature_names # Instância das features esperadas pelo modelo
    dataset = validate_dataset(dataset, model_expected_features) # Valida o dataset

    # Adicionar pontos IPP ao dataset
    dataset['lat_ipp'], dataset['long_ipp'] = zip(*dataset.apply(
        lambda row: calc_ipp(row['latitude'], row['longitude'], row['ELEVATION'], row['AZIMUTH']), axis=1))

    # Gera mapa antes da predição
    if 'S4' not in dataset.columns:
        raise ValueError("A coluna 'S4' não está presente no dataset.")
    
    dataset['intensidade_entrada'] = dataset['S4']  # Considera a coluna S4_30 como intensidade
    m_entrada = folium.Map(location=[dataset['lat_ipp'].mean(), dataset['long_ipp'].mean()], zoom_start=4)
    for index, row in dataset.iterrows(): # Adicionar pontos com Tooltip ao mapa de entrada
        folium.CircleMarker(
            location=[row['lat_ipp'], row['long_ipp']],
            radius=2.5,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6,
            popup=f"Intensidade: {row['intensidade_entrada']:.2f}"
        ).add_to(m_entrada)
    heat_data_entrada = [[row['lat_ipp'], row['long_ipp'], row['intensidade_entrada']] for index, row in dataset.iterrows()] # Adicionar HeatMap ao mapa de entrada
    HeatMap(heat_data_entrada).add_to(m_entrada)
    entrada_map_path = os.path.join(output_dir, "map_input.html")
    m_entrada.save(entrada_map_path) 

    # Gera mapa antes da predição
    data_to_predict = dataset.drop(['lat_ipp', 'long_ipp', 'latitude', 'longitude', 'intensidade_entrada'], axis=1)
    dataset['S4_predictions'] =  model.predict(data_to_predict)# Considera a predição como intensidade
    m_predito = folium.Map(location=[dataset['lat_ipp'].mean(), dataset['long_ipp'].mean()], zoom_start=4)
    for index, row in dataset.iterrows(): # Adicionar pontos com Tooltip ao mapa de predição
        folium.CircleMarker(
            location=[row['lat_ipp'], row['long_ipp']],
            radius=2.5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"Intensidade: {row['S4_predictions']:.2f}"
        ).add_to(m_predito)
    heat_data_predito = [[row['lat_ipp'], row['long_ipp'], row['S4_predictions']] for index, row in dataset.iterrows()] # Adicionar HeatMap ao mapa de predição
    HeatMap(heat_data_predito).add_to(m_predito)
    predito_map_path = os.path.join(output_dir, "map_output.html")
    m_predito.save(predito_map_path)    

    # Salva o arquivo de predição
    resultado_path = os.path.join(output_dir, "prediction.xlsx")
    dataset.to_excel(resultado_path, index=False)

    return {
        "input_map": entrada_map_path,
        "output_map": predito_map_path,
        "prediction": resultado_path
    }