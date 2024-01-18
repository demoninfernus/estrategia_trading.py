import pandas as pd
import numpy as np
import talib
import yfinance as yf
from mpmath import mp
import logging

# Configurar precisión para mpmath
mp.dps = 15

# Configurar el nivel de registro para mensajes de error
logging.basicConfig(level=logging.ERROR)

def descargar_datos_yahoo(symbol, start_date, end_date):
    """
    Descarga datos históricos de Yahoo Finance para el símbolo y el rango de fechas dado.
    
    Parameters:
        symbol (str): Símbolo de la acción.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
    
    Returns:
        pd.DataFrame or None: DataFrame con datos históricos o None si hay un error.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)[['Close']]
        return data
    except Exception as e:
        logging.error(f"Error al obtener datos: {str(e)}")
        return None

def calcular_indicadores(data):
    """
    Calcula indicadores técnicos utilizando la biblioteca TA-Lib.
    
    Parameters:
        data (pd.DataFrame): DataFrame con datos históricos.
    
    Returns:
        pd.DataFrame or None: DataFrame con indicadores calculados o None si hay un error.
    """
    try:
        data['MA20'] = talib.SMA(data['Close'], timeperiod=20)
        data['MA50'] = talib.SMA(data['Close'], timeperiod=50)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], _, _ = talib.MACD(data['Close'])
        return data
    except Exception as e:
        logging.error(f"Error al calcular indicadores: {str(e)}")
        return None

def generar_senales(data, rsi_threshold_compra, rsi_threshold_venta):
    """
    Genera señales de compra y venta basadas en condiciones específicas.
    
    Parameters:
        data (pd.DataFrame): DataFrame con datos e indicadores.
        rsi_threshold_compra (float): Umbral de RSI para señales de compra.
        rsi_threshold_venta (float): Umbral de RSI para señales de venta.
    
    Returns:
        pd.DataFrame or None: DataFrame con señales generadas o None si hay un error.
    """
    try:
        condiciones_compra = (data['MA20'] > data['MA50']) & (data['RSI'] < rsi_threshold_compra) & (data['MACD'] > 0)
        condiciones_venta = (data['MA20'] < data['MA50']) & (data['RSI'] > rsi_threshold_venta) & (data['MACD'] < 0)

        data['SenalCompra'] = np.where(condiciones_compra, 1, 0)
        data['SenalVenta'] = np.where(condiciones_venta, -1, 0)
        return data
    except Exception as e:
        logging.error(f"Error al generar señales de compra y venta: {str(e)}")
        return None

def aplicar_estrategia_trading(data, objetivo_porcentaje=2, stop_loss_porcentaje=2, dynamic_target=False, rsi_threshold_compra=30, rsi_threshold_venta=70):
    """
    Aplica la estrategia de trading.
    
    Parameters:
        data (pd.DataFrame): DataFrame con datos e indicadores.
        objetivo_porcentaje (float): Objetivo de ganancia en porcentaje.
        stop_loss_porcentaje (float): Nivel de stop-loss en porcentaje.
        dynamic_target (bool): Indicador de objetivo dinámico.
        rsi_threshold_compra (float): Umbral de RSI para señales de compra.
        rsi_threshold_venta (float): Umbral de RSI para señales de venta.
    
    Returns:
        pd.DataFrame or None: DataFrame con estrategia aplicada o None si hay un error.
    """
    try:
        if dynamic_target:
            # Implementar lógica para ajuste dinámico del objetivo
            # Pueden agregarse algoritmos más avanzados aquí
            pass

        condiciones_compra = (data['MA20'] > data['MA50']) & (data['RSI'] < rsi_threshold_compra) & (data['MACD'] > 0)
        condiciones_venta = (data['MA20'] < data['MA50']) & (data['RSI'] > rsi_threshold_venta) & (data['MACD'] < 0)

        data['SenalCompra'] = np.where(condiciones_compra, 1, 0)
        data['SenalVenta'] = np.where(condiciones_venta, -1, 0)

        data['Senal'] = data['SenalCompra'] + data['SenalVenta']

        data['Objetivo'] = np.round(data['Close'] * (1 + objetivo_porcentaje / 100), 2)
        data['StopLoss'] = np.round(data['Close'] * (1 - stop_loss_porcentaje / 100), 2)

        data['Senal'] = np.where(data['Close'] > data['MA50'], 1, -1)

        return data
    except Exception as e:
        logging.error(f"Error en la estrategia de trading: {str(e)}")
        return None

def obtener_parametros_usuario():
    """
    Obtiene parámetros de usuario mediante la entrada.
    
    Returns:
        tuple or None: Tuple con parámetros de usuario o None si hay un error.
    """
    try:
        symbol = input("Ingrese el símbolo de la acción a analizar (por ejemplo, 'AAPL'): ")
        start_date = input("Ingrese la fecha de inicio en formato 'YYYY-MM-DD': ")
        end_date = input("Ingrese la fecha de fin en formato 'YYYY-MM-DD': ")

        objetivo_porcentaje = float(input("Ingrese el objetivo de ganancia en porcentaje (por ejemplo, 2): "))
        stop_loss_porcentaje = float(input("Ingrese el nivel de stop-loss en porcentaje (por ejemplo, 2): "))
        rsi_threshold_compra = float(input("Ingrese el umbral de RSI para señales de compra (por ejemplo, 30): "))
        rsi_threshold_venta = float(input("Ingrese el umbral de RSI para señales de venta (por ejemplo, 70): "))

        dynamic_target = input("¿Desea habilitar el objetivo dinámico? (y/n): ").lower() == 'y'

        return symbol, start_date, end_date, objetivo_porcentaje, stop_loss_porcentaje, rsi_threshold_compra, rsi_threshold_venta, dynamic_target
    except ValueError as ve:
        logging.error(f"Error al ingresar parámetros: {str(ve)}")
        return None

def seguir_valor_real(symbol):
    """
    Obtiene datos en tiempo real de Yahoo Finance.
    
    Parameters:
        symbol (str): Símbolo de la acción.
    
    Returns:
        pd.DataFrame or None: DataFrame con datos en tiempo real o None si hay un error.
    """
    try:
        data_real = yf.download(symbol, start=pd.to_datetime('today') - pd.DateOffset(days=365), end=pd.to_datetime('today'))[['Close']]
        return data_real
    except Exception as e:
        logging.error(f"Error al seguir el valor real: {str(e)}")
        return None

def generar_matriz_financiera_optimizada(resultados):
    """
    Genera una matriz financiera con información sobre el balance y la ganancia.
    
    Parameters:
        resultados (pd.DataFrame): DataFrame con resultados de la estrategia.
    
    Returns:
        pd.DataFrame or None: DataFrame con matriz financiera optimizada o None si hay un error.
    """
    try:
        matriz_financiera = pd.DataFrame(index=resultados.index)
        matriz_financiera['Balance'] = 0
        matriz_financiera['Ganancia'] = 0

        balance = 0
        senales_compra = resultados['SenalCompra'].values
        senales_venta = resultados['SenalVenta'].values
        precios = resultados['Close'].values

        matriz_financiera['Balance'] = np.cumsum(np.where(senales_compra, -precios, np.where(senales_venta, precios, 0)))
        matriz_financiera['Ganancia'] = balance + np.where(senales_compra, -precios, np.where(senales_venta, precios, 0))

        return matriz_financiera
    except Exception as e:
        logging.error(f"Error al generar matriz financiera: {str(e)}")
        return None

def comparar_resultados_anteriores(resultados, matriz_anterior):
    """
    Compara los resultados actuales con los resultados anteriores.
    
    Parameters:
        resultados (pd.DataFrame): DataFrame con resultados de la estrategia.
        matriz_anterior (pd.DataFrame): DataFrame con matriz financiera anterior.
    
    Returns:
        pd.DataFrame or None: DataFrame con resultados comparados o None si hay un error.
    """
    try:
        if 'Ganancia_Anterior' not in matriz_anterior.columns:
            matriz_anterior['Ganancia_Anterior'] = 0

        resultados['Ganancia_Anterior'] = matriz_anterior['Ganancia']

        return resultados
    except Exception as e:
        logging.error(f"Error al comparar resultados anteriores: {str(e)}")
        return None

def imprimir_resultados(resultados):
    """
    Imprime columnas relevantes de los resultados.
    
    Parameters:
        resultados (pd.DataFrame): DataFrame con resultados de la estrategia.
    """
    try:
        columnas_relevantes = ['Close', 'Senal', 'Objetivo', 'StopLoss', 'Ganancia_Anterior']
        print(resultados[columnas_relevantes])
    except Exception as e:
        logging.error(f"Error al imprimir resultados: {str(e)}")

def evaluar_accion(symbol):
    """
    Agrega lógica para evaluar la acción (análisis técnico, fundamental, etc.).
    
    Parameters:
        symbol (str): Símbolo de la acción.
    """
    try:
        pass  # Puedes agregar tu lógica de evaluación aquí
    except Exception as e:
        logging.error(f"Error al evaluar la acción: {str(e)}")
        return None

def ejecutar_estrategia():
    """
    Ejecuta la estrategia de trading completa.
    
    Returns:
        tuple or None: Tuple con resultados, valor real y matriz financiera o None si hay un error.
    """
    try:
        # Obtener parámetros de usuario
        symbol, start_date, end_date, objetivo_porcentaje, stop_loss_porcentaje, rsi_threshold_compra, rsi_threshold_venta, dynamic_target = obtener_parametros_usuario()

        # Obtener datos históricos
        data = descargar_datos_yahoo(symbol, start_date, end_date)
        if data is None:
            return None, None, None

        # Aplicar indicadores técnicos
        data = calcular_indicadores(data)
        if data is None:
            return None, None, None

        # Generar señales de compra y venta
        data = generar_senales(data, rsi_threshold_compra, rsi_threshold_venta)
        if data is None:
            return None, None, None

        # Aplicar estrategia de trading
        data = aplicar_estrategia_trading(data, objetivo_porcentaje, stop_loss_porcentaje, dynamic_target)
        if data is None:
            return None, None, None

        # Seguir el valor real de la acción
        valor_real = seguir_valor_real(symbol)
        if valor_real is None:
            return None, None, None

        # Generar matriz financiera optimizada
        matriz_financiera = generar_matriz_financiera_optimizada(data)
        if matriz_financiera is None:
            return None, None, None

        # Comparar resultados con resultados anteriores
        resultados = comparar_resultados_anteriores(data, matriz_financiera)

        # Evaluar la acción
        evaluar_accion(symbol)

        return resultados, valor_real, matriz_financiera

    except Exception as e:
        logging.error(f"Error al ejecutar la estrategia: {str(e)}")
        return None, None, None

# Ejecutar la estrategia
resultados, valor_real, matriz_financiera = ejecutar_estrategia()

# Imprimir resultados y matriz financiera
imprimir_resultados(resultados)

print("\nMatriz Financiera:")
print(matriz_financiera)
