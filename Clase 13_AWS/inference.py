import joblib

# Datos fijos para prueba
features = [[1, 45, 58, 11, 22, 0]]
# Cargar el modelo
modelo = joblib.load("modelo.pkl")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])
