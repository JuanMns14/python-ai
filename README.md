#  Prueba de Programación para Desarrollador - Python - IA

## 12. Describe cómo entrenar un modelo simple usando scikit-learn.

Para entrenar un modelo usando scikit-learn implica varios pasos:

1. Instalar y cargar bibliotecas necesarias

    Primero, necesitas asegurarte de que tienes instalado `scikit-learn` y `pandas`. 
    
    Si no los tienes instalados, puedes hacerlo con los siguientes comandos:
    ```bash
    pip install scikit-learn pandas
    ```
    Luego, en el script de Python, importa las bibliotecas necesarias:
    ```py
    import string
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn.svm import LinearSVC 
    ```

2. Cargar y preparar los datos

    Se carga un archivo CSV llamado `spam.csv` que contiene los datos. El archivo `spam.csv` contiene mensajes etiquetados como `spam` o `ham`. 

    ```py
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
    df.columns = ["label", "text"]
    ```
    `encoding='latin-1'` se usa para evitar problemas con caracteres especiales en el archivo CSV.

    Se seleccionan solo las columnas necesarias y se renombraron a `label` (etiqueta, `spam` o `ham`) y `text` (mensaje).

3. Tokenización de texto

    Se define una función para tokenizar el texto, eliminando la puntuación y convirtiendo todo a minúsculas:

    ```py
    punctuation = set(string.punctuation)

    def tokenize(sentence):
        tokens = []
        for token in sentence.split():
            new_token = []
            for char in token:
                if char not in punctuation:
                    new_token.append(char.lower())
            if new_token:
                tokens.append("".join(new_token))
        return tokens
    ```
    La función `tokenize` toma un `sentence`, elimina la puntuación y convierte los caracteres a minúsculas.

4. Dividir los datos en entrenamiento y prueba

    Se divide el conjunto de datos en dos: uno para entrenamiento y otro para prueba:
    ```py
    train_text, test_text, train_labels, test_labels = train_test_split(
        df["text"], df["label"], stratify=df["label"]
    )
    print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")
    ```
    `train_test_split` divide los datos de manera estratificada, lo que significa que mantiene la proporción de etiquetas (`spam` y `ham`) en los conjuntos de entrenamiento y prueba.

5. Vectorización del texto
    El texto debe ser convertido a una representación numérica para que pueda ser utilizado por el modelo. Esto se hace usando `CountVectorizer`:
    ```py
    vectorizer = CountVectorizer(tokenizer=tokenize, binary=True, token_pattern=None)
    train_X = vectorizer.fit_transform(train_text)
    test_X = vectorizer.transform(test_text)
    ```
    `CountVectorizer` convierte el texto a una matriz de conteo de palabras, utilizando la función `tokenize` definida anteriormente.

    `binary=True` asegura que se utilice una representación binaria, donde cada palabra está representada como `0` o `1` dependiendo de si está presente en el texto.

    Si `tokenizer` está definido, `CountVectorizer` ignorará `token_pattern`. Sin embargo, si no se especifica `token_pattern=None`, scikit-learn generará una advertencia para informarte que `token_pattern` será ignorado porque `tokenizer` está definido. Configurarlo a None evita esta advertencia.

6. Entrenamiento del modelo

    Se utiliza un clasificador de máquina de vectores de soporte lineal (LinearSVC) para entrenar el modelo:
    ```py
    classifier = LinearSVC()
    classifier.fit(train_X, train_labels)
    ```
    `LinearSVC` es un clasificador que intenta encontrar una línea que separe los datos de diferentes clases (en este caso, `spam` y `ham`) en un espacio de alta dimensión.

7. Hacer predicciones y evaluar el modelo

    Una vez entrenado, el modelo se utiliza para hacer predicciones en el conjunto de prueba y se evalúa su precisión:
    ```py
    predictions = classifier.predict(test_X)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4%}")
    ```
    `accuracy_score` calcula la precisión del modelo, el porcentaje de etiquetas predichas correctamente.

8. Predicciones con nuevos ejemplos

    Se utilizan nuevas frases para probar el modelo y ver si las clasifica correctamente como spam o no spam:
    ```py
    spam = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize!"
    ham = "Can we reschedule our call to Monday morning? I'm tied up with meetings today."
    examples = [spam, ham]

    examples_X = vectorizer.transform(examples)
    predictions = classifier.predict(examples_X)

    for text, label in zip(examples, predictions):
    print(f"{label:2} - {text}")
    ```
    Se transforman las nuevas frases usando el mismo `vectorizer` y 
    se predice su categoría con el clasificador entrenado.

    Se imprimen las predicciones junto con los textos originales.

## 13. ¿Qué tipos de problemas se pueden resolver con redes neuronales en TensorFlow?

Las redes neuronales en TensorFlow pueden resolver una amplia variedad de problemas, incluyendo:

- Clasificación de Imágenes: Identificar y clasificar objetos en fotos.
- Reconocimiento de Voz y NLP: Convertir audio en texto, traducir idiomas y analizar sentimientos.
- Predicción de Series Temporales: Pronosticar ventas, precios de acciones y datos climáticos.
- Regresión: Estimar precios de viviendas o predecir variables continuas.
- Diagnóstico Médico: Detectar enfermedades a partir de imágenes médicas.
- Generación de Imágenes y Texto: Crear imágenes realistas y generar texto coherente.
- Sistemas de Recomendación: Sugerir productos o contenido según las preferencias del usuario.
- Detección de Anomalías: Identificar fraudes o comportamientos anómalos en datos.

TensorFlow es eficaz para manejar grandes datos y realizar cálculos complejos, adaptándose a diversos campos y aplicaciones.
