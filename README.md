# Proyecto 2: Detección de puntos de referencia con MediaPipe

**Universidad del Valle de Guatemala**  
**Facultad de Ingeniería**  
**Departamento de Ciencias de la Computación**  
**Visión por Computadora**

## Integrantes

- María Ramírez - 21342  
- Gustavo González - 21438  
- Diego Leiva - 21752  
- Pablo Orellana - 21970  

## Guía de Uso

### Descripción

Este proyecto contiene dos scripts principales:

- `pose_detector.py`: Detecta landmarks del cuerpo (poses) en un video y genera un nuevo video con el video original (landmarks sobrepuestos) a la izquierda y un esqueleto (Landmarks solos) sobre fondo negro a la derecha.
- `hand_tracker.py`: Detecta landmarks de manos en tiempo real desde la cámara web.

### Requisitos

Este proyecto incluye:

- Un archivo `requirements.txt` y/o `environment.yml` para instalar dependencias fácilmente.
- Los modelos necesarios dentro de la carpeta `models/`.

**Para crear el entorno en Conda (recomendado):**

```bash
conda env create -f environment.yml
conda activate MP-Tracker
```

**Para usar pip:**

```bash
pip install -r requirements.txt
```

**Modelos necesarios**
Los siguientes modelos ya están incluidos en el repositorio:

- `models/pose_landmarker_heavy.task`
- `models/hand_landmarker.task`

Si quisieras otras versiones de los modelos puedes descargarlos desde:

- [Pose Landmarker Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models)
- [Hand Landmarker Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

### 1. Detección de poses en video

**Uso básico:**

```bash
python pose_detector.py --input input/video.mp4 --output output/resultado.mp4
```

**Parámetros:**

- `--input`: Ruta del video de entrada. Por defecto, se utiliza `input/dancing-test.mp4`. 
- `--output`: Ruta del video de salida. Por defecto, se utiliza `output/pose_tracking_output.mp4.

> [!IMPORTANT]
>
> - El archivo de entrada debe tener extensión `.mp4`.
> - El archivo de salida también debe tener extensión `.mp4`.

### 2. Detección de manos en tiempo real (webcam)

#### Uso básico

```bash
python hand_tracker.py --hands 2
```

#### Parámetros

- `--hands`: número máximo de manos a detectar (entre 1 y 10). Por defecto: `2`.

> [!NOTE]
>
> - Usa la cámara web predeterminada (`cv2.VideoCapture(0)`).
> - Presiona `q` o `ESC` para salir.

#### Codificación de video

- El script usa por defecto el codec `'mp4v'`, que genera archivos `.mp4` más ligeros.
- Para compatibilidad con navegadores, puedes cambiar a `'avc1'`, aunque el archivo final será más pesado.

Dentro de `pose_detector.py`, puedes modificar:

```python
fourcc=cv2.VideoWriter_fourcc(*'mp4v') # => por avc1
```
