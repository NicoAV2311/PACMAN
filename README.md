PacPy — Build, empaquetado y resolución de problemas

Este repositorio contiene una versión en Python del juego Pac‑Man (archivo principal: `PACMAN.py`) y la carpeta de datos `pacman_data` con recursos (sonidos, configuración, highscores). Este README explica cómo preparar el entorno, generar un ejecutable de Windows con PyInstaller, resolver los problemas más comunes y qué archivos incluir en el repositorio.

## Estructura principal del proyecto

Resumen de los elementos importantes:

- `PACMAN.py` — entrypoint / lógica del juego.
- `pacman_data/` — recursos usados por el juego:
  - `sounds/` — WAVs usados por el juego:
    - `pacman_beginning.wav`
    - `pacman_chomp.wav`
    - `pacman_death.wav`
    - `pacman_eatfruit.wav`
    - `pacman_eatghost.wav`
    - `pacman_extrapac.wav`
    - `pacman_intermission.wav`
  - `highscores.json` — puntuaciones (opcionalmente ignorar en Git).
  - `settings.json` — configuración del juego.
- `build.ps1` — script PowerShell para crear la build con PyInstaller (modo carpeta por defecto).
- `PACMAN.spec` — spec de PyInstaller (ajustado para el proyecto).


## Requisitos

- Python 3.8+ (se usó Python 3.12 en desarrollo; usar la misma versión ayuda a evitar sorpresas).
- pip
- pyinstaller
- Opcional: `gh` (GitHub CLI) para crear repos repositorios desde la terminal.


## Preparar el entorno (Windows / PowerShell)

1) Abrir PowerShell en la carpeta del proyecto (donde está `PACMAN.py`).

2) (Opcional pero recomendado) Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Instalar dependencias:

```powershell
pip install -r requirements.txt
# o al menos:
pip install pygame pyinstaller pillow
```


## Build recomendada (carpeta `dist/`)

El repositorio incluye `build.ps1` que llama a PyInstaller y añade `pacman_data` como datos. Para construir:

```powershell
.\build.ps1
```

Esto generará una carpeta `dist\PACMAN` con `PACMAN.exe` y los assets necesarios.

Si prefieres ejecutarlo manualmente, un comando robusto es:

```powershell
py -3 -m PyInstaller --clean --noconfirm --add-data "pacman_data;pacman_data" --windowed --name PACMAN PACMAN.py
```

Notas:
- `--windowed` suprime la consola. Si estás depurando quítalo o ejecuta el exe desde PowerShell para ver la salida.
- `--onefile` crea un único `.exe` pero puede ralentizar el arranque y dar problemas con antivirus.


## Resolución de problemas comunes

1) PyInstaller falla buscando `numpy\.libs` al usar un `spec` personalizado:

- Causa: el `spec` contiene una entrada hard-coded apuntando a `numpy\.libs` que no existe en esa instalación.
- Solución: elimina esa entrada de `PACMAN.spec` y deja que `collect_all('numpy')` incluya lo necesario. Si necesitas incluir `.libs`, primero confirma su ubicación:

```powershell
py -3 -c "import numpy, os; p=os.path.dirname(numpy.__file__); print(p); print(os.path.exists(os.path.join(p,'.libs')))"
```

Si devuelve `True`, incluye explícitamente la carpeta en la build:

```powershell
py -3 -m PyInstaller --add-data "C:\ruta\a\site-packages\numpy\.libs;numpy/.libs" --collect-all numpy --name PACMAN PACMAN.spec
```

Como alternativa, copia la carpeta `.libs` dentro del proyecto (`./numpy.libs`) y usa `--add-data ".\\numpy.libs;numpy/.libs"`.


2) RuntimeError en numpy / `pygame.sndarray` (por ejemplo "CPU dispatcher tracer already initialized"):

- Causa: conflicto o ausencia de DLLs nativas (MKL/OpenMP) usadas por numpy; pueden quedar duplicadas en el bundle.
- Acciones:
  - Ejecuta el exe desde PowerShell (`& .\dist\PACMAN\PACMAN.exe`) y pega la salida completa aquí.
  - Si el problema es duplicación de DLL (por ejemplo `libiomp5.dll`), hay que identificar qué copia se carga y excluir/copiar la correcta al bundle.


3) El exe se cierra al doble clic sin mensaje:

- Ejecuta desde PowerShell para ver la salida:

```powershell
& .\dist\PACMAN\PACMAN.exe
```

- Además revisa si existe `pacman_data\pacman_error.log` — el juego escribe tracebacks no capturados ahí cuando se ejecuta por doble clic.


4) Sonidos que faltan o no suenan dentro del bundle:

- Asegúrate que `pacman_data\sounds\*.wav` estén presentes dentro de `dist\PACMAN\pacman_data\sounds`.
- Si usas `--onefile`, confirma que `resource_path()` en `PACMAN.py` localiza los archivos extraídos; ejecutar desde PowerShell ayuda a ver errores.


## Git & GitHub — pasos rápidos para subir el proyecto

1) Añade un `.gitignore` razonable (ejemplo mínimo):

```
__pycache__/
dist/
build/
.venv/
*.exe
*.dll
pacman_data/highscores.json
pacman_data/pacman_error.log
```

2) Inicializar repo y hacer commit:

```powershell
git init
git add -A
git commit -m "Initial commit"
```

3A) Si tienes `gh` (GitHub CLI) y estás autenticado, crear y pushear automáticamente:

```powershell
git branch -M main
gh repo create PACMAN --public --source=. --remote=origin --push --confirm
```

3B) Si prefieres crear el repo por la web, crea el repo en GitHub y luego:

```powershell
git branch -M main
git remote add origin https://github.com/<tu_usuario>/PACMAN.git
git push -u origin main
```


## Qué enviar cuando pidas ayuda

Si algo falló en la build o el ejecutable, pega aquí:

- La salida completa de `& .\dist\PACMAN\PACMAN.exe` en PowerShell.
- El contenido de `pacman_data\pacman_error.log` si existe.
- El contenido de `PACMAN.spec` si lo modificaste.

Con esa información puedo darte un parche concreto (añadir `--add-data`, excluir un DLL duplicado, o cambiar flags de PyInstaller).


## Extras y tareas opcionales que puedo hacer por ti

- Modificar `build.ps1` para usar `--onefile` y probar efectos (te avisaré si detecto problemas comunes).
- Añadir un pequeño script `check_build.py` que valide que los archivos esperados existen dentro de `dist/PACMAN` tras la build.
- Crear una `Makefile`/task para builds reproducibles.

Si quieres, aplico alguno de estos cambios ahora.
