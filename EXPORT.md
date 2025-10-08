## Exportar PacPy a otra máquina

Este documento describe dos opciones fiables para trasladar y ejecutar el proyecto PacPy en otra máquina con Windows. Incluye comandos (PowerShell), comprobaciones y soluciones a problemas frecuentes.

### Opciones disponibles
- **Opción A** — Entregar ejecutable precompilado (recomendado para profesores y usuarios finales).
- **Opción B** — Entregar código fuente + entorno virtual (recomendado para reproducibilidad y depuración).

---

## Opción A — Ejecutable precompilado (modo carpeta)

Esta opción es la más sencilla para quien recibirá el juego y no quiere instalar Python.

### 1) Construir el ejecutable (en tu máquina)
Ejecuta desde la raíz del proyecto:

```powershell
# Instalar PyInstaller si no está disponible
pip install pyinstaller

# Ejecutar el script de build incluido
.\build.ps1
```

### 2) Resultado
Tras la ejecución se creará la carpeta `dist\PACMAN` que contiene `PACMAN.exe` y los recursos necesarios.

### 3) Empaquetar para entrega (opcional)

```powershell
Compress-Archive -Path .\dist\PACMAN\* -DestinationPath .\PACMAN_dist.zip -Force
```

### 4) Instrucciones para la máquina destinataria
- Descomprimir `PACMAN_dist.zip` en una carpeta.
- Ejecutar `PACMAN.exe` (doble clic o desde PowerShell):

```powershell
& .\PACMAN.exe
```

### 5) Problemas frecuentes en la máquina destinataria
- El exe se cierra inmediatamente: abre `pacman_data\pacman_error.log` (misma carpeta) y revisa el traceback.
- Fallos de audio o ventana en blanco: verifica drivers de audio y compatibilidad de SDL2 en el equipo.

### 6) Nota sobre `--onefile`
El modo `--onefile` crea un único ejecutable pero puede aumentar falsos positivos de antivirus y tarda más en arrancar (extrae a carpeta temporal). Para entrega y depuración, el modo carpeta es más fiable.

---

## Opción B — Código fuente + entorno virtual

Esta opción es adecuada si el receptor necesita editar o depurar el proyecto.

### 1) Empaquetar el proyecto
Incluye siempre la carpeta `pacman_data` (sonidos, imágenes, settings):

```powershell
Compress-Archive -Path .\* -DestinationPath ..\PacPy_source.zip -Force
```

### 2) En la máquina destinataria (Windows) — pasos mínimos

```powershell
# Descomprimir el proyecto y crear un entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

> `requirements.txt` incluido en el repo contiene las dependencias mínimas (por ejemplo pygame, pyinstaller, Pillow opcional).

### 3) Ejecutar el juego

```powershell
py -3 .\PACMAN.py
```

### 4) Problemas comunes y soluciones
- `pip install pygame` falla: asegúrate de usar una versión de Python compatible con la rueda de Pygame; instala la versión recomendada (ver `requirements.txt`).
- Audio: revisa que el dispositivo de audio y drivers estén operativos.
- El juego se cierra sin consola: abre `pacman_data\pacman_error.log`.

---

## Lista de verificación antes de exportar
- [ ] Incluir la carpeta `pacman_data` con sonidos e imágenes.
- [ ] Incluir `requirements.txt` si entregas código fuente.
- [ ] Probar el `dist\PACMAN\PACMAN.exe` en una máquina de prueba si es posible.
- [ ] Anotar la `seed` usada para ejecuciones que necesiten reproducibilidad.

---

## Consejos prácticos
- Para depuración rápida del ejecutable, ejecuta `PACMAN.exe` desde PowerShell para ver salida/errores, o reconstruye quitando `--windowed` para mostrar la consola.
- Para máxima compatibilidad para entrega al profesor, envía `PACMAN_dist.zip` (modo carpeta).
- Si requieres reproducibilidad absoluta, entrega también `settings.json` y la `seed` usada.

---