# video2robot

End-to-end pipeline: Video (or Prompt) → Human Pose Extraction → Robot Motion Conversion

```
[Video] → PromptHMR → [SMPL-X] → GMR → [Robot Motion]
```

## Requisitos

- GPU NVIDIA con CUDA 12.8 (probado en RTX 3050 6GB)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git con soporte LFS

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone --recursive https://github.com/josue99999/video2robot_h1_2.git
cd video2robot_h1_2
git submodule update --init --recursive
```

---

### 2. Entorno GMR (retargeting al robot)

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
pip install -e third_party/GMR/
```

---

### 3. Entorno phmr (extracción de pose)

```bash
conda create -n phmr python=3.10 -y
conda activate phmr
```

#### 3.1 Instalar dependencias base

```bash
cd third_party/PromptHMR
pip install -e .
conda install -c conda-forge ffmpeg -y
```

#### 3.2 Instalar extensiones CUDA

> Asegúrate de tener CUDA 12.8 en `/usr/local/cuda-12.8/`. Si tu nvcc por defecto apunta a otra versión, exporta el PATH antes de cada `pip install`:
> ```bash
> export PATH=/usr/local/cuda-12.8/bin:$PATH
> ```
> También necesitas los headers de CUDA de los paquetes pip de NVIDIA:
> ```bash
> export CPATH=$(python -c "
> import os, glob, site
> dirs = []
> for sp in site.getsitepackages():
>     dirs += glob.glob(os.path.join(sp, 'nvidia/*/include'))
> print(':'.join(dirs))
> ")
> ```

```bash
# lietorch (SLAM)
cd pipeline/droidcalib
PATH=/usr/local/cuda-12.8/bin:$PATH pip install lietorch/

# droid_backends_intr (SLAM backends)
PATH=/usr/local/cuda-12.8/bin:$PATH pip install . --config-settings="--build-option=--nvcc-extra-args=-gencode arch=compute_86,code=sm_86"

# Volver al raíz del repo
cd ../../..

# detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# sam2
pip install sam2

# torch_scatter
pip install torch_scatter --no-build-isolation

# xformers
pip install xformers --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

#### 3.3 Librerías CUDA en tiempo de ejecución

Crea el script de activación para que los envs de conda encuentren las librerías de PyTorch:

```bash
mkdir -p /home/$USER/miniconda3/envs/phmr/etc/conda/activate.d/
cat > /home/$USER/miniconda3/envs/phmr/etc/conda/activate.d/torch_libs.sh << 'EOF'
#!/bin/sh
TORCH_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
NVIDIA_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_LIB:${LD_LIBRARY_PATH:-}"
EOF
```

#### 3.4 Parche chumpy (compatibilidad numpy moderno)

```bash
python -c "
import site, os
path = os.path.join(site.getusersitepackages(), 'chumpy/__init__.py')
content = open(path).read()
old = 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf'
new = '''from numpy import nan, inf
import numpy as _np
bool = _np.bool_
int = _np.int_
float = _np.float64
complex = _np.complex128
object = _np.object_
unicode = _np.str_
str = _np.str_'''
open(path, 'w').write(content.replace(old, new))
print('chumpy patched OK')
"
```

---

### 4. Modelos corporales (SMPL / SMPL-X)

#### 4.1 SMPL-X (descarga automática)

```bash
cd third_party/PromptHMR
conda run -n phmr python -m gdown --folder -O ./data/ \
  https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7

conda run -n phmr python -m gdown -O ./data/body_models/smplx/ \
  1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk
```

#### 4.2 SMPL (requiere registro gratuito)

1. Regístrate en https://smpl.is.tue.mpg.de y descarga `SMPL_python_v.1.0.0.zip`
2. Extrae los modelos y crea el neutral promediando male + female:

```bash
mkdir -p third_party/PromptHMR/data/body_models/smpl
unzip -j SMPL_python_v.1.0.0.zip 'smpl/models/*' \
  -d third_party/PromptHMR/data/body_models/smpl/

conda run -n phmr python << 'EOF'
import pickle, numpy as np, scipy.sparse

smpl_dir = 'third_party/PromptHMR/data/body_models/smpl'
with open(f'{smpl_dir}/basicModel_f_lbs_10_207_0_v1.0.0.pkl','rb') as f:
    female = pickle.load(f, encoding='latin1')
with open(f'{smpl_dir}/basicmodel_m_lbs_10_207_0_v1.0.0.pkl','rb') as f:
    male = pickle.load(f, encoding='latin1')

def to_np(x):
    return x.toarray() if scipy.sparse.issparse(x) else np.array(x)

neutral = {}
for k in ['v_template','posedirs','weights','weights_prior','shapedirs']:
    neutral[k] = (to_np(female[k]) + to_np(male[k])) / 2.0
neutral['J_regressor'] = (to_np(female['J_regressor']) + to_np(male['J_regressor'])) / 2.0
for k in ['f','kintree_table','bs_style','bs_type','J']:
    neutral[k] = to_np(female[k]) if hasattr(female[k],'__iter__') else female[k]
neutral['J_regressor_prior'] = (to_np(female['J_regressor_prior']) + to_np(male['J_regressor_prior'])) / 2.0

with open(f'{smpl_dir}/SMPL_NEUTRAL.pkl','wb') as f:
    pickle.dump(neutral, f, protocol=2)
print('SMPL_NEUTRAL.pkl creado OK')
EOF
```

#### 4.3 Symlinks de modelos

```bash
# SMPL-X en PromptHMR → apunta a los de GMR
ln -s $(pwd)/third_party/GMR/assets/body_models/smplx \
      third_party/PromptHMR/data/body_models/smplx

# body_models en GMR → apunta al directorio de assets
ln -s $(pwd)/third_party/GMR/assets/body_models \
      third_party/GMR/assets/body_models
```

> Si los modelos SMPL-X no están en `third_party/GMR/assets/body_models/smplx/`, descárgalos desde https://smpl-x.is.tue.mpg.de y coloca los archivos `.npz` y `.pkl` ahí.

---

### 5. Checkpoints de PromptHMR

```bash
cd third_party/PromptHMR
conda run -n phmr bash scripts/fetch_data.sh
```

---

## Uso

> Los scripts cambian automáticamente al entorno conda correcto. No necesitas activar nada manualmente.

```bash
# Pipeline completo desde un video
python scripts/run_pipeline.py --video /ruta/al/video.mp4

# RTX 3050 / GPUs con poca VRAM — usar siempre --static-camera
python scripts/run_pipeline.py --video /ruta/al/video.mp4 --static-camera

# Elegir robot
python scripts/run_pipeline.py --video /ruta/al/video.mp4 --static-camera --robot unitree_h1

# Continuar desde un proyecto existente (salta pasos ya completados)
python scripts/run_pipeline.py --project data/mi_proyecto --static-camera

# Pipeline desde prompt de texto (requiere GOOGLE_API_KEY)
python scripts/run_pipeline.py --action "Action sequence: The subject walks forward."

# Pasos individuales
python scripts/extract_pose.py --project data/mi_proyecto --static-camera
python scripts/convert_to_robot.py --project data/mi_proyecto --robot unitree_g1

# Visualización
python scripts/visualize.py --project data/mi_proyecto --robot        # MuJoCo
python scripts/visualize.py --project data/mi_proyecto --robot-viser  # navegador
python scripts/visualize.py --project data/mi_proyecto --pose          # pose humana
```

---

## Robots soportados

| Robot | ID | DOF |
|---|---|---|
| Unitree G1 | `unitree_g1` | 29 |
| Unitree G1 + manos | `unitree_g1_with_hands` | — |
| Unitree H1 | `unitree_h1` | 19 |
| Unitree H1-2 | `unitree_h1_2` | — |
| Booster T1 | `booster_t1` | 23 |
| Booster T1 29DOF | `booster_t1_29dof` | 29 |
| Booster K1 | `booster_k1` | — |
| Fourier N1 | `fourier_n1` | — |
| Stanford Toddy | `stanford_toddy` | — |
| EngineAI PM01 | `engineai_pm01` | — |
| Kuavo S45 | `kuavo_s45` | — |
| Galaxea R1 Pro | `galaxea_r1pro` | — |

---

## Formato de salida

```python
# robot_motion.pkl
{
    "fps":       30.0,
    "robot_type": "unitree_g1",
    "num_frames": 300,
    "root_pos":  np.ndarray,  # (N, 3)
    "root_rot":  np.ndarray,  # (N, 4) cuaternión xyzw
    "dof_pos":   np.ndarray,  # (N, DOF)
}
```

---

## Variables de entorno

```bash
GOOGLE_API_KEY   # Para generación de video con Veo
OPENAI_API_KEY   # Para generación de video con Sora
```

---

## Créditos

- [PromptHMR](https://github.com/yufu-wang/PromptHMR) — extracción de pose humana 3D
- [GMR](https://github.com/YanjieZe/GMR) — retargeting de movimiento a robots

## Licencia

- El código principal es MIT.
- PromptHMR es **solo para investigación no comercial**. El uso comercial requiere permiso de sus autores.
