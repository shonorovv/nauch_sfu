# GPU-ускорение через CuPy / CUDA

Модель переведена на автоматическое переключение GPU/CPU через файл `backend.py`.  
При запуске: если CuPy и CUDA доступны - расчёт идёт на видеокарте, иначе автоматически на CPU.

---

## Установка CuPy

```powershell
# Узнать версию CUDA:
nvcc --version

# Установить CuPy (заменить cuda12x на свою версию):
pip install cupy-cuda12x

# Проверить:
python -c "import cupy; cupy.array([1.0]) + 1; print('GPU работает')"
```

Поддерживаемые версии: `cupy-cuda11x`, `cupy-cuda12x`.  
Полный список: https://docs.cupy.dev/en/stable/install.html

---

## Как включить/отключить GPU

В `config.py`:

```python
use_gpu = True    # GPU (CuPy) - используется по умолчанию
use_gpu = False   # CPU (NumPy/scipy) - для отладки или если нет CUDA
```

---

## Параметр max_saved_frames

Решает проблему памяти: вместо хранения всех 200 000 временных шагов в памяти  
сохраняется только каждый N-й шаг (максимум `max_saved_frames` кадров).

```python
max_saved_frames = 2000   # ~320 МБ вместо ~32 ГБ при dr=0.05
```

При `dr=0.05`, `dt=0.05`, `t_max=10000`:
- Без прореживания: 20000 × 200001 × 8 байт = 32 ГБ (вылет из памяти)
- С прореживанием (2000 кадров): 20000 × 2001 × 8 байт = 320 МБ

Физически это означает: время сохраняется с шагом ~5 мкс вместо 0.05 мкс.  
Для анализа энтропии, профилей, сравнения сценариев - более чем достаточно.

---

## Архитектура: файл backend.py

Файл `backend.py` автоматически выбирает CuPy или NumPy:

```python
import backend

backend.GPU        # True если CuPy активен
backend.xp         # cupy или numpy - используется для всех вычислений
backend.diags      # cupyx.scipy.sparse.diags или scipy.sparse.diags
backend.factorized # LU-разложение для решателя
backend.to_gpu(arr)  # перенести массив на GPU (или оставить на CPU)
backend.to_cpu(arr)  # вернуть массив с GPU на CPU
```

---

## Что работает на GPU

Горячий контур (200 000 шагов по времени):
- Умножение вектора: `xp.multiply(inv_dt, p_prev, out=rhs)`
- Решение СЛАУ: `solver_fn(rhs)` - LU-факторизация на GPU
- Умножение разреженной матрицы: `matrix.dot(p)` - для проверки сходимости
- Нормировка: `xp.dot(p, dV)`

Предварительные вычисления (один раз, остаются на CPU):
- Геометрическая сетка: `r_values`, `dV`, `face_measures`
- Профили коэффициентов: `D_values`, `Omega_values`
- Построение диагоналей матрицы

---

## Ожидаемое ускорение

| Сетка | n_r | n_t | CPU | GPU | Ускорение |
|---|---|---|---|---|---|
| dr=0.5, dt=0.5 | 2 000 | 20 000 | ~5 с | ~2 с | 2.5× |
| dr=0.1, dt=0.1 | 10 000 | 100 000 | ~10 мин | ~1 мин | 10× |
| dr=0.05, dt=0.05 | 20 000 | 200 000 | ~40 мин | ~3 мин | 15× |

Числа приблизительные - зависят от модели GPU и объёма VRAM.

---

## Диагностика

```python
import cupy
print(f"GPU: {cupy.cuda.Device().id}")
print(f"Версия CUDA: {cupy.cuda.runtime.runtimeGetVersion()}")

mempool = cupy.get_default_memory_pool()
print(f"Использовано VRAM: {mempool.used_bytes() / 1e6:.1f} МБ")
```

---

## Если VRAM не хватает

Уменьши разрешение сетки или количество кадров в `config.py`:

```python
max_saved_frames = 500   # ещё меньше кадров
dr = 0.1                 # более грубая сетка
```

Горячий цикл использует только ~3 × n_r × 8 байт VRAM (текущий p, RHS, dV).  
Для n_r = 20 000 это ~500 КБ - намного меньше любой современной GPU.
