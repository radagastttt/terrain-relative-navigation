# TRN — Terrain-Relative Navigation

Прототип алгоритма навигации БПЛА по рельефу местности  
(профиль высот + цифровая модель рельефа + фильтр частиц).

## Установка

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Структура

- `src/` — ядро: TerrainMap, Trajectory, Barometer, TRNParticleFilter
- `scripts/` — запуск экспериментов и диагностики
- `tests/` — unit-тесты
- `config.yaml` — параметры DEM, траектории, фильтра и стадий эксперимента
- `results/` — логи и графики

## Быстрый старт

```bash
# Проверка окружения
python -m scripts.check_env

# Диагностика шага 2 (траектория + измерения)
python -m scripts.check_step2

# Диагностика фильтра (информативность рельефа, сетка параметров, Neff)
python -m scripts.check_step3

# Полный эксперимент по стадиям (ideal / noisy / outliers)
python -m scripts.run_experiment

# Изолированный анализ bias/outliers
python -m scripts.debug_run
```

## Реальный DEM (GeoTIFF)

```bash
# Скачать Copernicus GLO-30 (Централен Балкан, Ботев) и перепроецировать в UTM 35N
# -> data/raw/central_balkan_cop30_utm35n.tif
python -m scripts.download_dem
# через OpenTopography API (нужен бесплатный ключ):
python -m scripts.download_dem --api-key <KEY>

# config.yaml: dem.source: geotiff, dem.path: data/raw/...
python -m scripts.run_experiment --config config.yaml
```

Загрузчик: `src/dem_loader.py` — `load_real_dem(path) -> (dem, transform, crs)`,
`terrain_params_from_transform(transform, shape) -> (origin_xy, dx, dy)`.
Траектория (`trajectory.start_xy`) задаётся в метрах UTM 35N.

Интерактивное окно с картой и профилем (из папки `src`):

```bash
python main.py                                            # диалог выбора файла (Отмена -> DEM из config.yaml)
python main.py --no-dialog                                # DEM из config.yaml (dem.source)
python main.py --dem ../data/raw/pre_balkan_lovech_cop30_utm35n.tif   # любой GeoTIFF
python main.py --synthetic                                # синтетический DEM
```

## Реальный рельеф разных типов

```bash
# DEM трёх районов (Copernicus GLO-30, UTM 35N, 30 м)
python -m scripts.download_dem --bbox 24.45 42.15 24.75 42.35 --out data/raw/thracian_plain_cop30_utm35n.tif
python -m scripts.download_dem --bbox 24.55 43.05 24.85 43.25 --out data/raw/pre_balkan_lovech_cop30_utm35n.tif
python -m scripts.download_dem

python -m scripts.experiment_real_terrain --config config.yaml
# параметры: --runs 30 --sigma0 100 --sigma-meas 1.0 --n-particles 5000
```

В каждом районе — 30 случайных прямых траекторий по 3 км (старт и курс
случайные), начальная ошибка ~N(0, σ0²), σ0 = 100 м, σ_meas = 1 м.
Успех — финальная ошибка < 30 м.
Результаты: `results/figures/real_terrain/`, `results/logs/real_terrain.log`.

| Район | Тип | Высоты, м | Ср. уклон | СКО профиля (мед.) | Успех | final мед. | final p90 |
|---|---|---|---|---|---|---|---|
| Горнотракийская низменность | flat | 154–299 | 0.6° | 2.2 м | 33% | 62 м | 294 м |
| Предбалкан (Ловеч) | hilly | 114–667 | 6.9° | 32.5 м | 93% | 2.9 м | 9.2 м |
| Централен Балкан (Ботев) | mountainous | 320–2378 | 19.8° | 136 м | 97% | 1.4 м | 3.0 м |

Определяющий фактор — изменчивость высоты вдоль траектории: при СКО профиля
> ~10 м привязка надёжна (ошибка единицы метров), при < ~5 м — как правило,
нет. На равнине редкие удачные привязки опираются на микрорельеф
(террасы, насыпи); Copernicus GLO-30 — модель поверхности (DSM) с
постройками и лесом, поэтому на реальном полёте такие детали ненадёжны.

## Устойчивость к начальной ошибке положения

```bash
python -m scripts.experiment_init_error --config config.yaml
# параметры: --sigmas 50 100 200 --runs 20 --n-particles 20000 --sigma-meas 1.0
```

Для каждого σ0 выполняется серия Монте-Карло прогонов (по умолчанию 20).
Центр облака частиц смещён от истинного старта на случайный вектор
~N(0, σ0²) по каждой оси (ошибка ИНС), разброс облака = σ0.
Успех — финальная ошибка < 30 м (1 пиксель DEM).
Результаты: `results/figures/init_error/`, `results/logs/init_error.log`.

Реальный DEM (Централен Балкан), σ_meas = 1 м:

| σ0, м | успех, % (5000 ч.) | final мед., м | успех, % (20000 ч.) | final мед., м |
|-------|--------------------|---------------|---------------------|---------------|
| 50    | 95                 | 1.03          |                     |               |
| 100   | 100                | 0.79          |                     |               |
| 200   | 75                 | 1.52          | 85                  | 1.09          |
| 300   | 80                 | 1.29          |                     |               |
| 500   | 50                 | 29.5          | 75                  | 1.82          |
| 750   | 40                 | 138           |                     |               |
| 1000  | 35                 | 180           | 30                  | 218           |

Неудачи — ложная привязка к похожему участку рельефа (ошибка 150–250 м
и более), а не медленная сходимость: при успехе фильтр сходится за 2–8 шагов.
Увеличение числа частиц помогает до σ0 ≈ 500 м; при σ0 ≈ 1 км
120-секундного профиля уже не хватает, чтобы однозначно выделить положение.

## Вывод о применимости

Полный отчёт с результатами всех экспериментов: [REPORT.md](REPORT.md).

- На холмах и в горах (СКО высоты вдоль трассы > ~10 м) TRN уточняет
  положение с 50–300 м до 1–3 м за 3 км полёта, успех 93–97%.
- На равнине (СКО < ~5 м) метод неприменим: успех ~30%.
- Допустимая начальная ошибка — до ~300 м (5000 частиц) / ~500 м (20000).

## Стадии эксперимента (config.yaml → stages)

| Стадия     | σ_meas | σ_proc | n_particles | Примечание          |
|------------|--------|--------|-------------|---------------------|
| ideal      | 0.2    | 0.5    | 5000        | почти идеальные изм.|
| noisy_0.5  | 0.5    | 0.5    | 5000        | реалистичный шум    |
| noisy_1.0  | 1.0    | 0.5    | 5000        | увеличенный шум     |
| outliers   | 3.0    | 0.5    | 5000        | + 3 искусственных выброса |

Результаты сохраняются в `results/figures/` и `results/logs/run.log`.

## Ожидаемые метрики (типичный прогон)

- ideal: RMSE* ≈ 1.5 м, final ≈ 2 м
- noisy_0.5 / 1.0: RMSE* 1.5–3.5 м
- outliers: RMSE* ≈ 4–5 м (фильтр восстанавливается после выбросов)

На ровной местности (малая std профиля) ошибка растёт — это ожидаемо.
