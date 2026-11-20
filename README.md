# BatteryGuard_AI — Dashboard + modelo integrados

```
Navegador (Dashboard, Tailwind/Vite :5173)
      │  fetch('/api/predict')   (Vite reenvía /api -> :8000)
      ▼
API FastAPI (backend/api.py :8000)
      │  carga una sola vez: clasificador XGBoost + regresor de ciclos + SHAP + referencias
      ▼
backend/models/*.joblib   <-- los genera backend/src/models/train.py con TU dataset
```

## 1. Primera vez
```bash
npm install
pip install -r backend/requirements.txt
```

## 2. Dale el Dataset a la IA (elige UNA opción)
**A) Por terminal**
```bash
cd backend
# copia tu CSV (nulos ya imputados) a data/processed/baterias_procesado.csv
python src/models/train.py --data data/processed/baterias_procesado.csv --output models/
```
**B) Desde el navegador:** arranca la API (paso 3), abre http://localhost:8000/docs → `POST /api/train` → *Try it out* → sube el CSV. Consulta el avance en `GET /api/salud`.

Revisa en la salida: F1 por clase (sobre todo *Descarte*), el diagnóstico de sobreajuste y el MAE del regresor de ciclos.

## 3. Ejecutar (dos terminales)
```bash
npm run api     # terminal 1 -> API en :8000
npm run dev     # terminal 2 -> Dashboard en :5173
```

## 4. Datos que lee el modelo desde el Dashboard
| Campo del Dashboard | Columna del dataset |
|---|---|
| SOH | battery_health_percent (esencial) |
| Advertencias BMS / Fallos previos / Voltaje anómalo | BMS_warning_count / previous_faults / abnormal_voltage_events (esenciales) |
| Riesgo de fuga térmica | thermal_runaway_risk (esencial) |
| Ciclos, voltaje pack, resistencia | cycle_count, pack_voltage, internal_resistance |
| Temp. máx. celda, eficiencia descarga | cell_temperature_max, discharge_efficiency |
| Año del vehículo, km, química | vehicle_age_years, odometer_km, battery_chemistry |

Si el CSV usa otros nombres, ajusta `buildPayload()` en `Script/dashboard.js`.
Si el modelo no está entrenado o la API está apagada, el Dashboard sigue funcionando con sus reglas originales y avisa.
