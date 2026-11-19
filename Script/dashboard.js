const views = ['datos', 'diagnostico', 'resultado'];
const navButtons = [...document.querySelectorAll('[data-view]')];
const sections = Object.fromEntries(views.map((name) => [name, document.getElementById(`view-${name}`)]));

function showView(name) {
  if (!sections[name]) return;
  views.forEach((view) => {
    const active = view === name;
    sections[view].classList.toggle('is-visible', active);
    sections[view].hidden = !active;
  });
  navButtons.forEach((button) => {
    const active = button.dataset.view === name;
    button.classList.toggle('is-active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  window.scrollTo({ top: 0, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}

function validateForm(form) {
  return form.reportValidity();
}

navButtons.forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
document.querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click', () => {
  const destination = button.dataset.go;
  if (destination === 'diagnostico' && !validateForm(document.getElementById('vehicle-form'))) return;
  showView(destination);
}));

const fastCharge = document.getElementById('fast-charge');
const fastChargeOutput = document.getElementById('fast-charge-output');
fastCharge.addEventListener('input', () => { fastChargeOutput.value = `${fastCharge.value}%`; });

const bmsForm = document.getElementById('bms-form');
const bmsNotice = document.getElementById('bms-notice');
const valueOf = (id) => {
  const raw = document.getElementById(id).value.trim();
  return raw === '' ? null : Number(raw);
};
const formatNumber = (value, maximumFractionDigits = 1) => new Intl.NumberFormat('es-MX', { maximumFractionDigits }).format(value);
const display = (id, value) => { document.getElementById(id).textContent = value ?? 'No disponible'; };

function readDiagnosticData() {
  return {
    soh: valueOf('soh'),
    soc: valueOf('soc'),
    cycles: valueOf('cyc'),
    packVoltage: valueOf('pv'),
    cellVoltage: valueOf('cv'),
    resistance: valueOf('ir'),
    imbalance: valueOf('vi'),
    warnings: valueOf('wc'),
    previousFaults: valueOf('faults'),
    sensorFaults: valueOf('sensor-faults'),
    abnormalVoltageEvents: valueOf('voltage-events'),
    daysSinceService: valueOf('ls'),
    thermalRisk: valueOf('thermal-risk'),
    cellTempMax: valueOf('cell-temp-max'),
    dischargeEff: valueOf('discharge-eff')
  };
}

function updateNotice() {
  const data = readDiagnosticData();
  const warningCount = data.warnings ?? 0;
  const faultCount = (data.previousFaults ?? 0) + (data.sensorFaults ?? 0) + (data.abnormalVoltageEvents ?? 0);
  if (warningCount > 0 || faultCount > 0) {
    bmsNotice.textContent = `Hay ${warningCount} advertencia(s) BMS y ${faultCount} evento(s) de falla/anomalía registrados en los campos completados. Revisa los códigos y su severidad.`;
  } else {
    bmsNotice.textContent = 'Los campos son opcionales. No introduzcas ceros para representar datos desconocidos; deja vacío lo que el equipo no proporcione.';
  }
}

bmsForm.addEventListener('input', updateNotice);

function setResultStatus(title, detail, tone = 'neutral') {
  const container = document.getElementById('result-status');
  container.replaceChildren();
  const dot = document.createElement('span');
  dot.className = 'status-dot';
  if (tone === 'warning') dot.style.background = '#d97706';
  if (tone === 'danger') dot.style.background = '#b8423a';
  if (tone === 'ok') dot.style.background = '#2f8f5b';
  const content = document.createElement('div');
  const heading = document.createElement('b');
  heading.textContent = title;
  const description = document.createElement('small');
  description.textContent = detail;
  content.append(heading, description);
  container.append(dot, content);
}

function setReasons(items) {
  const list = document.getElementById('reason-list');
  list.replaceChildren();
  items.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    list.append(li);
  });
  if (items.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No hay eventos adicionales registrados en los campos disponibles.';
    list.append(li);
  }
}

function updateHealthVisual(soh) {
  const health = soh === null ? 0 : Math.max(0, Math.min(100, soh));
  document.getElementById('health-track-fill').style.width = `${health}%`;
  document.querySelectorAll('.battery-cells i').forEach((cell, index) => {
    const lower = index * 10;
    const upper = (index + 1) * 10;
    cell.classList.toggle('filled', soh !== null && health >= upper);
    cell.classList.toggle('partial', soh !== null && health > lower && health < upper);
  });
}

async function analyzeBattery() {
  if (!validateForm(document.getElementById('vehicle-form'))) {
    showView('datos');
    return;
  }

  if (!validateForm(bmsForm)) return;
  const data = readDiagnosticData();
  const availableValues = Object.values(data).filter((value) => value !== null).length;
  if (availableValues === 0) {
    bmsNotice.textContent = 'Introduce al menos una lectura o registro del BMS para generar el informe.';
    bmsForm.querySelector('input')?.focus();
    return;
  }

  const capacity = valueOf('cap');
  const estimatedCapacity = capacity !== null && data.soh !== null ? capacity * data.soh / 100 : null;
  const vehicleLabel = `${document.getElementById('marca').value} ${document.getElementById('modelo').value}`;
  display('result-vehicle', vehicleLabel);
  display('result-vehicle-year', valueOf('anio') === null ? null : String(valueOf('anio')));
  display('result-odometer', valueOf('odo') === null ? null : `${formatNumber(valueOf('odo'), 0)} km`);
  display('result-chemistry', document.querySelector('input[name="quim"]:checked')?.value ?? null);
  display('result-soh', data.soh === null ? '—' : `${formatNumber(data.soh)}%`);
  display('result-capacity-design', capacity === null ? null : `${formatNumber(capacity, 2)} kWh`);
  display('result-capacity-estimated', estimatedCapacity === null ? null : `≈ ${formatNumber(estimatedCapacity, 2)} kWh`);
  display('result-soc', data.soc === null ? null : `${formatNumber(data.soc)}%`);
  display('result-cycles', data.cycles === null ? null : formatNumber(data.cycles, 0));
  display('result-voltage', data.packVoltage === null ? null : `${formatNumber(data.packVoltage, 2)} V`);
  display('result-cell-voltage', data.cellVoltage === null ? null : `${formatNumber(data.cellVoltage, 3)} V`);
  display('result-resistance', data.resistance === null ? null : `${formatNumber(data.resistance, 1)} mΩ`);
  display('result-imbalance', data.imbalance === null ? null : `${formatNumber(data.imbalance, 2)} mV*`);

  const reasons = [];
  if (data.warnings !== null) reasons.push(`${formatNumber(data.warnings, 0)} advertencia(s) BMS reportada(s).`);
  if (data.previousFaults !== null) reasons.push(`${formatNumber(data.previousFaults, 0)} fallo(s) previo(s) registrado(s).`);
  if (data.sensorFaults !== null) reasons.push(`${formatNumber(data.sensorFaults, 0)} fallo(s) de sensor registrado(s).`);
  if (data.abnormalVoltageEvents !== null) reasons.push(`${formatNumber(data.abnormalVoltageEvents, 0)} evento(s) de voltaje anómalo.`);
  if (data.daysSinceService !== null) reasons.push(`${formatNumber(data.daysSinceService, 0)} días desde el último mantenimiento.`);
  if (data.cycles !== null) reasons.push(`${formatNumber(data.cycles, 0)} ciclos acumulados reportados por el BMS.`);
  setReasons(reasons);

  const faultCount = (data.previousFaults ?? 0) + (data.sensorFaults ?? 0) + (data.abnormalVoltageEvents ?? 0);
  const hasAlerts = (data.warnings ?? 0) > 0 || faultCount > 0;
  const overdueService = data.daysSinceService !== null && data.daysSinceService >= 365;
  let status = 'Sin alertas registradas en los campos disponibles';
  let detail = 'Esto no demuestra por sí solo que la batería esté libre de fallas. Contrasta las lecturas con los límites del fabricante.';
  let tone = 'neutral';
  let recommendation = 'Contrasta las mediciones con la ficha técnica del vehículo y conserva el historial de lecturas. La ausencia de alertas introducidas no equivale a certificar el estado de la batería.';

  if (data.soh !== null && data.soh < 70) {
    status = 'Requiere evaluación de capacidad';
    detail = 'El SOH reportado está por debajo del umbral orientativo del prototipo (70%).';
    tone = 'danger';
    recommendation = 'Realiza una prueba de capacidad/autonomía conforme al procedimiento del fabricante y verifica el SOH con un equipo compatible. No decidas el reemplazo basándote únicamente en este porcentaje.';
  } else if (hasAlerts || overdueService) {
    status = 'Revisión técnica recomendada';
    detail = 'Hay advertencias, eventos registrados o mantenimiento pendiente de contrastar.';
    tone = 'warning';
    recommendation = 'Consulta los códigos de falla, verifica su estado (activo o histórico) y contrasta las mediciones con los límites del fabricante. Programa el mantenimiento correspondiente.';
  } else if (data.soh === null) {
    status = 'Información insuficiente para valorar SOH';
    detail = 'No se proporcionó un SOH reportado por el BMS; no se ha estimado uno de forma independiente.';
    recommendation = 'Obtén una lectura SOH compatible o realiza una prueba de capacidad. VoltCheck no debe inferir el SOH a partir de una sola lectura de voltaje o del conteo de ciclos.';
  }

  setResultStatus(status, detail, tone);
  document.getElementById('recommendation-text').textContent = recommendation;
  document.getElementById('soh-origin').textContent = 'SOH informado por el BMS';
  updateHealthVisual(data.soh);
  applyModel(await queryModel(data), reasons);
  showView('resultado');
}


/* ---------- Conexión con el modelo de IA (backend FastAPI, ver backend/api.py) ---------- */
const MODEL_TEXT = {
  0: 'Óptimo: sigue con tus revisiones habituales y vuelve a analizar si cambian las lecturas.',
  1: 'Mantenimiento preventivo: agenda una revisión técnica, verifica los códigos del BMS y contrasta las mediciones con el fabricante.',
  2: 'Descarte: el modelo detecta señales de falla. Confirma con una prueba de capacidad y una inspección profesional antes de decidir el reemplazo.'
};

// Nombres de columna del dataset <- campos del Dashboard (null = dato no disponible)
function buildPayload(d) {
  const year = valueOf('anio');
  const chem = document.querySelector('input[name="quim"]:checked')?.value;
  return {
    battery_health_percent: d.soh, cycle_count: d.cycles, pack_voltage: d.packVoltage,
    internal_resistance: d.resistance, BMS_warning_count: d.warnings, previous_faults: d.previousFaults,
    abnormal_voltage_events: d.abnormalVoltageEvents, thermal_runaway_risk: d.thermalRisk,
    cell_temperature_max: d.cellTempMax, discharge_efficiency: d.dischargeEff,
    vehicle_age_years: year === null ? null : new Date().getFullYear() - year,
    odometer_km: valueOf('odo'),
    battery_chemistry: chem && chem !== 'Desconocida' ? chem : null,
    state_of_charge: d.soc, cell_voltage_avg: d.cellVoltage, voltage_imbalance: d.imbalance,
    sensor_fault_count: d.sensorFaults, last_service_days: d.daysSinceService,
    battery_capacity_kwh: valueOf('cap'), daily_distance: valueOf('dist'),
    charging_cycles_last_month: valueOf('ciclos'),
    fast_charge_ratio: Number(fastCharge.value) / 100,
    fleet_or_private: document.querySelector('input[name="uso"]:checked')?.value === 'flotilla' ? 'Fleet' : 'Private'
  };
}

async function queryModel(data) {
  try {
    const res = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(buildPayload(data))
    });
    const body = await res.json();
    if (!res.ok) return { ok: false, error: body.detail ?? `Error ${res.status}` };
    return { ok: true, data: body };
  } catch {
    return { ok: false, error: 'no se pudo conectar con el servidor (¿está corriendo la API en el puerto 8000?).' };
  }
}

function applyModel(result, readouts) {
  const note = document.getElementById('model-note');
  const card = document.getElementById('life-card');
  if (!result.ok) {
    card.hidden = true;
    note.textContent = `Modelo de IA no disponible: ${result.error} Se muestran solo las reglas orientativas del prototipo.`;
    return;
  }
  const m = result.data;
  const pct = (m.confianza * 100).toFixed(1);
  setResultStatus(`${m.etiqueta} · ${pct}% de confianza`, `Fiabilidad ${m.fiabilidad.toLowerCase()}. Datos proporcionados: ${m.campos_proporcionados}.`, { 0: 'ok', 1: 'warning', 2: 'danger' }[m.clase_predicha]);
  document.getElementById('recommendation-text').textContent = MODEL_TEXT[m.clase_predicha];
  const factors = m.explicacion.map((e) => `Modelo · ${e.variable} = ${formatNumber(Number(e.valor), 2)} (${e.origen}): ${e.efecto}, ${e.impacto_relativo_pct}% de influencia.`);
  setReasons([...factors, ...m.advertencias.map((a) => `Aviso · ${a}`), ...readouts]);
  const [lo, hi] = m.ciclos_rango;
  document.getElementById('life-cycles').textContent = `≈ ${formatNumber(m.ciclos_restantes, 0)}`;
  document.getElementById('life-fill').style.width = `${Math.max(0, Math.min(100, ((m.ciclos_restantes - lo) / (hi - lo)) * 100))}%`;
  document.getElementById('life-note').textContent = m.clase_predicha === 2
    ? 'Descarte se decide por señales de falla (temperatura, riesgo térmico, fallos), no por los ciclos restantes. Esta cifra es solo una estimación estadística.'
    : `Estimación estadística sobre el rango del entrenamiento (${formatNumber(lo, 0)}–${formatNumber(hi, 0)} ciclos). No es una garantía.`;
  card.hidden = false;
  note.textContent = '';
}


/* ---------- Indicadores de referencia bajo cada campo (datos reales del dataset, GET /api/ayudas) ---------- */
const CLASS_UI = { 'Óptimo': ['Óptimo', 'ok'], 'Mantenimiento preventivo': ['Mantenimiento', 'warn'], 'Descarte': ['Descarte', 'bad'] };
const hintNumber = (v) => formatNumber(v, Math.abs(v) >= 100 ? 0 : 1);

async function loadHints() {
  try {
    const res = await fetch('/api/ayudas');
    if (!res.ok) return;
    const ayudas = await res.json();
    document.querySelectorAll('input[data-col]').forEach((input) => {
      const info = ayudas[input.dataset.col];
      if (!info) return;
      const hint = document.createElement('small');
      hint.className = 'hint';
      const title = document.createElement('b');
      title.textContent = Object.keys(info.por_clase).length ? 'Valores típicos en el dataset' : 'Referencia del dataset';
      hint.append(title);
      Object.entries(info.por_clase).forEach(([clase, [lo, hi]]) => {
        const [label, css] = CLASS_UI[clase] ?? [clase, 'ok'];
        const row = document.createElement('span');
        row.className = `hint-row ${css}`;
        row.textContent = `${label}: ${hintNumber(lo)}–${hintNumber(hi)}`;
        hint.append(row);
      });
      if (!Object.keys(info.por_clase).length) {
        const med = document.createElement('span');
        med.className = 'hint-row';
        med.textContent = `Valor mediano: ${hintNumber(info.mediana)}`;
        hint.append(med);
      }
      const total = document.createElement('span');
      total.className = 'hint-total';
      total.textContent = `Rango total: ${hintNumber(info.min)}–${hintNumber(info.max)}`;
      if (info.separa === false) {
        const nota = document.createElement('span');
        nota.className = 'hint-total';
        nota.textContent = 'Por sí solo no distingue el estado de la batería; el modelo lo combina con otras lecturas.';
        hint.append(nota);
      }
      const flag = document.createElement('span');
      flag.className = 'hint-flag';
      flag.textContent = 'Fuera del rango visto en el entrenamiento: revisa el valor y su unidad.';
      hint.append(total, flag);
      input.closest('.field').append(hint);
      input.addEventListener('input', () => {
        const v = input.value.trim() === '' ? null : Number(input.value);
        hint.classList.toggle('is-out', v !== null && (v < info.min || v > info.max));
      });
    });
  } catch { /* sin API: el Dashboard funciona sin indicadores */ }
}
loadHints();

document.getElementById('analyze-button').addEventListener('click', analyzeBattery);
document.getElementById('recalculate-button').addEventListener('click', analyzeBattery);
document.getElementById('reset-vehicle').addEventListener('click', () => {
  window.setTimeout(() => { fastChargeOutput.value = `${fastCharge.value}%`; }, 0);
});

updateNotice();
