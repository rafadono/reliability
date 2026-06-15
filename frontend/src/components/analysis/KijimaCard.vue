<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div class="flex flex-col md:flex-row items-start md:items-center gap-4">
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ $t('charts.kijima.title') }}</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
          </button>
        </div>
      </div>
      <div class="flex gap-2 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
          <option value="">{{ $t('sidebar.all_equip') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <button @click="loadAnalysis" :disabled="loading" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 disabled:opacity-50">
          {{ loading ? $t('sidebar.loading') : $t('charts.weibull.refit') }}
        </button>
      </div>
    </div>

    <div v-show="!isCollapsed">
      <!-- Section info -->
      <p class="text-sm text-gray-600 dark:text-slate-400 mb-4">{{ $t('charts.kijima.desc') }}</p>

      <!-- Advanced filters for Types to Fit and Censored Types -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">{{ $t('charts.weibull.types_to_fit') }}</label>
          <div class="space-y-2 max-h-40 overflow-y-auto bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700">
            <label v-for="t in availableTypes" :key="'fit-'+t" class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-700/50 p-1 rounded">
              <input type="checkbox" :value="t" v-model="typesToFit" :disabled="censoredTypes.includes(t)" class="rounded text-blue-600 focus:ring-blue-500 disabled:opacity-50" />
              <span class="text-sm text-gray-700 dark:text-slate-300" :class="{ 'opacity-50': censoredTypes.includes(t) }">{{ t }}</span>
            </label>
          </div>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">{{ $t('charts.weibull.censored_types') }}</label>
          <div class="space-y-2 max-h-40 overflow-y-auto bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700">
            <label v-for="t in availableTypes" :key="'cen-'+t" class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-700/50 p-1 rounded">
              <input type="checkbox" :value="t" v-model="censoredTypes" :disabled="typesToFit.includes(t)" class="rounded text-orange-600 focus:ring-orange-500 disabled:opacity-50" />
              <span class="text-sm text-gray-700 dark:text-slate-300" :class="{ 'opacity-50': typesToFit.includes(t) }">{{ t }}</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Curve selections -->
      <div class="mb-6 p-4 bg-white dark:bg-slate-800/40 rounded-lg border border-gray-200 dark:border-slate-700">
        <label class="block text-sm font-bold text-gray-800 dark:text-slate-200 mb-2">{{ $t('charts.kijima.models_to_plot') }}</label>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
          <label v-for="opt in modelOptions" :key="opt.id" class="flex items-center gap-2 cursor-pointer p-2 rounded-md hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors">
            <input type="checkbox" v-model="selectedCurves" :value="opt.id" class="rounded focus:ring-blue-500" :style="{ color: opt.color }" />
            <span class="text-sm font-semibold" :style="{ color: opt.color }">{{ opt.label }}</span>
          </label>
        </div>
      </div>

      <!-- Parameter quick details -->
      <div v-if="hasFitData" class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <div v-for="mc in activeFitSummary" :key="mc.name" class="p-4 rounded-lg border shadow-sm" :class="mc.bgClass">
          <p class="text-xs font-bold uppercase tracking-wider mb-1" :style="{ color: mc.color }">{{ mc.name }}</p>
          <div class="space-y-1 text-sm text-gray-800 dark:text-slate-200">
            <div><span class="text-xs opacity-75">β:</span> <strong>{{ mc.beta }}</strong></div>
            <div><span class="text-xs opacity-75">η:</span> <strong>{{ mc.eta }}</strong></div>
            <div v-if="mc.ar !== undefined"><span class="text-xs opacity-75">a_r / a_p:</span> <strong>{{ mc.ar }} / {{ mc.ap }}</strong></div>
            <div v-if="mc.br !== undefined && (mc.br !== 0 || mc.bp !== 0)"><span class="text-xs opacity-75">b_r / b_p:</span> <strong>{{ mc.br }} / {{ mc.bp }}</strong></div>
          </div>
        </div>
      </div>

      <!-- Charts grid -->
      <div v-if="hasFitData" class="space-y-6">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm">
            <h3 class="font-bold text-gray-800 dark:text-white mb-2">{{ $t('charts.weibull.rel_curve') }} R(t)</h3>
            <div class="h-80 relative"><canvas ref="relChartRef"></canvas></div>
          </div>
          <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm">
            <h3 class="font-bold text-gray-800 dark:text-white mb-2">{{ $t('charts.weibull.hazard_rate') }} h(t)</h3>
            <div class="h-80 relative"><canvas ref="hazardChartRef"></canvas></div>
          </div>
        </div>

        <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm">
          <h3 class="font-bold text-gray-800 dark:text-white mb-2">{{ $t('charts.kijima.virtual_age') }} V(t)</h3>
          <div class="h-80 relative"><canvas ref="vAgeChartRef"></canvas></div>
        </div>

        <!-- Comprehensive comparison table -->
        <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm overflow-x-auto">
          <h3 class="font-bold text-gray-800 dark:text-white mb-3">{{ $t('charts.kijima.comparison_table') }}</h3>
          <table class="min-w-full divide-y divide-gray-200 dark:divide-slate-700 text-sm">
            <thead class="bg-gray-50 dark:bg-slate-900/60">
              <tr>
                <th scope="col" class="px-4 py-2 text-left font-semibold text-gray-600 dark:text-slate-300">{{ $t('charts.kijima.model_type') }}</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">β (Forma)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">η (Escala)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">a_r (Correctivo)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">a_p (Preventivo)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">b_r (Pendiente)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">b_p (Pendiente)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">AIC</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">BIC</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">KS p-valor</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">MTBF Estimado</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-slate-700">
              <tr v-for="row in comparisonTableData" :key="row.name" :class="{ 'bg-blue-50/20 dark:bg-blue-900/10': row.active }">
                <td class="px-4 py-2 font-medium" :style="{ color: row.color }">{{ row.name }}</td>
                <td class="px-4 py-2 text-center">{{ row.beta }}</td>
                <td class="px-4 py-2 text-center">{{ row.eta }}</td>
                <td class="px-4 py-2 text-center">{{ row.ar }}</td>
                <td class="px-4 py-2 text-center">{{ row.ap }}</td>
                <td class="px-4 py-2 text-center">{{ row.br }}</td>
                <td class="px-4 py-2 text-center">{{ row.bp }}</td>
                <td class="px-4 py-2 text-center">{{ row.aic }}</td>
                <td class="px-4 py-2 text-center">{{ row.bic }}</td>
                <td class="px-4 py-2 text-center">
                  <span :class="row.p_value >= 0.05 ? 'text-green-600 dark:text-green-400 font-bold' : 'text-red-600 dark:text-red-400'">
                    {{ row.p_value }}
                  </span>
                </td>
                <td class="px-4 py-2 text-center font-semibold">{{ row.mtbf }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <div v-else-if="!loading" class="text-center py-8 text-gray-500 dark:text-slate-400 border border-dashed rounded-lg border-gray-300 dark:border-slate-700">
        No hay datos suficientes para graficar. Haz clic en "Ajustar Curva".
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'
import { Chart } from 'chart.js/auto'

const { t, locale } = useI18n()

defineProps({
  availableEquipment: Array,
  availableTypes: Array
})

const isCollapsed = ref(false)
const loading = ref(false)
const localFilters = ref({ equipment: '' })

const typesToFit = ref([])
const censoredTypes = ref([])

// Active curves to draw
const selectedCurves = ref(['weibull', 'k1_c', 'k2_c', 'k1_td', 'k2_td'])

// Fit results
const weibullResult = ref(null)
const kijimaResult = ref([])

const hasFitData = computed(() => {
  return weibullResult.value !== null || kijimaResult.value.length > 0
})

const modelOptions = computed(() => [
  { id: 'weibull', label: t('charts.kijima.weibull'), color: '#6366f1' },
  { id: 'k1_c', label: t('charts.kijima.k1_c'), color: '#3b82f6' },
  { id: 'k2_c', label: t('charts.kijima.k2_c'), color: '#10b981' },
  { id: 'k1_td', label: t('charts.kijima.k1_td'), color: '#f59e0b' },
  { id: 'k2_td', label: t('charts.kijima.k2_td'), color: '#ef4444' }
])

const activeFitSummary = computed(() => {
  const list = []
  if (selectedCurves.value.includes('weibull') && weibullResult.value?.parameters) {
    list.push({
      name: t('charts.kijima.weibull'),
      beta: weibullResult.value.parameters.beta?.toFixed(3),
      eta: weibullResult.value.parameters.eta?.toFixed(1),
      color: '#6366f1',
      bgClass: 'bg-indigo-50/30 border-indigo-100 dark:bg-indigo-950/10 dark:border-indigo-900/30'
    })
  }

  const namesMap = {
    'Kijima I': { id: 'k1_c', color: '#3b82f6', bg: 'bg-blue-50/30 border-blue-100 dark:bg-blue-950/10 dark:border-blue-900/30' },
    'Kijima II': { id: 'k2_c', color: '#10b981', bg: 'bg-emerald-50/30 border-emerald-100 dark:bg-emerald-950/10 dark:border-emerald-900/30' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b', bg: 'bg-amber-50/30 border-amber-100 dark:bg-amber-950/10 dark:border-amber-900/30' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444', bg: 'bg-rose-50/30 border-rose-100 dark:bg-rose-950/10 dark:border-rose-900/30' }
  }

  kijimaResult.value.forEach(m => {
    const cfg = namesMap[m.model_name]
    if (cfg && selectedCurves.value.includes(cfg.id)) {
      list.push({
        name: m.model_name,
        beta: m.beta?.toFixed(3),
        eta: m.eta?.toFixed(1),
        ar: m.ar?.toFixed(3),
        ap: m.ap?.toFixed(3),
        br: m.br?.toFixed(4),
        bp: m.bp?.toFixed(4),
        color: cfg.color,
        bgClass: cfg.bg
      })
    }
  })

  return list
})

const comparisonTableData = computed(() => {
  const rows = []
  
  // Weibull
  if (weibullResult.value) {
    rows.push({
      name: t('charts.kijima.weibull'),
      beta: weibullResult.value.parameters?.beta?.toFixed(3) || '-',
      eta: weibullResult.value.parameters?.eta?.toFixed(1) || '-',
      ar: '-',
      ap: '-',
      br: '-',
      bp: '-',
      aic: weibullResult.value.goodness_of_fit?.aic?.toFixed(1) || '-',
      bic: weibullResult.value.goodness_of_fit?.bic?.toFixed(1) || '-',
      p_value: '-',
      mtbf: '-', // We can compute it if wanted, but standard Weibull MTBF is in stats
      color: '#6366f1',
      active: selectedCurves.value.includes('weibull')
    })
  }

  const namesMap = {
    'Kijima I': { id: 'k1_c', color: '#3b82f6' },
    'Kijima II': { id: 'k2_c', color: '#10b981' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444' }
  }

  kijimaResult.value.forEach(m => {
    const cfg = namesMap[m.model_name]
    rows.push({
      name: m.model_name,
      beta: m.beta?.toFixed(3) || '-',
      eta: m.eta?.toFixed(1) || '-',
      ar: m.ar?.toFixed(3) || '-',
      ap: m.ap?.toFixed(3) || '-',
      br: m.br?.toFixed(4) || '-',
      bp: m.bp?.toFixed(4) || '-',
      aic: m.AIC?.toFixed(1) || '-',
      bic: m.BIC?.toFixed(1) || '-',
      p_value: m.p_value !== undefined ? m.p_value.toFixed(4) : '-',
      mtbf: m.mean ? m.mean.toFixed(1) + ' hrs' : '-',
      color: cfg?.color || '#94a3b8',
      active: selectedCurves.value.includes(cfg?.id)
    })
  })

  return rows
})

// Chart elements
const relChartRef = ref(null)
const hazardChartRef = ref(null)
const vAgeChartRef = ref(null)

let relChartInstance = null
let hazardChartInstance = null
let vAgeChartInstance = null

watch(selectedCurves, () => {
  renderCharts()
}, { deep: true })

const loadAnalysis = async () => {
  loading.value = true
  try {
    await apiService.setFilters(localFilters.value.equipment)
    const toFit = typesToFit.value.length ? typesToFit.value : null
    const toCens = censoredTypes.value.length ? censoredTypes.value : null

    // Parallel calls
    const [wRes, kRes] = await Promise.all([
      apiService.fitData(undefined, undefined, toFit, toCens, 'TBX'),
      apiService.fitKijima(undefined, undefined, toFit, toCens)
    ])

    if (wRes.data.status === 'success') {
      weibullResult.value = wRes.data
    } else {
      weibullResult.value = null
    }

    if (kRes.data.status === 'success') {
      kijimaResult.value = kRes.data.models
    } else {
      kijimaResult.value = []
    }

    await nextTick()
    renderCharts()
  } catch (err) {
    console.error('Error loading Kijima curves:', err)
  } finally {
    loading.value = false
  }
}

const renderCharts = () => {
  if (!hasFitData.value) return

  renderRelAndHazardCharts()
  renderVAgeChart()
}

const renderRelAndHazardCharts = () => {
  if (!relChartRef.value || !hazardChartRef.value) return
  if (relChartInstance) relChartInstance.destroy()
  if (hazardChartInstance) hazardChartInstance.destroy()

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const relDatasets = []
  const hazardDatasets = []

  // 1. Traditional Weibull
  if (selectedCurves.value.includes('weibull') && weibullResult.value?.reliability_curve) {
    const curve = weibullResult.value.reliability_curve
    const betaStr = weibullResult.value.parameters?.beta?.toFixed(2)
    const etaStr = weibullResult.value.parameters?.eta?.toFixed(1)
    const legendLabel = `Weibull (β=${betaStr}, η=${etaStr})`

    const relPoints = curve.time.map((t, idx) => ({ x: t, y: curve.reliability[idx] }))
    const hazPoints = curve.time.map((t, idx) => ({ x: t, y: curve.hazard_rate[idx] }))

    relDatasets.push({
      label: legendLabel,
      data: relPoints,
      borderColor: '#6366f1',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.1
    })

    hazardDatasets.push({
      label: legendLabel,
      data: hazPoints,
      borderColor: '#6366f1',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.1
    })
  }

  // 2. Kijima Models
  const kNames = {
    'Kijima I': { id: 'k1_c', color: '#3b82f6' },
    'Kijima II': { id: 'k2_c', color: '#10b981' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444' }
  }

  kijimaResult.value.forEach(m => {
    const cfg = kNames[m.model_name]
    if (cfg && selectedCurves.value.includes(cfg.id)) {
      const betaStr = m.beta?.toFixed(2)
      const etaStr = m.eta?.toFixed(1)
      const arStr = m.ar?.toFixed(2)
      const apStr = m.ap?.toFixed(2)
      const brStr = m.br !== 0 ? `, br=${m.br?.toFixed(4)}` : ''
      const bpStr = m.bp !== 0 ? `, bp=${m.bp?.toFixed(4)}` : ''
      const legendLabel = `${m.model_name} (β=${betaStr}, η=${etaStr}, ar=${arStr}, ap=${apStr}${brStr}${bpStr})`

      const relPoints = m.t.map((t, idx) => ({ x: t, y: m.R[idx] }))
      const hazPoints = m.t.map((t, idx) => ({ x: t, y: m.failure_rate[idx] }))

      relDatasets.push({
        label: legendLabel,
        data: relPoints,
        borderColor: cfg.color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      })

      hazardDatasets.push({
        label: legendLabel,
        data: hazPoints,
        borderColor: cfg.color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      })
    }
  })

  // Options for linear x scale
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: textColor,
          font: { size: 10 }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: t('charts.weibull.time') || 'Tiempo',
          color: textColor
        },
        ticks: { color: textColor },
        grid: { color: gridColor }
      },
      y: {
        ticks: { color: textColor },
        grid: { color: gridColor }
      }
    }
  }

  relChartInstance = new Chart(relChartRef.value, {
    type: 'line',
    data: { datasets: relDatasets },
    options: {
      ...chartOptions,
      scales: {
        ...chartOptions.scales,
        y: {
          ...chartOptions.scales.y,
          min: 0,
          max: 1.05,
          title: {
            display: true,
            text: 'R(t)',
            color: textColor
          }
        }
      }
    }
  })

  hazardChartInstance = new Chart(hazardChartRef.value, {
    type: 'line',
    data: { datasets: hazardDatasets },
    options: {
      ...chartOptions,
      scales: {
        ...chartOptions.scales,
        y: {
          ...chartOptions.scales.y,
          title: {
            display: true,
            text: 'h(t)',
            color: textColor
          }
        }
      }
    }
  })
}

const renderVAgeChart = () => {
  if (!vAgeChartRef.value) return
  if (vAgeChartInstance) vAgeChartInstance.destroy()

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const vAgeDatasets = []

  const kNames = {
    'Kijima I': { id: 'k1_c', color: '#3b82f6' },
    'Kijima II': { id: 'k2_c', color: '#10b981' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444' }
  }

  // Generate sawtooth coordinates for each active Kijima model
  kijimaResult.value.forEach(m => {
    const cfg = kNames[m.model_name]
    if (cfg && selectedCurves.value.includes(cfg.id)) {
      const T = m.T
      const V = m.V
      
      const points = []
      if (T && V && T.length > 0 && V.length > 0) {
        points.push({ x: T[0], y: 0 })
        for (let i = 0; i < V.length; i++) {
          const tNext = T[i + 1]
          const xNext = tNext - T[i]
          const vBefore = (i === 0 ? 0 : V[i - 1]) + xNext
          points.push({ x: tNext, y: vBefore })
          // drop/step at the repair/PM event
          points.push({ x: tNext, y: V[i] })
        }
      }

      vAgeDatasets.push({
        label: `${m.model_name} V(t)`,
        data: points,
        borderColor: cfg.color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: cfg.color,
        showLine: true,
        stepped: false, // We manually computed sawtooth steps
        tension: 0
      })
    }
  });

  // Ideal baseline (where ar=ap=1.0 - pure Weibull/no restoration)
  if (vAgeDatasets.length > 0) {
    const maxT = Math.max(...kijimaResult.value.map(m => m.T ? m.T[m.T.length - 1] : 0))
    vAgeDatasets.push({
      label: 'Sin Reparación (V(t) = t)',
      data: [{ x: 0, y: 0 }, { x: maxT, y: maxT }],
      borderColor: '#94a3b8',
      borderDash: [5, 5],
      borderWidth: 1.5,
      pointRadius: 0,
      backgroundColor: 'transparent'
    })
  }

  vAgeChartInstance = new Chart(vAgeChartRef.value, {
    type: 'line',
    data: { datasets: vAgeDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: textColor }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Tiempo Calendario Acumulado',
            color: textColor
          },
          ticks: { color: textColor },
          grid: { color: gridColor }
        },
        y: {
          title: {
            display: true,
            text: 'Edad Virtual V(t)',
            color: textColor
          },
          ticks: { color: textColor },
          grid: { color: gridColor }
        }
      }
    }
  })
}

const handleThemeChange = () => {
  renderCharts()
}

watch(locale, () => {
  renderCharts()
})

onMounted(() => {
  loadAnalysis()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>
