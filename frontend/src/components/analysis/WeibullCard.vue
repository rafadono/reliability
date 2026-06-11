<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div class="flex flex-col md:flex-row items-start md:items-center gap-4">
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ $t('weibull.title') }}</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
          </button>
        </div>
        <!-- Tabs -->
        <div class="flex bg-gray-100 dark:bg-slate-900 p-1 rounded-lg">
          <button 
            @click="activeTab = 'TBX'"
            :class="activeTab === 'TBX' ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white'"
            class="px-4 py-1.5 text-sm font-medium rounded-md transition-colors"
          >
            {{ $t('weibull.reliability') }}
          </button>
          <button 
            @click="activeTab = 'TTX'"
            :class="activeTab === 'TTX' ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white'"
            class="px-4 py-1.5 text-sm font-medium rounded-md transition-colors"
          >
            {{ $t('weibull.maintainability') }}
          </button>
        </div>
      </div>
      <div class="flex gap-2 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
          <option value="">{{ $t('sidebar.all_equip') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <button @click="loadAnalysis" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">{{ $t('weibull.refit') }}</button>
      </div>
    </div>
    
    <div v-show="!isCollapsed">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">{{ $t('weibull.types_to_fit') }}</label>
          <div class="space-y-2 max-h-40 overflow-y-auto bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700">
            <label v-for="t in availableTypes" :key="'fit-'+t" class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-700/50 p-1 rounded">
              <input type="checkbox" :value="t" v-model="typesToFit" :disabled="censoredTypes.includes(t)" class="rounded text-blue-600 focus:ring-blue-500 disabled:opacity-50" />
              <span class="text-sm text-gray-700 dark:text-slate-300" :class="{ 'opacity-50': censoredTypes.includes(t) }">{{ t }}</span>
            </label>
          </div>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">{{ $t('weibull.censored_types') }}</label>
          <div class="space-y-2 max-h-40 overflow-y-auto bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700">
            <label v-for="t in availableTypes" :key="'cen-'+t" class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-700/50 p-1 rounded">
              <input type="checkbox" :value="t" v-model="censoredTypes" :disabled="typesToFit.includes(t)" class="rounded text-orange-600 focus:ring-orange-500 disabled:opacity-50" />
              <span class="text-sm text-gray-700 dark:text-slate-300" :class="{ 'opacity-50': typesToFit.includes(t) }">{{ t }}</span>
            </label>
          </div>
        </div>
      </div>

      <div v-if="currentFitData && currentFitData.parameters">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div class="bg-blue-50 dark:bg-blue-950/20 p-4 rounded-lg border border-blue-100/10">
          <p class="text-gray-600 dark:text-slate-400 text-sm">{{ $t('weibull.beta') }}</p>
          <p class="text-2xl font-bold text-blue-600 dark:text-blue-400">{{ currentFitData.parameters.beta?.toFixed(3) }}</p>
        </div>
        <div class="bg-green-50 dark:bg-green-950/20 p-4 rounded-lg border border-green-100/10">
          <p class="text-gray-600 dark:text-slate-400 text-sm">{{ $t('weibull.eta') }}</p>
          <p class="text-2xl font-bold text-green-600 dark:text-green-400">{{ currentFitData.parameters.eta?.toFixed(1) }}</p>
        </div>
      </div>
      <div v-if="currentFitData.reliability_curve" class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm">
          <h3 class="font-bold text-gray-800 dark:text-white mb-2">{{ activeTab === 'TTX' ? $t('weibull.cum_prob') : $t('weibull.rel_curve') }}</h3>
          <div class="h-64 relative"><canvas ref="weibullChartRef"></canvas></div>
        </div>
        <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm">
          <h3 class="font-bold text-gray-800 dark:text-white mb-2">{{ activeTab === 'TTX' ? $t('weibull.prob_density') : $t('weibull.hazard_rate') }}</h3>
          <div class="h-64 relative"><canvas ref="hazardChartRef"></canvas></div>
        </div>
      </div>
      
      <!-- Proactive Analysis Section -->
      <div v-if="activeTab === 'TBX'" id="proactive-pm-section" class="mt-8 border-t border-gray-200 dark:border-slate-700 pt-6">
        <h3 class="text-lg font-bold text-gray-800 dark:text-white mb-4">{{ $t('weibull.proactive_calc') }}</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
          <!-- Optimal PM -->
          <div class="bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
            <h4 class="font-semibold text-gray-900 dark:text-white mb-3">{{ $t('weibull.opt_pm') }}</h4>
            <div class="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('weibull.pm_cost') }}</label>
                <input type="number" v-model.number="pmCost" class="input-field mt-1 w-full"/>
              </div>
              <div>
                <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('weibull.fail_cost') }}</label>
                <input type="number" v-model.number="failureCost" class="input-field mt-1 w-full"/>
              </div>
            </div>
            <button @click="calculateOptimalPm" class="btn-secondary w-full">{{ $t('weibull.calc_opt') }}</button>
            <div v-if="optimalPmInterval !== ''" class="mt-4 text-center bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 p-3 rounded">
              <p class="text-sm text-gray-600 dark:text-slate-400">{{ $t('weibull.opt_interval') }}</p>
              <p class="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{{ typeof optimalPmInterval === 'number' ? optimalPmInterval.toFixed(1) + ' hrs' : (optimalPmInterval === 'Infinity' ? $t('weibull.na_beta') : $t('weibull.error')) }}</p>
            </div>
          </div>
          <!-- Conditional Reliability -->
          <div class="bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
            <h4 class="font-semibold text-gray-900 dark:text-white mb-3">{{ $t('weibull.cond_rel') }}</h4>
            <div class="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('weibull.curr_age') }}</label>
                <input type="number" v-model.number="currentAge" class="input-field mt-1 w-full"/>
              </div>
              <div>
                <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('weibull.target_age') }}</label>
                <input type="number" v-model.number="missionTime" class="input-field mt-1 w-full"/>
              </div>
            </div>
            <button @click="calculateConditionalReliability" class="btn-secondary w-full">{{ $t('weibull.calc_mission') }}</button>
            <div v-if="conditionalReliability" class="mt-4 text-center bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 p-3 rounded">
              <template v-if="conditionalReliability === 'Error'">
                <p class="text-red-600 font-bold">{{ $t('weibull.calc_error') }}</p>
              </template>
              <template v-else>
                <p class="text-sm text-gray-600 dark:text-slate-400">{{ $t('weibull.success_prob') }}</p>
                <p class="text-2xl font-bold" :class="conditionalReliability.success_probability > 0.9 ? 'text-green-600' : 'text-red-600'">
                  {{ (conditionalReliability.success_probability * 100).toFixed(1) }}%
                </p>
              </template>
            </div>
          </div>
        </div>
      </div>
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
const localFilters = ref({ equipment: '' })
const activeTab = ref('TBX')
const fitDataTBX = ref(null)
const fitDataTTX = ref(null)

const currentFitData = computed(() => activeTab.value === 'TBX' ? fitDataTBX.value : fitDataTTX.value)

const typesToFit = ref([])
const censoredTypes = ref([])

// Proactive analysis state
const pmCost = ref(100)
const failureCost = ref(1000)
const optimalPmInterval = ref('')
const currentAge = ref(0)
const missionTime = ref(100)
const conditionalReliability = ref(null)

const weibullChartRef = ref(null)
const hazardChartRef = ref(null)
let weibullChartInstance = null
let hazardChartInstance = null

watch(activeTab, () => {
  loadAnalysis()
})

const loadAnalysis = async () => {
  try {
    await apiService.setFilters(localFilters.value.equipment)
    const toFit = typesToFit.value.length ? typesToFit.value : null
    const toCens = censoredTypes.value.length ? censoredTypes.value : null
    const res = await apiService.fitData(undefined, undefined, toFit, toCens, activeTab.value)
    
    if (activeTab.value === 'TBX') {
      fitDataTBX.value = res.data
    } else {
      fitDataTTX.value = res.data
    }
    
    if (res.data.reliability_curve) {
      await nextTick()
      renderWeibullChart(res.data.reliability_curve)
    }
  } catch (err) { console.error('Error loading Weibull analysis:', err) }
}

const renderWeibullChart = (curve) => {
  if (!weibullChartRef.value || !hazardChartRef.value) return
  if (weibullChartInstance) weibullChartInstance.destroy()
  if (hazardChartInstance) hazardChartInstance.destroy()
  
  const isMaint = activeTab.value === 'TTX'

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'
  
  weibullChartInstance = new Chart(weibullChartRef.value, {
    type: 'line',
    data: {
      labels: curve.time.map(t => t.toFixed(0)),
      datasets: [{
        label: isMaint ? t('weibull.cum_prob') : t('weibull.rel_curve'),
        data: isMaint ? curve.cdf : curve.reliability,
        borderColor: isMaint ? '#10b981' : '#2563eb', backgroundColor: isMaint ? 'rgba(16, 185, 129, 0.1)' : 'rgba(37, 99, 235, 0.1)', fill: true, pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor } }
      },
      scales: {
        x: { ticks: { color: textColor }, grid: { color: gridColor } },
        y: { ticks: { color: textColor }, grid: { color: gridColor } }
      }
    }
  })

  hazardChartInstance = new Chart(hazardChartRef.value, {
    type: 'line',
    data: {
      labels: curve.time.map(t => t.toFixed(0)),
      datasets: [{
        label: isMaint ? t('weibull.prob_density') : t('weibull.hazard_rate'),
        data: isMaint ? curve.pdf : curve.hazard_rate,
        borderColor: isMaint ? '#f59e0b' : '#ef4444', backgroundColor: isMaint ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)', fill: true, pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor } }
      },
      scales: {
        x: { ticks: { color: textColor }, grid: { color: gridColor } },
        y: { ticks: { color: textColor }, grid: { color: gridColor } }
      }
    }
  })
}

const getApiFilters = () => ({
  equipment: localFilters.value.equipment || undefined,
  failure_type: undefined,
  types_to_fit: typesToFit.value.length ? typesToFit.value : null,
  censored_failure_types: censoredTypes.value.length ? censoredTypes.value : null,
  target_column: 'TBX'
})

const calculateOptimalPm = async () => {
  try {
    const filters = getApiFilters()
    const res = await apiService.getOptimalPm(filters, pmCost.value, failureCost.value)
    optimalPmInterval.value = res.data.optimal_pm_interval === null ? 'Infinity' : res.data.optimal_pm_interval
  } catch (err) {
    console.error(err)
    optimalPmInterval.value = 'Error'
  }
}

const calculateConditionalReliability = async () => {
  try {
    const filters = getApiFilters()
    const res = await apiService.getConditionalReliability(filters, currentAge.value, missionTime.value)
    conditionalReliability.value = res.data
  } catch (err) {
    console.error(err)
    conditionalReliability.value = 'Error'
  }
}

const handleThemeChange = () => {
  if (currentFitData.value && currentFitData.value.reliability_curve) {
    renderWeibullChart(currentFitData.value.reliability_curve)
  }
}

watch(locale, () => {
  if (currentFitData.value && currentFitData.value.reliability_curve) {
    renderWeibullChart(currentFitData.value.reliability_curve)
  }
})

onMounted(() => {
  loadAnalysis()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>