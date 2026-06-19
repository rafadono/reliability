<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ $t('charts.apm.title') }}</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-500 dark:text-slate-400">{{ $t('charts.apm.desc') }}</p>
      </div>
      <div class="flex gap-2 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
          <option value="">{{ $t('charts.kpi.all_equip') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <button @click="loadAnalysis" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">{{ $t('charts.apm.update') }}</button>
      </div>
    </div>

    <!-- Checkbox selector for types -->
    <div v-show="!isCollapsed" class="flex flex-wrap gap-4 items-center mb-6 p-3 bg-gray-50 dark:bg-slate-900/50 rounded-lg border border-gray-200 dark:border-slate-700 text-sm">
      <span class="font-semibold text-gray-700 dark:text-slate-300">Tipos de Detención para MTBF:</span>
      <div class="flex flex-wrap gap-3">
        <label v-for="t in availableTypes" :key="t" class="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" :value="t" v-model="selectedTypes" @change="loadAnalysis" class="rounded text-blue-600 focus:ring-blue-500" />
          <span class="text-gray-700 dark:text-slate-300">{{ t }}</span>
        </label>
      </div>
    </div>

    <div v-show="!isCollapsed" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Bad Actors Table -->
      <div class="overflow-x-auto bg-white dark:bg-slate-800 rounded border border-gray-200 dark:border-slate-700 shadow-sm">
        <table class="min-w-full divide-y divide-gray-200 dark:divide-slate-700 text-sm">
          <thead class="bg-gray-50 dark:bg-slate-900/40">
            <tr>
              <th class="px-4 py-3 text-left font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.apm.equip_type') }}</th>
              <th class="px-4 py-3 text-right font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.apm.failures') }}</th>
              <th class="px-4 py-3 text-right font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.apm.mttr') }}</th>
              <th class="px-4 py-3 text-right font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.apm.mtbf') }}</th>
              <th class="px-4 py-3 text-right font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.apm.availability') }}</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-200 dark:divide-slate-700">
            <tr v-for="actor in badActorsData" :key="actor.name" class="hover:bg-gray-50 dark:hover:bg-slate-700/50">
              <td class="px-4 py-3 font-medium text-gray-900 dark:text-white">{{ actor.name }}</td>
              <td class="px-4 py-3 text-right text-red-600 dark:text-red-400 font-medium">{{ actor.failures }}</td>
              <td class="px-4 py-3 text-right text-orange-600 dark:text-orange-400">{{ actor.mttr.toFixed(1) }}</td>
              <td class="px-4 py-3 text-right text-green-600 dark:text-green-400">{{ actor.mtbf.toFixed(1) }}</td>
              <td class="px-4 py-3 text-right">
                <span :class="actor.availability > 95 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'" class="font-bold">
                  {{ actor.availability.toFixed(2) }}%
                </span>
              </td>
            </tr>
            <tr v-if="!badActorsData || badActorsData.length === 0">
              <td colspan="5" class="px-4 py-8 text-center text-gray-500 dark:text-slate-400">{{ $t('charts.apm.no_data') }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Growth Chart -->
      <div class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm flex flex-col">
        <h3 class="font-bold text-gray-800 dark:text-white mb-1">{{ $t('charts.apm.growth_title') }}</h3>
        <p class="text-xs text-gray-500 dark:text-slate-400 mb-2">{{ $t('charts.apm.growth_desc') }}</p>
        <div class="flex-1 min-h-[250px] relative"><canvas ref="growthChartRef"></canvas></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'
import { Chart } from 'chart.js/auto'

const { t } = useI18n()

const props = defineProps({
  availableEquipment: Array,
  availableTypes: Array
})

const isCollapsed = ref(false)

const localFilters = ref({ equipment: '' })
const selectedTypes = ref([])
const badActorsData = ref([])
const growthChartRef = ref(null)
let growthChartInstance = null

watch(() => props.availableTypes, (newVal) => {
  if (newVal && newVal.length > 0 && selectedTypes.value.length === 0) {
    selectedTypes.value = [...newVal]
  }
}, { immediate: true })

const loadAnalysis = async () => {
  try {
    await apiService.setFilters(localFilters.value.equipment)
    
    const targetTypes = selectedTypes.value.length > 0 ? selectedTypes.value : null
    const [badActorsRes, growthRes] = await Promise.all([
      apiService.getBadActors(undefined, undefined, 'equipment', targetTypes),
      apiService.getGrowthAnalysis(undefined, undefined, targetTypes)
    ])
    
    badActorsData.value = badActorsRes.data.bad_actors || []
    
    if (growthRes.data.cumulative_time) {
      lastGrowthData = growthRes.data
      await nextTick()
      renderGrowthChart(growthRes.data)
    }
  } catch (err) { console.error('Error loading APM analysis:', err) }
}

const renderGrowthChart = (data) => {
  if (!growthChartRef.value) return
  if (growthChartInstance) growthChartInstance.destroy()
  
  const chartData = data.cumulative_time.map((t, index) => ({
    x: t,
    y: data.cumulative_failures[index]
  }))

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  growthChartInstance = new Chart(growthChartRef.value, {
    type: 'line',
    data: {
      datasets: [{
        label: t('charts.apm.cum_failures'),
        data: chartData,
        borderColor: '#059669', backgroundColor: 'rgba(5, 150, 105, 0.1)', fill: true,
        stepped: true,
        pointRadius: 0
      }]
    },
    options: { 
      responsive: true, 
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor } }
      },
      scales: {
        x: {
          type: 'linear',
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: t('charts.apm.cum_time'), color: textColor }
        },
        y: {
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: t('charts.apm.cum_failures'), color: textColor },
          beginAtZero: true
        }
      }
    }
  })
}

// Temporary data store to allow chart redrawing on theme changes
let lastGrowthData = null

const handleThemeChange = () => {
  if (lastGrowthData) {
    renderGrowthChart(lastGrowthData)
  }
}

onMounted(async () => {
  await loadAnalysis()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>