<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">Historical KPI Trends</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? 'Expand ⌄' : 'Collapse ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-500 dark:text-slate-400">Monthly historical trend of key performance indicators.</p>
      </div>
      
      <div class="flex flex-wrap items-center gap-3 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <!-- KPI selector -->
        <div class="flex items-center gap-1">
          <span class="text-xs font-medium text-gray-600 dark:text-slate-400">KPI:</span>
          <select v-model="selectedKpi" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
            <option value="failures">Failure Count</option>
            <option value="downtime">Total Downtime (hrs)</option>
            <option value="mtbf">MTBF (hrs)</option>
            <option value="mttr">MTTR (hrs)</option>
            <option value="availability">Availability (%)</option>
          </select>
        </div>

        <!-- Equipment Filter -->
        <div class="flex items-center gap-1">
          <span class="text-xs font-medium text-gray-600 dark:text-slate-400">Asset:</span>
          <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
            <option value="">All Equipment</option>
            <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
          </select>
        </div>

        <button @click="loadTrendData" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">Update</button>
      </div>
    </div>

    <div v-show="!isCollapsed">
      <div v-if="loading" class="flex justify-center items-center py-12">
        <span class="text-gray-500 dark:text-slate-400 animate-pulse text-sm">Loading historical data...</span>
      </div>
      <div v-else-if="trendData && trendData.length > 0">
        <ChartTrend :data="trendData" :selectedKpi="selectedKpi" :kpiLabel="kpiLabel" />
      </div>
      <div v-else class="text-center py-12 text-gray-500 dark:text-slate-400 text-sm">
        No failure logs matching current filters.
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { apiService } from '../../api'
import ChartTrend from '../ChartTrend.vue'

defineProps({
  availableEquipment: Array
})

const isCollapsed = ref(false)
const selectedKpi = ref('failures')
const localFilters = ref({ equipment: '' })
const trendData = ref([])
const loading = ref(false)

const kpiLabel = computed(() => {
  switch (selectedKpi.value) {
    case 'failures':
      return 'Failure Count'
    case 'downtime':
      return 'Total Downtime (hrs)'
    case 'mtbf':
      return 'MTBF (hrs)'
    case 'mttr':
      return 'MTTR (hrs)'
    case 'availability':
      return 'Availability (%)'
    default:
      return 'Value'
  }
})

const loadTrendData = async () => {
  loading.value = true
  try {
    await apiService.setFilters(localFilters.value.equipment)
    const res = await apiService.getKpiTrend(localFilters.value.equipment)
    trendData.value = res.data.trend || []
  } catch (err) {
    console.error('Error loading trend analysis data:', err)
  } finally {
    loading.value = false
  }
}

// Watch equipment selection changes to automatically update trend
watch(() => localFilters.value.equipment, () => {
  loadTrendData()
})

onMounted(() => {
  loadTrendData()
})
</script>
