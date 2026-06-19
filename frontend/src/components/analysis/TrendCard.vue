<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ $t('charts.kpi.title') }}</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-500 dark:text-slate-400">{{ $t('charts.kpi.desc') }}</p>
      </div>
      
      <div class="flex flex-wrap items-center gap-3 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <!-- KPI selector -->
        <div class="flex items-center gap-1">
          <span class="text-xs font-semibold text-gray-600 dark:text-slate-400">{{ $t('charts.kpi.kpi_label') }}</span>
          <select v-model="selectedKpi" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500 py-1">
            <option value="failures">{{ $t('charts.kpi.metric_failures') }}</option>
            <option value="downtime">{{ $t('charts.kpi.metric_downtime') }}</option>
            <option value="mtbf">{{ $t('charts.kpi.metric_mtbf') }}</option>
            <option value="mttr">{{ $t('charts.kpi.metric_mttr') }}</option>
            <option value="availability">{{ $t('charts.kpi.metric_availability') }}</option>
          </select>
        </div>

        <!-- Equipment Filter (Multi-select) -->
        <div class="relative flex items-center gap-1" ref="dropdownEl">
          <span class="text-xs font-semibold text-gray-600 dark:text-slate-400">{{ $t('charts.kpi.asset_label') }}</span>
          <div class="relative">
            <button 
              @click="showEqDropdown = !showEqDropdown"
              type="button"
              class="text-sm border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded px-3 py-1 flex items-center justify-between w-48 text-left focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            >
              <span class="truncate font-medium">{{ selectedEqLabel }}</span>
              <span class="ml-2 text-xs text-gray-500 dark:text-slate-400">▼</span>
            </button>
            
            <div 
              v-show="showEqDropdown"
              class="absolute left-0 mt-1 w-56 bg-white dark:bg-slate-800 border border-gray-300 dark:border-slate-700 rounded shadow-lg max-h-60 overflow-y-auto z-50 p-2 space-y-1 text-sm text-gray-950 dark:text-slate-100"
            >
              <label class="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-slate-700/60 cursor-pointer select-none text-xs font-semibold border-b border-gray-100 dark:border-slate-700 pb-1.5 mb-1">
                <input 
                  type="checkbox" 
                  :checked="isAllSelected" 
                  @change="toggleAllEquipment"
                  class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer bg-white dark:bg-slate-900"
                />
                <span>{{ $t('charts.kpi.all_equip') }}</span>
              </label>
              
              <label 
                v-for="eq in availableEquipment" 
                :key="eq" 
                class="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-slate-700/60 cursor-pointer select-none text-xs"
              >
                <input 
                  type="checkbox" 
                  :value="eq" 
                  v-model="selectedEquipments"
                  class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer bg-white dark:bg-slate-900"
                />
                <span>{{ eq }}</span>
              </label>
            </div>
          </div>
        </div>

        <button @click="loadTrendData" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">{{ $t('charts.kpi.update') }}</button>
      </div>
    </div>

    <!-- Checkbox selector for types -->
    <div v-show="!isCollapsed" class="flex flex-wrap gap-4 items-center mb-6 p-3 bg-gray-50 dark:bg-slate-900/50 rounded-lg border border-gray-200 dark:border-slate-700 text-sm">
      <span class="font-semibold text-gray-700 dark:text-slate-300">Tipos de Detención para KPI:</span>
      <div class="flex flex-wrap gap-3">
        <label v-for="t in availableTypes" :key="t" class="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" :value="t" v-model="selectedTypes" @change="loadTrendData" class="rounded text-blue-600 focus:ring-blue-500" />
          <span class="text-gray-700 dark:text-slate-300">{{ t }}</span>
        </label>
      </div>
    </div>

    <div v-show="!isCollapsed">
      <div v-if="loading" class="flex justify-center items-center py-12">
        <span class="text-gray-500 dark:text-slate-400 animate-pulse text-sm">{{ $t('charts.kpi.loading') }}</span>
      </div>
      <div v-else-if="trendData && Object.keys(trendData).length > 0">
        <ChartTrend :data="trendData" :selectedKpi="selectedKpi" :kpiLabel="kpiLabel" />
      </div>
      <div v-else class="text-center py-12 text-gray-500 dark:text-slate-400 text-sm">
        {{ $t('charts.kpi.no_data') }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'
import ChartTrend from '../ChartTrend.vue'

const { t } = useI18n()

const props = defineProps({
  availableEquipment: Array,
  availableTypes: Array
})

const isCollapsed = ref(false)
const selectedTypes = ref([])

watch(() => props.availableTypes, (newVal) => {
  if (newVal && newVal.length > 0 && selectedTypes.value.length === 0) {
    selectedTypes.value = [...newVal]
  }
}, { immediate: true })
const selectedKpi = ref('failures')
const showEqDropdown = ref(false)
const selectedEquipments = ref([])
const dropdownEl = ref(null)
const trendData = ref({})
const loading = ref(false)

const isAllSelected = computed(() => {
  return props.availableEquipment && selectedEquipments.value.length === props.availableEquipment.length
})

const toggleAllEquipment = () => {
  if (isAllSelected.value) {
    selectedEquipments.value = []
  } else {
    selectedEquipments.value = props.availableEquipment ? [...props.availableEquipment] : []
  }
}

const selectedEqLabel = computed(() => {
  if (!selectedEquipments.value || selectedEquipments.value.length === 0) {
    return t('charts.kpi.all_equip') || 'All Equipment'
  }
  if (props.availableEquipment && selectedEquipments.value.length === props.availableEquipment.length) {
    return t('charts.kpi.all_equip') || 'All Equipment'
  }
  if (selectedEquipments.value.length > 2) {
    return `${selectedEquipments.value.length} selected`
  }
  return selectedEquipments.value.join(', ')
})

const kpiLabel = computed(() => {
  switch (selectedKpi.value) {
    case 'failures':
      return t('charts.kpi.metric_failures')
    case 'downtime':
      return t('charts.kpi.metric_downtime')
    case 'mtbf':
      return t('charts.kpi.metric_mtbf')
    case 'mttr':
      return t('charts.kpi.metric_mttr')
    case 'availability':
      return t('charts.kpi.metric_availability')
    default:
      return t('charts.kpi.value')
  }
})

const loadTrendData = async () => {
  loading.value = true
  try {
    const targetEqs = selectedEquipments.value.length > 0 ? selectedEquipments.value : null
    const targetTypes = selectedTypes.value.length > 0 ? selectedTypes.value : null
    const res = await apiService.getKpiTrend(targetEqs, undefined, targetTypes)
    trendData.value = res.data.trends || {}
  } catch (err) {
    console.error('Error loading trend analysis data:', err)
  } finally {
    loading.value = false
  }
}

const handleClickOutside = (event) => {
  if (dropdownEl.value && !dropdownEl.value.contains(event.target)) {
    showEqDropdown.value = false
  }
}

watch(() => props.availableEquipment, (newVal) => {
  if (newVal && newVal.length > 0 && selectedEquipments.value.length === 0) {
    selectedEquipments.value = [...newVal]
  }
}, { immediate: true })

// Watch selected equipments or KPI to load new trend data
watch(selectedKpi, () => {
  loadTrendData()
})

onMounted(() => {
  loadTrendData()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>
