<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div class="flex items-center gap-2">
        <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ $t('charts.pareto.title') }}</h2>
        <button 
          @click="isCollapsed = !isCollapsed"
          class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
        >
          {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
        </button>
      </div>
      <div class="flex gap-2 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
          <option value="">{{ $t('sidebar.all_equip') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <select v-model="localFilters.type" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
          <option value="">{{ $t('sidebar.all_types') }}</option>
          <option v-for="t in availableTypes" :key="t" :value="t">{{ t }}</option>
        </select>
        <button @click="loadAnalysis" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">{{ $t('sidebar.apply') }}</button>
      </div>
    </div>
    
    <div v-show="!isCollapsed">
      <div class="space-y-4 mb-4">
        <div class="flex items-center gap-2 mb-2 flex-wrap">
           <span v-if="drilldownEq" class="bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2">
             Drilled Equipment: {{ drilldownEq }}
             <button @click="resetToEquipment" class="hover:text-blue-900 font-bold">&times;</button>
           </span>
           <span v-if="drilldownType" class="bg-purple-100 dark:bg-purple-900/40 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2">
             Drilled Type: {{ drilldownType }}
             <button @click="resetToType" class="hover:text-purple-900 font-bold">&times;</button>
           </span>
        </div>
        <div class="flex items-center justify-between gap-2">
          <select v-model="groupBy" class="input-field flex-1" @change="resetDrilldown">
            <option value="equipment">{{ $t('charts.pareto.group_equip') }}</option>
            <option value="type">{{ $t('charts.pareto.group_type') }}</option>
            <option value="mdf">{{ $t('charts.pareto.group_mode') }}</option>
          </select>
        </div>
      </div>
      <ChartPareto v-if="paretoData" :data="paretoData" @bar-click="handleBarClick" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { apiService } from '../../api'
import ChartPareto from '../ChartPareto.vue'

defineProps({
  availableEquipment: Array,
  availableTypes: Array
})

const isCollapsed = ref(false)

const localFilters = ref({ equipment: '', type: '' })
const paretoData = ref(null)
const drilldownEq = ref(null)
const drilldownType = ref(null)
const groupBy = ref('equipment')

const loadAnalysis = async () => {
  try {
    await apiService.setFilters(localFilters.value.equipment, localFilters.value.type)
    
    const eq = drilldownEq.value || undefined
    const type = drilldownType.value || undefined
    const res = await apiService.getParetoAnalysis(groupBy.value, eq, type)
    paretoData.value = res.data
  } catch (err) { console.error(err) }
}

const handleBarClick = (barName) => {
  if (groupBy.value === 'equipment') {
    drilldownEq.value = barName
    groupBy.value = 'type'
    loadAnalysis()
  } else if (groupBy.value === 'type') {
    drilldownType.value = barName
    groupBy.value = 'mdf'
    loadAnalysis()
  }
}

const resetDrilldown = () => {
  drilldownEq.value = null
  drilldownType.value = null
  loadAnalysis()
}

const resetToEquipment = () => {
  drilldownEq.value = null
  drilldownType.value = null
  groupBy.value = 'equipment'
  loadAnalysis()
}
const resetToType = () => {
  drilldownType.value = null
  groupBy.value = 'type'
  loadAnalysis()
}

onMounted(loadAnalysis)
</script>