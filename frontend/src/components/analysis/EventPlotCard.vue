<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">Event Plot Timeline</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? 'Expand ⌄' : 'Collapse ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-500 dark:text-slate-400">Visualize failure events over time for each asset.</p>
      </div>
      <div class="flex flex-wrap items-center gap-3 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        
        <!-- Custom Checkbox Dropdown -->
        <div class="relative" ref="dropdownEl">
          <button 
            @click="isOpen = !isOpen"
            type="button"
            class="text-sm border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded px-3 py-1.5 flex items-center justify-between w-48 text-left focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
          >
            <span class="truncate font-medium">{{ dropdownLabel }}</span>
            <span class="ml-2 text-xs text-gray-500 dark:text-slate-400">▼</span>
          </button>
          
          <div 
            v-show="isOpen"
            class="absolute left-0 mt-1 w-64 bg-white dark:bg-slate-800 border border-gray-350 dark:border-slate-700 rounded shadow-lg max-h-60 overflow-y-auto z-50 p-2 space-y-1 text-sm text-gray-950 dark:text-slate-100"
          >
            <div class="flex justify-between items-center pb-1.5 mb-1.5 border-b border-gray-200 dark:border-slate-700">
              <button 
                @click="selectAll" 
                type="button" 
                class="text-xs text-blue-600 dark:text-blue-400 font-semibold hover:underline"
              >
                Select All
              </button>
              <button 
                @click="clearAll" 
                type="button" 
                class="text-xs text-red-600 dark:text-red-400 font-semibold hover:underline"
              >
                Clear All
              </button>
            </div>
            
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
              >
              <span>{{ eq }}</span>
            </label>
          </div>
        </div>

        <div class="flex items-center gap-1">
          <span class="text-xs font-medium text-gray-600 dark:text-slate-400">From:</span>
          <input 
            type="date" 
            v-model="localFilters.dateFrom" 
            :min="minAvailableDate" 
            :max="localFilters.dateTo || maxAvailableDate" 
            class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500"
          >
        </div>
        <div class="flex items-center gap-1">
          <span class="text-xs font-medium text-gray-600 dark:text-slate-400">To:</span>
          <input 
            type="date" 
            v-model="localFilters.dateTo" 
            :min="localFilters.dateFrom || minAvailableDate" 
            :max="maxAvailableDate" 
            class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500"
          >
        </div>
        <div class="flex items-center gap-1">
          <span class="text-xs font-medium text-gray-600 dark:text-slate-400">Min Dur (hrs):</span>
          <input 
            type="number" 
            v-model.number="minDuration" 
            min="0" 
            step="0.1" 
            class="w-16 text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500"
          >
        </div>
        <button @click="loadAnalysis" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">Update</button>
      </div>
    </div>
    <div v-show="!isCollapsed">
      <ChartEventPlot v-if="filteredEventPlotData" :data="filteredEventPlotData" :date-from="localFilters.dateFrom" :date-to="localFilters.dateTo" :min-duration="minDuration" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { apiService } from '../../api'
import ChartEventPlot from '../ChartEventPlot.vue'

const props = defineProps({
  availableEquipment: Array
})

const isCollapsed = ref(false)
const isOpen = ref(false)
const dropdownEl = ref(null)
const selectedEquipments = ref([])

const localFilters = ref({ dateFrom: '', dateTo: '' })
const minDuration = ref(0.1)
const minAvailableDate = ref('')
const maxAvailableDate = ref('')
const eventPlotData = ref(null)

const selectAll = () => {
  selectedEquipments.value = [...props.availableEquipment]
}

const clearAll = () => {
  selectedEquipments.value = []
}

const dropdownLabel = computed(() => {
  if (selectedEquipments.value.length === 0) {
    return 'None selected'
  }
  if (selectedEquipments.value.length === props.availableEquipment.length) {
    return 'All Equipment'
  }
  return `${selectedEquipments.value.length} Selected`
})

const filteredEventPlotData = computed(() => {
  if (!eventPlotData.value) return null
  const filtered = {}
  for (const eq of selectedEquipments.value) {
    if (eventPlotData.value[eq]) {
      filtered[eq] = eventPlotData.value[eq]
    }
  }
  return filtered
})

watch(() => props.availableEquipment, (newVal) => {
  if (newVal && newVal.length > 0 && selectedEquipments.value.length === 0) {
    selectedEquipments.value = [...newVal]
  }
}, { immediate: true })

const loadAnalysis = async () => {
  try {
    // Clear global equipment filter to fetch all event history
    await apiService.setFilters('')
    const res = await apiService.getEventPlot()
    eventPlotData.value = res.data.events
    
    minAvailableDate.value = res.data.min_date || ''
    maxAvailableDate.value = res.data.max_date || ''
    
    if (!localFilters.value.dateFrom || (minAvailableDate.value && localFilters.value.dateFrom < minAvailableDate.value)) {
      localFilters.value.dateFrom = minAvailableDate.value
    }
    if (!localFilters.value.dateTo || (maxAvailableDate.value && localFilters.value.dateTo > maxAvailableDate.value)) {
      localFilters.value.dateTo = maxAvailableDate.value
    }
  } catch (err) { console.error('Error loading Event Plot:', err) }
}

const handleClickOutside = (event) => {
  if (dropdownEl.value && !dropdownEl.value.contains(event.target)) {
    isOpen.value = false
  }
}

onMounted(() => {
  loadAnalysis()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>