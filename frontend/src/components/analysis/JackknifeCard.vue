<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">Maintenance Jackknife</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? 'Expand ⌄' : 'Collapse ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-500 dark:text-slate-400">Frequency vs Downtime Scatter Plot</p>
      </div>
      <div class="flex flex-wrap items-center gap-2 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <button 
          @click="showExplanation = !showExplanation"
          class="text-xs font-semibold px-2.5 py-1 rounded bg-indigo-50 dark:bg-indigo-950/30 text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 transition-colors"
        >
          {{ showExplanation ? 'Hide Region Guide ⌃' : 'Show Region Guide ⌄' }}
        </button>
        <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500">
          <option value="">All Equipment</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <button @click="loadAnalysis" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">Update</button>
      </div>
    </div>
    
    <div v-show="!isCollapsed">
    
    <!-- Collapsible Region Guides -->
    <div v-show="showExplanation" class="mb-4 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs transition-smooth">
      <div class="bg-red-50 dark:bg-red-950/20 p-2.5 rounded text-red-800 dark:text-red-350 border border-red-100 dark:border-red-900/30">
        <strong class="block mb-0.5">Acute & Chronic (Top Right)</strong>
        High Failure Frequency AND High Downtime. Immediate action required.
      </div>
      <div class="bg-orange-50 dark:bg-orange-950/20 p-2.5 rounded text-orange-800 dark:text-orange-355 border border-orange-100 dark:border-orange-900/30">
        <strong class="block mb-0.5">Acute (Top Left)</strong>
        Low Frequency, but High Downtime. Improve maintainability/MTTR.
      </div>
      <div class="bg-yellow-50 dark:bg-yellow-950/20 p-2.5 rounded text-yellow-800 dark:text-yellow-355 border border-yellow-100 dark:border-yellow-900/30">
        <strong class="block mb-0.5">Chronic (Bottom Right)</strong>
        High Frequency, Low Downtime. Improve reliability/MTBF.
      </div>
      <div class="bg-green-50 dark:bg-green-950/20 p-2.5 rounded text-green-800 dark:text-green-355 border border-green-100 dark:border-green-900/30">
        <strong class="block mb-0.5">Acceptable (Bottom Left)</strong>
        Low Frequency, Low Downtime. Normal operation.
      </div>
    </div>

    <div class="space-y-4 mb-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Plot Nodes By</label>
          <select v-model="compareBy" @change="loadAnalysis" class="input-field w-full">
            <option v-if="!localFilters.equipment" value="equipment">Equipment Level</option>
            <option value="type">Type Level</option>
            <option value="mode">Failure Mode Level</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">X-Axis Scale (Freq)</label>
          <select v-model="scaleX" class="input-field w-full">
            <option value="linear">Linear</option>
            <option value="logarithmic">Logarithmic</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Y-Axis Scale (Downtime)</label>
          <select v-model="scaleY" class="input-field w-full">
            <option value="linear">Linear</option>
            <option value="logarithmic">Logarithmic</option>
          </select>
        </div>
      </div>
    </div>
    
    <ChartJackknife v-if="jackknifeData" :data="jackknifeData" :scaleX="scaleX" :scaleY="scaleY" metricY="total" />

    <!-- Classified Region Tables -->
    <div v-if="classifiedRegions" class="mt-6 border-t border-gray-150 dark:border-slate-700 pt-4">
      <div class="flex justify-between items-center mb-3">
        <h3 class="text-sm font-bold text-gray-900 dark:text-white">Classification Details</h3>
        <button 
          @click="showTableDetails = !showTableDetails"
          class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-355 transition-colors"
        >
          {{ showTableDetails ? 'Hide Tables ⌃' : 'Show Tables ⌄' }}
        </button>
      </div>
      <div v-show="showTableDetails" class="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <!-- Acute & Chronic Table -->
        <div class="rounded border border-red-200 dark:border-red-900/40 overflow-hidden text-xs">
          <div class="bg-red-100 dark:bg-red-950/40 px-3 py-2 font-semibold text-red-900 dark:text-red-300 flex justify-between">
            <span>Acute & Chronic</span>
            <span>Count: {{ classifiedRegions.acuteChronic.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">Freq</th>
                <th class="px-3 py-1.5 text-right">Downtime (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.acuteChronic" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-red-600 dark:text-red-400">{{ item.x }}</td>
                <td class="px-3 py-1.5 text-right">{{ item.y_total.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.acuteChronic.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Acute Table -->
        <div class="rounded border border-orange-200 dark:border-orange-900/40 overflow-hidden text-xs">
          <div class="bg-orange-100 dark:bg-orange-950/40 px-3 py-2 font-semibold text-orange-900 dark:text-orange-300 flex justify-between">
            <span>Acute (High Downtime)</span>
            <span>Count: {{ classifiedRegions.acute.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">Freq</th>
                <th class="px-3 py-1.5 text-right">Downtime (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.acute" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right">{{ item.x }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-orange-600 dark:text-orange-400">{{ item.y_total.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.acute.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Chronic Table -->
        <div class="rounded border border-yellow-250 dark:border-yellow-900/40 overflow-hidden text-xs">
          <div class="bg-yellow-100/80 dark:bg-yellow-950/30 px-3 py-2 font-semibold text-yellow-900 dark:text-yellow-300 flex justify-between">
            <span>Chronic (High Frequency)</span>
            <span>Count: {{ classifiedRegions.chronic.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">Freq</th>
                <th class="px-3 py-1.5 text-right">Downtime (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.chronic" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-yellow-600 dark:text-yellow-400">{{ item.x }}</td>
                <td class="px-3 py-1.5 text-right">{{ item.y_total.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.chronic.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Acceptable Table -->
        <div class="rounded border border-green-200 dark:border-green-900/40 overflow-hidden text-xs">
          <div class="bg-green-100 dark:bg-green-950/40 px-3 py-2 font-semibold text-green-900 dark:text-green-300 flex justify-between">
            <span>Acceptable</span>
            <span>Count: {{ classifiedRegions.acceptable.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">Freq</th>
                <th class="px-3 py-1.5 text-right">Downtime (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.acceptable" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-green-600 dark:text-green-400">{{ item.x }}</td>
                <td class="px-3 py-1.5 text-right">{{ item.y_total.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.acceptable.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
      </div>
      </div>
    </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { apiService } from '../../api'
import ChartJackknife from '../ChartJackknife.vue'

defineProps({
  availableEquipment: Array
})

const isCollapsed = ref(false)
const showTableDetails = ref(false)
const localFilters = ref({ equipment: '' })
const jackknifeData = ref(null)
const compareBy = ref('equipment')
const scaleX = ref('linear')
const scaleY = ref('linear')
const showExplanation = ref(false)

const classifiedRegions = computed(() => {
  if (!jackknifeData.value?.scatter_data) return null
  const avgX = jackknifeData.value.averages?.failures || 0
  const avgY = jackknifeData.value.averages?.total_downtime || 0

  const regions = {
    acuteChronic: [],
    acute: [],
    chronic: [],
    acceptable: []
  }

  jackknifeData.value.scatter_data.forEach(item => {
    const x = item.x
    const y = item.y_total
    if (x > avgX && y > avgY) {
      regions.acuteChronic.push(item)
    } else if (x <= avgX && y > avgY) {
      regions.acute.push(item)
    } else if (x > avgX && y <= avgY) {
      regions.chronic.push(item)
    } else {
      regions.acceptable.push(item)
    }
  })

  return regions
})

const loadAnalysis = async () => {
  try {
    await apiService.setFilters(localFilters.value.equipment)
    const res = await apiService.getJackknifeAnalysis(undefined, undefined, compareBy.value, null)
    jackknifeData.value = res.data
  } catch (err) { console.error('Error loading Jackknife analysis:', err) }
}

// Watch equipment filter to auto switch and hide equipment level compare option
watch(() => localFilters.value.equipment, (newEq) => {
  if (newEq && compareBy.value === 'equipment') {
    compareBy.value = 'type'
  }
  loadAnalysis()
})

onMounted(loadAnalysis)
</script>