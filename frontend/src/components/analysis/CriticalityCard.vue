<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">Criticality Matrix</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? 'Expand ⌄' : 'Collapse ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-550 dark:text-slate-400">Frequency vs Average Downtime (Consequence)</p>
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
      <div class="bg-red-50 dark:bg-red-950/20 p-2.5 rounded text-red-800 dark:text-red-300 border border-red-100 dark:border-red-900/30">
        <strong class="block mb-0.5">High Risk (Top Right)</strong>
        High Frequency AND High MTTR. Severe impact on maintainability.
      </div>
      <div class="bg-orange-50 dark:bg-orange-950/20 p-2.5 rounded text-orange-800 dark:text-orange-355 border border-orange-100 dark:border-orange-900/30">
        <strong class="block mb-0.5">High Consequence (Top Left)</strong>
        Low Frequency, but High MTTR. Difficult to repair when it fails.
      </div>
      <div class="bg-yellow-50 dark:bg-yellow-950/20 p-2.5 rounded text-yellow-800 dark:text-yellow-355 border border-yellow-100 dark:border-yellow-900/30">
        <strong class="block mb-0.5">High Frequency (Bottom Right)</strong>
        High Frequency, Low MTTR. Chronic issues, easy but annoying repairs.
      </div>
      <div class="bg-green-50 dark:bg-green-950/20 p-2.5 rounded text-green-800 dark:text-green-355 border border-green-100 dark:border-green-900/30">
        <strong class="block mb-0.5">Low Risk (Bottom Left)</strong>
        Low Frequency, Low MTTR. Normal operational wear.
      </div>
    </div>

    <div class="space-y-4 mb-4">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Plot Nodes By</label>
          <select v-model="compareBy" @change="loadAnalysis" class="input-field w-full">
            <option v-if="!localFilters.equipment" value="equipment">Equipment Level</option>
            <option value="type">Type Level</option>
            <option value="mode">Failure Mode Level</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">X-Axis Metric</label>
          <select v-model="metricX" class="input-field w-full">
            <option value="count">Failure Count</option>
            <option value="probability">Probability (%)</option>
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
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Y-Axis Scale (MTTR)</label>
          <select v-model="scaleY" class="input-field w-full">
            <option value="linear">Linear</option>
            <option value="logarithmic">Logarithmic</option>
          </select>
        </div>
      </div>
    </div>
    <ChartCriticality v-if="criticalityData" :data="criticalityData" :scaleX="scaleX" :scaleY="scaleY" :metricX="metricX" />

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
        
        <!-- High Risk Table -->
        <div class="rounded border border-red-200 dark:border-red-900/40 overflow-hidden text-xs">
          <div class="bg-red-100 dark:bg-red-950/40 px-3 py-2 font-semibold text-red-900 dark:text-red-300 flex justify-between">
            <span>High Risk</span>
            <span>Count: {{ classifiedRegions.highRisk.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">{{ metricX === 'probability' ? 'Prob (%)' : 'Freq' }}</th>
                <th class="px-3 py-1.5 text-right">MTTR (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.highRisk" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-red-600 dark:text-red-400">
                  {{ metricX === 'probability' ? item.x_val.toFixed(2) + '%' : item.x_val }}
                </td>
                <td class="px-3 py-1.5 text-right">{{ item.y_avg.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.highRisk.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- High Consequence Table -->
        <div class="rounded border border-orange-200 dark:border-orange-900/40 overflow-hidden text-xs">
          <div class="bg-orange-100 dark:bg-orange-950/40 px-3 py-2 font-semibold text-orange-900 dark:text-orange-300 flex justify-between">
            <span>High Consequence</span>
            <span>Count: {{ classifiedRegions.highConsequence.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">{{ metricX === 'probability' ? 'Prob (%)' : 'Freq' }}</th>
                <th class="px-3 py-1.5 text-right">MTTR (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.highConsequence" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right">
                  {{ metricX === 'probability' ? item.x_val.toFixed(2) + '%' : item.x_val }}
                </td>
                <td class="px-3 py-1.5 text-right font-medium text-orange-600 dark:text-orange-400">{{ item.y_avg.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.highConsequence.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- High Frequency Table -->
        <div class="rounded border border-yellow-250 dark:border-yellow-900/40 overflow-hidden text-xs">
          <div class="bg-yellow-100/80 dark:bg-yellow-950/30 px-3 py-2 font-semibold text-yellow-900 dark:text-yellow-300 flex justify-between">
            <span>High Frequency</span>
            <span>Count: {{ classifiedRegions.highFrequency.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">{{ metricX === 'probability' ? 'Prob (%)' : 'Freq' }}</th>
                <th class="px-3 py-1.5 text-right">MTTR (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.highFrequency" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-yellow-600 dark:text-yellow-400">
                  {{ metricX === 'probability' ? item.x_val.toFixed(2) + '%' : item.x_val }}
                </td>
                <td class="px-3 py-1.5 text-right">{{ item.y_avg.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.highFrequency.length === 0">
                <td colspan="3" class="px-3 py-4 text-center text-gray-400">Empty region</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Low Risk Table -->
        <div class="rounded border border-green-200 dark:border-green-900/40 overflow-hidden text-xs">
          <div class="bg-green-100 dark:bg-green-950/40 px-3 py-2 font-semibold text-green-900 dark:text-green-300 flex justify-between">
            <span>Low Risk</span>
            <span>Count: {{ classifiedRegions.lowRisk.length }}</span>
          </div>
          <table class="w-full text-left bg-white dark:bg-slate-900">
            <thead>
              <tr class="bg-gray-50 dark:bg-slate-950/40 text-gray-600 dark:text-slate-300 border-b border-gray-200 dark:border-slate-700 font-semibold">
                <th class="px-3 py-1.5">Item</th>
                <th class="px-3 py-1.5 text-right">{{ metricX === 'probability' ? 'Prob (%)' : 'Freq' }}</th>
                <th class="px-3 py-1.5 text-right">MTTR (h)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
              <tr v-for="item in classifiedRegions.lowRisk" :key="item.name" class="hover:bg-gray-50 dark:hover:bg-slate-800/40 text-gray-700 dark:text-slate-300">
                <td class="px-3 py-1.5">{{ item.name }}</td>
                <td class="px-3 py-1.5 text-right font-medium text-green-600 dark:text-green-400">
                  {{ metricX === 'probability' ? item.x_val.toFixed(2) + '%' : item.x_val }}
                </td>
                <td class="px-3 py-1.5 text-right">{{ item.y_avg.toFixed(1) }}</td>
              </tr>
              <tr v-if="classifiedRegions.lowRisk.length === 0">
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
import ChartCriticality from '../ChartCriticality.vue'

defineProps({
  availableEquipment: Array
})

const isCollapsed = ref(false)
const showTableDetails = ref(false)
const localFilters = ref({ equipment: '' })
const criticalityData = ref(null)
const compareBy = ref('mode') // Default to mode
const scaleX = ref('linear')
const scaleY = ref('linear')
const metricX = ref('count')
const showExplanation = ref(false)

const classifiedRegions = computed(() => {
  if (!criticalityData.value?.scatter_data) return null
  const avgX = metricX.value === 'probability' ? (criticalityData.value.averages?.probability * 100 || 0) : (criticalityData.value.averages?.failures || 0)
  const avgY = criticalityData.value.averages?.avg_downtime || 0

  const regions = {
    highRisk: [],
    highConsequence: [],
    highFrequency: [],
    lowRisk: []
  }

  criticalityData.value.scatter_data.forEach(item => {
    const x = metricX.value === 'probability' ? item.x_prob * 100 : item.x
    const y = item.y_avg
    if (x > avgX && y > avgY) {
      regions.highRisk.push({ ...item, x_val: x })
    } else if (x <= avgX && y > avgY) {
      regions.highConsequence.push({ ...item, x_val: x })
    } else if (x > avgX && y <= avgY) {
      regions.highFrequency.push({ ...item, x_val: x })
    } else {
      regions.lowRisk.push({ ...item, x_val: x })
    }
  })

  return regions
})

const loadAnalysis = async () => {
  try {
    await apiService.setFilters(localFilters.value.equipment)
    const res = await apiService.getJackknifeAnalysis(undefined, undefined, compareBy.value, null)
    criticalityData.value = res.data
  } catch (err) { console.error('Error loading Criticality analysis:', err) }
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