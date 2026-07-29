<template>
  <div class="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 p-6 transition-all duration-300">
    <div class="border-b border-gray-100 dark:border-slate-700 pb-4 mb-6">
      <h4 class="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
        {{ $t('charts.iso_analysis.fta_title') }}
        <span class="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-slate-700 px-2 py-0.5 rounded-full">Norma: IEC 61025</span>
      </h4>
      <p class="text-xs text-gray-500 dark:text-slate-400">{{ $t('charts.iso_analysis.fta_desc') }}</p>
      <p v-if="sharedState.fta && !sharedState.fta.error" class="text-[11px] text-emerald-700 dark:text-emerald-400 mt-1">
        Último cálculo del Workbench: {{ (sharedState.fta.top_event_probability * 100).toFixed(2) }}%
      </p>
    </div>

    <!-- Controles de Probabilidad General -->
    <div class="bg-emerald-50/50 dark:bg-slate-900/20 border border-emerald-100/50 dark:border-slate-700/50 rounded-lg p-4 mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <span class="text-xs font-bold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.top_event_prob') }}</span>
        <div class="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">
          {{ (calculatedTopProbability * 100).toFixed(4) }}%
        </div>
      </div>
      <div class="text-xs text-gray-500 dark:text-slate-400 max-w-sm">
        {{ $t('charts.iso_analysis.fta_instruction') }}
      </div>
    </div>

    <!-- Lienzo del Árbol de Fallas -->
    <div class="flex flex-col items-center py-6 border border-gray-100 dark:border-slate-700 rounded-xl bg-gray-50/30 dark:bg-slate-900/30 overflow-x-auto min-w-[700px]">
      
      <!-- TOP EVENT -->
      <div class="flex flex-col items-center">
        <div class="bg-red-600 text-white font-extrabold text-xs px-5 py-3 rounded-lg shadow-md border-2 border-red-500 text-center w-52 relative">
          TOP: {{ $t('charts.iso_analysis.top_event_label') }}
          <span class="block text-[10px] font-normal mt-1 opacity-90">P = {{ (calculatedTopProbability * 100).toFixed(2) }}%</span>
        </div>
        
        <!-- Línea conectora -->
        <div class="w-0.5 h-8 bg-gray-300 dark:bg-slate-600"></div>

        <!-- COMPUERTA CENTRAL -->
        <div class="flex flex-col items-center relative">
          <button 
            @click="toggleGate"
            class="px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest shadow-sm border transition-all duration-300 cursor-pointer"
            :class="gateType === 'AND' ? 'bg-indigo-600 text-white border-indigo-500 hover:bg-indigo-700' : 'bg-orange-500 text-white border-orange-400 hover:bg-orange-600'"
          >
            {{ $t('charts.iso_analysis.gate_label') }} {{ gateType }}
          </button>

          <!-- Líneas de ramificación -->
          <div class="w-[380px] h-4 border-t-2 border-x-2 border-gray-300 dark:border-slate-600 mt-2"></div>
        </div>

        <!-- EVENTOS BÁSICOS (HIJOS) -->
        <div class="flex justify-between w-[500px] mt-2 gap-6">
          <div 
            v-for="(event, idx) in basicEvents" 
            :key="idx"
            class="flex flex-col items-center w-40"
          >
            <!-- Línea vertical conectora -->
            <div class="w-0.5 h-4 bg-gray-300 dark:bg-slate-600 mb-1"></div>

            <div class="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg p-3 shadow-sm w-full text-center relative hover:shadow-md transition-shadow">
              <span class="block text-[10px] font-bold text-gray-800 dark:text-slate-200 truncate" :title="event.name || $t('charts.iso_analysis.' + event.key)">
                {{ event.name || $t('charts.iso_analysis.' + event.key) }}
              </span>
              
              <!-- Slider de Probabilidad -->
              <div class="mt-2.5 space-y-1">
                <div class="flex justify-between text-[9px] text-gray-500 dark:text-slate-400 font-semibold">
                  <span>{{ $t('charts.iso_analysis.prob_label') }}:</span>
                  <span>{{ (event.prob * 100).toFixed(1) }}%</span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="100" 
                  step="1"
                  v-model.number="event.probPercent"
                  @input="updateEventProb(idx)"
                  class="w-full h-1 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { sharedState } from '../../sharedState'

const defaultBasicEvents = () => [
  { key: 'fta_event_bearing', name: null, prob: 0.12, probPercent: 12 },
  { key: 'fta_event_seal', name: null, prob: 0.08, probPercent: 8 },
  { key: 'fta_event_motor', name: null, prob: 0.05, probPercent: 5 }
]

const basicEventsFromConfig = (events) => events.map(e => {
  const prob = typeof e.probability === 'number' ? e.probability : 0
  return {
    key: null,
    name: e.name || '',
    prob,
    probPercent: Math.round(prob * 100)
  }
})

const ftaNodeConfig = sharedState.nodeConfigs && sharedState.nodeConfigs['fta']

const gateType = ref(
  (ftaNodeConfig && (ftaNodeConfig.gate_type === 'AND' || ftaNodeConfig.gate_type === 'OR'))
    ? ftaNodeConfig.gate_type
    : 'OR'
) // AND o OR

const basicEvents = ref(
  (ftaNodeConfig && Array.isArray(ftaNodeConfig.basic_events) && ftaNodeConfig.basic_events.length > 0)
    ? basicEventsFromConfig(ftaNodeConfig.basic_events)
    : defaultBasicEvents()
)

const calculatedTopProbability = computed(() => {
  const probs = basicEvents.value.map(e => e.prob)
  if (gateType.value === 'AND') {
    // Multiplicación de todas las probabilidades
    return probs.reduce((acc, p) => acc * p, 1)
  } else {
    // OR: 1 - product(1 - p)
    const noneOccur = probs.reduce((acc, p) => acc * (1 - p), 1)
    return 1 - noneOccur
  }
})

const toggleGate = () => {
  gateType.value = gateType.value === 'AND' ? 'OR' : 'AND'
}

const updateEventProb = (idx) => {
  const ev = basicEvents.value[idx]
  ev.prob = ev.probPercent / 100
}

// This component is kept alive by <keep-alive> in Dashboard.vue, so a later
// Workbench re-execution (with a different gate/basic-events configuration)
// needs a live watcher, not just the one-time setup-time seed above.
watch(() => sharedState.nodeConfigs && sharedState.nodeConfigs['fta'], (config) => {
  if (config && (config.gate_type === 'AND' || config.gate_type === 'OR')) {
    gateType.value = config.gate_type
  }
  if (config && Array.isArray(config.basic_events) && config.basic_events.length > 0) {
    basicEvents.value = basicEventsFromConfig(config.basic_events)
  }
}, { deep: true })
</script>

<style scoped>
input[type="range"]::-webkit-slider-thumb {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #10b981;
  cursor: pointer;
  -webkit-appearance: none;
}
</style>
