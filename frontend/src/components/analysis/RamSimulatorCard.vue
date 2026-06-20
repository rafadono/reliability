<template>
  <div class="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 p-6 transition-all duration-300">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-100 dark:border-slate-700 pb-4 mb-6">
      <div>
        <h4 class="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <svg class="w-5 h-5 text-amber-600 dark:text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          {{ $t('charts.iso_analysis.ram_title') }}
        </h4>
        <p class="text-xs text-gray-500 dark:text-slate-400">{{ $t('charts.iso_analysis.ram_desc') }}</p>
      </div>

      <!-- Selector de Equipo para Simular -->
      <div class="flex items-center gap-2">
        <label class="text-xs font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.equipment_label') }}</label>
        <select 
          v-model="selectedEquipment"
          class="text-sm bg-gray-50 dark:bg-slate-700 border border-gray-200 dark:border-slate-600 text-gray-900 dark:text-white rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-amber-500 outline-none"
        >
          <option value="">{{ $t('charts.iso_analysis.whole_plant') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Columna 1: Controles del Simulador -->
      <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-5 border border-gray-100 dark:border-slate-800 space-y-6">
        <h5 class="text-xs font-bold text-gray-700 dark:text-slate-300 uppercase tracking-wider">{{ $t('charts.iso_analysis.sim_parameters') }}</h5>
        
        <!-- Slider 1: Eficiencia Preventiva -->
        <div class="space-y-2">
          <div class="flex justify-between items-center text-xs font-semibold">
            <span class="text-gray-600 dark:text-slate-400">{{ $t('charts.iso_analysis.prev_efficiency') }}</span>
            <span class="text-amber-600 dark:text-amber-400 font-bold">{{ Math.round(preventiveEfficiency * 100) }}%</span>
          </div>
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.05"
            v-model.number="preventiveEfficiency"
            class="w-full h-1.5 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <p class="text-[10px] text-gray-400 dark:text-slate-500 leading-snug">{{ $t('charts.iso_analysis.prev_efficiency_desc') }}</p>
        </div>

        <!-- Slider 2: Demora Logística -->
        <div class="space-y-2">
          <div class="flex justify-between items-center text-xs font-semibold">
            <span class="text-gray-600 dark:text-slate-400">{{ $t('charts.iso_analysis.avg_logistics_delay') }}</span>
            <span class="text-amber-600 dark:text-amber-400 font-bold">{{ logisticsDelay.toFixed(1) }} hrs</span>
          </div>
          <input 
            type="range" 
            min="0" 
            max="24" 
            step="0.5"
            v-model.number="logisticsDelay"
            class="w-full h-1.5 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <p class="text-[10px] text-gray-400 dark:text-slate-500 leading-snug">{{ $t('charts.iso_analysis.logistics_delay_desc') }}</p>
        </div>

        <button 
          @click="runSimulation" 
          :disabled="loading"
          class="w-full bg-amber-600 hover:bg-amber-700 disabled:bg-gray-300 dark:disabled:bg-slate-700 disabled:text-gray-500 text-white text-xs font-bold py-2.5 rounded-lg shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2"
        >
          <span v-if="loading" class="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
          {{ $t('charts.iso_analysis.run_ram_sim') }}
        </button>
      </div>

      <!-- Columna 2: Kpis del Resultado -->
      <div class="lg:col-span-2 space-y-6">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Tarjeta Disponibilidad -->
          <div class="bg-gray-50 dark:bg-slate-900/20 border border-gray-100 dark:border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.simulated_availability') }}</div>
            <div class="my-3">
              <span class="text-4xl font-extrabold text-amber-600 dark:text-amber-400">{{ results.availability || '--' }}%</span>
            </div>
            <div class="text-[10px] text-gray-400 dark:text-slate-500">
              {{ $t('charts.iso_analysis.uptime_hours') }} <strong>{{ results.uptime_hours || '--' }} hrs</strong> / {{ $t('charts.iso_analysis.downtime_hours') }} <strong>{{ results.downtime_hours || '--' }} hrs</strong> (Base 8,760 hrs anuales).
            </div>
          </div>

          <!-- Tarjeta Aseguramiento de Producción -->
          <div class="bg-gray-50 dark:bg-slate-900/20 border border-gray-100 dark:border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.production_assurance_iso') }}</div>
            <div class="my-3">
              <span class="text-4xl font-extrabold text-blue-600 dark:text-blue-400">{{ results.production_assurance || '--' }}%</span>
            </div>
            <div class="text-[10px] text-gray-400 dark:text-slate-500">
              {{ $t('charts.iso_analysis.production_assurance_desc') }}
            </div>
          </div>
        </div>

        <!-- Historial Gráfico Mensual -->
        <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-5 border border-gray-100 dark:border-slate-800" v-if="results.timeline && results.timeline.length > 0">
          <h5 class="text-xs font-bold text-gray-700 dark:text-slate-300 mb-4 uppercase tracking-wider">{{ $t('charts.iso_analysis.monthly_profile') }}</h5>
          
          <!-- Gráfico SVG Premium y Moderno -->
          <div class="relative w-full h-36">
            <svg viewBox="0 0 500 120" class="w-full h-full overflow-visible">
              <!-- Grid lines -->
              <line x1="0" y1="20" x2="500" y2="20" stroke="#f1f5f9" stroke-width="1" class="dark:stroke-slate-800" />
              <line x1="0" y1="60" x2="500" y2="60" stroke="#f1f5f9" stroke-width="1" class="dark:stroke-slate-800" />
              <line x1="0" y1="100" x2="500" y2="100" stroke="#e2e8f0" stroke-width="1" class="dark:stroke-slate-700" />
              
              <!-- Línea de datos -->
              <path 
                :d="svgPath" 
                fill="none" 
                stroke="url(#gradient-stroke)" 
                stroke-width="3" 
                stroke-linecap="round"
                class="transition-all duration-500"
              />
              
              <!-- Puntos de datos -->
              <circle 
                v-for="(pt, idx) in chartPoints" 
                :key="idx"
                :cx="pt.x"
                :cy="pt.y"
                r="4.5"
                fill="#d97706"
                stroke="#ffffff"
                stroke-width="1.5"
                class="hover:scale-150 transition-transform duration-200 cursor-pointer"
                :title="`${pt.month}: ${pt.val}%`"
              />
              
              <!-- Nombres de Meses -->
              <text 
                v-for="(pt, idx) in chartPoints" 
                :key="'t-'+idx"
                :x="pt.x"
                y="116"
                text-anchor="middle"
                class="text-[9px] font-semibold fill-gray-400 dark:fill-slate-500"
              >
                {{ pt.month }}
              </text>

              <defs>
                <linearGradient id="gradient-stroke" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#3b82f6" />
                  <stop offset="100%" stop-color="#d97706" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>
      </div>
    </div>

    <!-- Bad Actors Relacionados de la Simulación -->
    <div class="mt-6 border-t border-gray-100 dark:border-slate-800 pt-6" v-if="results.bad_actors && results.bad_actors.length > 0">
      <h5 class="text-xs font-bold text-gray-700 dark:text-slate-300 mb-4 uppercase tracking-wider">{{ $t('charts.iso_analysis.identified_bad_actors') }}</h5>
      <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div 
          v-for="(actor, idx) in results.bad_actors" 
          :key="idx"
          class="bg-gray-50 dark:bg-slate-900/30 rounded-xl p-3 border border-gray-100 dark:border-slate-800/80 flex flex-col justify-between"
        >
          <div class="text-[10px] font-bold text-gray-500 dark:text-slate-400 truncate">{{ actor.equipment }}</div>
          <div class="my-1.5 flex items-baseline gap-1">
            <span class="text-lg font-extrabold text-amber-600 dark:text-amber-400">{{ actor.downtime }}</span>
            <span class="text-[9px] text-gray-400 dark:text-slate-500">hrs</span>
          </div>
          <div class="text-[9px] text-gray-400 dark:text-slate-500 font-semibold">{{ actor.failures }} {{ $t('charts.iso_analysis.events_registered') }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { apiService } from '../../api'

const props = defineProps({
  availableEquipment: {
    type: Array,
    required: true
  }
})

const selectedEquipment = ref('')
const preventiveEfficiency = ref(0.8)
const logisticsDelay = ref(4.0)
const loading = ref(false)
const results = ref({})

const runSimulation = async () => {
  loading.value = true
  try {
    const response = await apiService.simulateRam(
      selectedEquipment.value,
      preventiveEfficiency.value,
      logisticsDelay.value
    )
    if (response.data.status === 'success') {
      results.value = response.data
    }
  } catch (error) {
    console.error('Error running RAM simulation:', error)
  } finally {
    loading.value = false
  }
}

// Convert data coordinates to SVG point path
const chartPoints = computed(() => {
  if (!results.value.timeline) return []
  const data = results.value.timeline
  const width = 500
  const height = 120
  const paddingX = 30
  const paddingY = 20
  const usableWidth = width - paddingX * 2
  const usableHeight = height - paddingY * 2

  const stepX = usableWidth / (data.length - 1)

  return data.map((item, idx) => {
    // Availability is mapped between 50% (height=100) and 100% (height=20)
    const val = item.availability
    const ratio = (val - 50) / 50 // 0 to 1
    const y = 100 - (ratio * usableHeight)
    return {
      x: paddingX + idx * stepX,
      y: y,
      month: item.month,
      val: val
    }
  })
})

const svgPath = computed(() => {
  const pts = chartPoints.value
  if (pts.length === 0) return ''
  return `M ${pts[0].x} ${pts[0].y} ` + pts.slice(1).map(p => `L ${p.x} ${p.y}`).join(' ')
})

watch(selectedEquipment, () => {
  runSimulation()
})

onMounted(() => {
  runSimulation()
})
</script>

<style scoped>
input[type="range"]::-webkit-slider-thumb {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #d97706;
  cursor: pointer;
  -webkit-appearance: none;
}
.animate-fade-in {
  animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
