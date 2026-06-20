<template>
  <div class="p-6 max-w-7xl mx-auto min-h-screen">
    <!-- Encabezado de la Planta -->
    <div class="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <h2 class="text-3xl font-extrabold text-gray-900 dark:text-white tracking-tight">{{ $t('dashboard.title') }}</h2>
        <p class="text-sm text-gray-500 dark:text-slate-400 mt-1">{{ $t('dashboard.desc') }}</p>
      </div>
    </div>

    <!-- Navegación de Pestañas Estilo Moderno / Glassmorphic -->
    <div class="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md rounded-xl p-1.5 border border-gray-100 dark:border-slate-700/80 shadow-sm mb-8 flex flex-wrap gap-1 shrink-0">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        @click="activeTab = tab.id"
        class="flex-1 min-w-[120px] text-center px-4 py-2.5 rounded-lg text-xs font-bold transition-all duration-300 flex items-center justify-center gap-2"
        :class="[
          activeTab === tab.id
            ? 'bg-blue-600 text-white shadow-md shadow-blue-500/10'
            : 'text-gray-600 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 hover:text-gray-900 dark:hover:text-white'
        ]"
      >
        <component :is="tab.icon" class="w-4 h-4 shrink-0" />
        {{ tab.name }}
      </button>
    </div>

    <!-- Contenido Dinámico de Pestañas con Efecto de Transición -->
    <div class="transition-all duration-300">
      <keep-alive>
        <component 
          :is="activeComponent" 
          :available-equipment="availableEquipment"
          :available-types="availableTypes"
        />
      </keep-alive>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, h } from 'vue'
import { apiService } from '../api'

// Importación de Contenedores de Pestañas
import QuantReliabilityTab from './analysis/QuantReliabilityTab.vue'
import RcmFmecaTab from './analysis/RcmFmecaTab.vue'
import RcaFtaTab from './analysis/RcaFtaTab.vue'
import RamAssuranceTab from './analysis/RamAssuranceTab.vue'
import AiCopilotTab from './analysis/AiCopilotTab.vue'

const availableEquipment = ref([])
const availableTypes = ref([])
const activeTab = ref('quant')

// Iconos inline usando SVG funcionales (h)
const ChartBarIcon = () => h('svg', { class: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2' })
])
const AdjustmentsIcon = () => h('svg', { class: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4' })
])
const SearchIcon = () => h('svg', { class: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' })
])
const LightningBoltIcon = () => h('svg', { class: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M13 10V3L4 14h7v7l9-11h-7z' })
])
const ChatIcon = () => h('svg', { class: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
  h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z' })
])

const tabs = [
  { id: 'quant', name: 'Analisis Cuantitativo', icon: ChartBarIcon, component: QuantReliabilityTab },
  { id: 'rcm_fmea', name: 'RCM & FMECA', icon: AdjustmentsIcon, component: RcmFmecaTab },
  { id: 'rca_fta', name: 'RCA & FTA', icon: SearchIcon, component: RcaFtaTab },
  { id: 'ram', name: 'Aseguramiento RAM (ISO)', icon: LightningBoltIcon, component: RamAssuranceTab },
  { id: 'copilot', name: 'Copiloto IA', icon: ChatIcon, component: AiCopilotTab }
]

const activeComponent = computed(() => {
  const current = tabs.find(t => t.id === activeTab.value)
  return current ? current.component : QuantReliabilityTab
})

import { watch, nextTick } from 'vue'

const props = defineProps({
  isLoading: Boolean,
  activeTabProp: String
})

watch(() => props.activeTabProp, (newVal) => {
  if (newVal) {
    activeTab.value = newVal
  }
})

const changeTabAndScroll = async (tabId, cardId) => {
  activeTab.value = tabId
  await nextTick()
  setTimeout(() => {
    const el = document.getElementById(cardId)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, 100)
}

defineExpose({
  changeTabAndScroll
})

const loadInitialFilters = async () => {
  try {
    const response = await apiService.getAvailableFilters()
    availableEquipment.value = response.data.equipment || []
    availableTypes.value = response.data.types || []
  } catch (error) {
    console.error('Error loading filters', error)
  }
}

onMounted(async () => {
  await loadInitialFilters()
})
</script>