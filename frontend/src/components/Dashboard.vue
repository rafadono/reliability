<template>
  <div class="p-6 max-w-7xl mx-auto min-h-screen">
    <div class="mb-8">
      <h2 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">{{ $t('dashboard.title') }}</h2>
      <p class="text-gray-600 dark:text-slate-400">{{ $t('dashboard.desc') }}</p>
    </div>

    <div class="space-y-8">
      <ParetoCard 
        id="pareto-card"
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />

      <JackknifeCard id="jackknife-card" :available-equipment="availableEquipment" />

      <CriticalityCard id="criticality-card" :available-equipment="availableEquipment" />

      <WeibullKijimaCard 
        id="weibull-kijima-card"
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />

      <EventPlotCard id="event-plot-card" :available-equipment="availableEquipment" />

      <ApmCard id="apm-card" :available-equipment="availableEquipment" :available-types="availableTypes" />

      <TrendCard id="trend-card" :available-equipment="availableEquipment" :available-types="availableTypes" />

      <AiAnalysisCard id="ai-card" />
    </div>

  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../api'
import ParetoCard from './analysis/ParetoCard.vue'
import JackknifeCard from './analysis/JackknifeCard.vue'
import CriticalityCard from './analysis/CriticalityCard.vue'
import WeibullKijimaCard from './analysis/WeibullKijimaCard.vue'
import EventPlotCard from './analysis/EventPlotCard.vue'
import ApmCard from './analysis/ApmCard.vue'
import TrendCard from './analysis/TrendCard.vue'
import AiAnalysisCard from './analysis/AiAnalysisCard.vue'

const availableEquipment = ref([])
const availableTypes = ref([])

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