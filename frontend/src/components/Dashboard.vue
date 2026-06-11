<template>
  <div class="p-8">
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">Analysis Dashboard</h1>
      <p class="text-gray-600 dark:text-slate-400">Independent analysis modules. Customize filters for each chart.</p>
    </div>

    <div class="space-y-8">
      <ParetoCard 
        id="pareto-card"
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />

      <JackknifeCard id="jackknife-card" :available-equipment="availableEquipment" />

      <CriticalityCard id="criticality-card" :available-equipment="availableEquipment" />

      <WeibullCard 
        id="weibull-card"
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />

      <EventPlotCard id="event-plot-card" :available-equipment="availableEquipment" />

      <ApmCard id="apm-card" :available-equipment="availableEquipment" />

      <TrendCard id="trend-card" :available-equipment="availableEquipment" />

      <NlpCard id="nlp-card" />
    </div>

  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { apiService } from '../api'
import ParetoCard from './analysis/ParetoCard.vue'
import JackknifeCard from './analysis/JackknifeCard.vue'
import CriticalityCard from './analysis/CriticalityCard.vue'
import WeibullCard from './analysis/WeibullCard.vue'
import EventPlotCard from './analysis/EventPlotCard.vue'
import ApmCard from './analysis/ApmCard.vue'
import TrendCard from './analysis/TrendCard.vue'
import NlpCard from './analysis/NlpCard.vue'

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