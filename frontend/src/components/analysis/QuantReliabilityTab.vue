<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Análisis RAM Cuantitativo</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Herramientas estadísticas tradicionales para el modelado de confiabilidad, disponibilidad y análisis de fallas.</p>
    </div>

    <ParetoCard 
      id="pareto-card"
      :available-equipment="availableEquipment" 
      :available-types="availableTypes" 
    />

    <JackknifeCard 
      id="jackknife-card" 
      :available-equipment="availableEquipment" 
    />

    <CriticalityCard 
      id="criticality-card" 
      :available-equipment="availableEquipment" 
    />

    <WeibullKijimaCard 
      id="weibull-kijima-card"
      :available-equipment="availableEquipment" 
      :available-types="availableTypes" 
    />

    <EventPlotCard 
      id="event-plot-card" 
      :available-equipment="availableEquipment" 
    />
  </div>
</template>

<script setup>
import ParetoCard from './ParetoCard.vue'
import JackknifeCard from './JackknifeCard.vue'
import CriticalityCard from './CriticalityCard.vue'
import WeibullKijimaCard from './WeibullKijimaCard.vue'
import EventPlotCard from './EventPlotCard.vue'

defineProps({
  availableEquipment: {
    type: Array,
    required: true
  },
  availableTypes: {
    type: Array,
    required: true
  }
})
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
