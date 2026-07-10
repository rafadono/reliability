<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Análisis RAM Cuantitativo</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Herramientas estadísticas tradicionales para el modelado de confiabilidad, disponibilidad y análisis de fallas.</p>
    </div>

    <AnalysisCardWrapper
      :active="isExecuted('pareto')"
      node-type="pareto"
      @navigate="$emit('navigate', $event)"
    >
      <ParetoCard 
        id="pareto-card"
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />
    </AnalysisCardWrapper>

    <AnalysisCardWrapper
      :active="isExecuted('jackknife')"
      node-type="jackknife"
      @navigate="$emit('navigate', $event)"
    >
      <JackknifeCard 
        id="jackknife-card" 
        :available-equipment="availableEquipment" 
      />
    </AnalysisCardWrapper>

    <AnalysisCardWrapper
      :active="isExecuted('criticality')"
      node-type="criticality"
      @navigate="$emit('navigate', $event)"
    >
      <CriticalityCard 
        id="criticality-card" 
        :available-equipment="availableEquipment" 
      />
    </AnalysisCardWrapper>

    <AnalysisCardWrapper
      :active="isExecuted('weibull_kijima')"
      node-type="weibull_kijima"
      @navigate="$emit('navigate', $event)"
    >
      <WeibullKijimaCard 
        id="weibull-kijima-card"
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />
    </AnalysisCardWrapper>

    <AnalysisCardWrapper
      :active="isExecuted('event_plot')"
      node-type="event_plot"
      @navigate="$emit('navigate', $event)"
    >
      <EventPlotCard 
        id="event-plot-card" 
        :available-equipment="availableEquipment" 
      />
    </AnalysisCardWrapper>
  </div>
</template>

<script setup>
import { sharedState } from '../../sharedState'
import AnalysisCardWrapper from './AnalysisCardWrapper.vue'
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

defineEmits(['navigate'])

const isExecuted = (nodeType) => {
  const executed = sharedState.executedNodes || []
  if (nodeType === 'event_plot' || nodeType === 'criticality') {
    return executed.includes('dataSource') || executed.includes('filter')
  }
  if (nodeType === 'weibull_kijima') {
    return executed.includes('weibull') || executed.includes('kijima')
  }
  return executed.includes(nodeType)
}
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
