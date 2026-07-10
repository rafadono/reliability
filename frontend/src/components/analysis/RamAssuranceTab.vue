<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Aseguramiento de Producción (RAM)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Modelamiento predictivo y análisis de confiabilidad, disponibilidad y mantenibilidad para plantas industriales.</p>
    </div>

    <!-- Simulador RAM -->
    <AnalysisCardWrapper
      :active="isExecuted('ram_sim')"
      node-type="ram_sim"
      @navigate="$emit('navigate', $event)"
    >
      <RamSimulatorCard 
        :available-equipment="availableEquipment" 
      />
    </AnalysisCardWrapper>

    <!-- APM Bad Actors y Growth -->
    <AnalysisCardWrapper
      :active="isExecuted('apm')"
      node-type="apm"
      @navigate="$emit('navigate', $event)"
    >
      <ApmCard 
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />
    </AnalysisCardWrapper>

    <!-- Tendencia de KPIs -->
    <AnalysisCardWrapper
      :active="isExecuted('trend')"
      node-type="trend"
      @navigate="$emit('navigate', $event)"
    >
      <TrendCard 
        :available-equipment="availableEquipment" 
        :available-types="availableTypes" 
      />
    </AnalysisCardWrapper>
  </div>
</template>

<script setup>
import { sharedState } from '../../sharedState'
import AnalysisCardWrapper from './AnalysisCardWrapper.vue'
import RamSimulatorCard from './RamSimulatorCard.vue'
import ApmCard from './ApmCard.vue'
import TrendCard from './TrendCard.vue'

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
  if (nodeType === 'ram_sim') {
    return executed.includes('ramSimulator')
  }
  if (nodeType === 'apm') {
    return executed.includes('trend') || executed.includes('weibull') || executed.includes('kijima')
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
