<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Aseguramiento de Producción (RAM ISO 20815)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Modelamiento predictivo y análisis de confiabilidad, disponibilidad y mantenibilidad según la norma internacional ISO 20815 para plantas industriales.</p>
    </div>

    <!-- Simulador RAM -->
    <RamSimulatorCard 
      :available-equipment="availableEquipment" 
    />

    <!-- APM Bad Actors y Growth -->
    <ApmCard 
      :available-equipment="availableEquipment" 
      :available-types="availableTypes" 
    />

    <!-- Tendencia de KPIs -->
    <TrendCard 
      :available-equipment="availableEquipment" 
      :available-types="availableTypes" 
    />
  </div>
</template>

<script setup>
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
