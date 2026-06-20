<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Ingeniería de Mantenimiento (RCM & FMECA)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Implementación de estándares internacionales para la confiabilidad operacional. Flujo RCM según SAE JA1011/12 y evaluación del RPN según IEC 60812.</p>
    </div>

    <RcmWizardCard 
      :available-equipment="availableEquipment" 
    />

    <FmecaTableCard />
  </div>
</template>

<script setup>
import RcmWizardCard from './RcmWizardCard.vue'
import FmecaTableCard from './FmecaTableCard.vue'

defineProps({
  availableEquipment: {
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
