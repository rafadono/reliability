<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Análisis Deductivo y Causa Raíz (RCA & FTA)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Herramientas estructuradas para el diagnóstico de eventos de fallas mayores e identificación de causas de raíz operacionales. Cumplimiento de las normas IEC 62740 e IEC 61025.</p>
    </div>

    <IshikawaCard 
      :available-equipment="availableEquipment" 
    />

    <FtaCanvasCard />
  </div>
</template>

<script setup>
import IshikawaCard from './IshikawaCard.vue'
import FtaCanvasCard from './FtaCanvasCard.vue'

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
