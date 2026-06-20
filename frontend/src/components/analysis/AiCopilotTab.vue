<template>
  <div class="space-y-8 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Copiloto de Confiabilidad IA</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Asistencia inteligente basada en Inteligencia Artificial y Procesamiento de Lenguaje Natural para el análisis cualitativo e interpretación del historial técnico de mantenimiento.</p>
    </div>

    <!-- Chat de Copiloto Inteligente -->
    <AiChatCard />

    <!-- Minería de texto / Clasificación semántica de Hugging Face -->
    <AiAnalysisCard id="ai-card" />
  </div>
</template>

<script setup>
import AiChatCard from './AiChatCard.vue'
import AiAnalysisCard from './AiAnalysisCard.vue'
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
