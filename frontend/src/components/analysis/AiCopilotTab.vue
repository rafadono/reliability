<template>
  <div class="space-y-6 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Copiloto de Confiabilidad IA</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Asistencia inteligente basada en Inteligencia Artificial y Procesamiento de Lenguaje Natural para el análisis cualitativo e interpretación del historial técnico de mantenimiento.</p>
    </div>

    <!-- Módulos Ejecutados -->
    <div v-if="hasActiveModules" class="space-y-8">
      <AnalysisCardWrapper
        v-if="isExecuted('ai_chat')"
        :active="true"
        node-type="ai_chat"
        title="Asistente Conversacional IA"
        @navigate="$emit('navigate', $event)"
      >
        <AiChatCard />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('comment_mining')"
        :active="true"
        node-type="comment_mining"
        title="Minería de Comentarios y Clasificación NLP"
        @navigate="$emit('navigate', $event)"
      >
        <AiAnalysisCard id="ai-card" />
      </AnalysisCardWrapper>
    </div>

    <!-- Módulos No Configurados (Grid de Tarjetas Cuadradas) -->
    <div v-if="hasInactiveModules" class="space-y-3 pt-2">
      <div v-if="hasActiveModules" class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
        Módulos No Configurados
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnalysisCardWrapper
          v-if="!isExecuted('ai_chat')"
          :active="false"
          node-type="ai_chat"
          title="Asistente Conversacional IA"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('comment_mining')"
          :active="false"
          node-type="comment_mining"
          title="Minería de Comentarios NLP"
          @navigate="$emit('navigate', $event)"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { sharedState } from '../../sharedState'
import AnalysisCardWrapper from './AnalysisCardWrapper.vue'
import AiChatCard from './AiChatCard.vue'
import AiAnalysisCard from './AiAnalysisCard.vue'

defineEmits(['navigate'])

const isExecuted = (nodeType) => {
  const executed = sharedState.executedNodes || []
  if (nodeType === 'ai_chat') return true // Always active for user interaction
  return executed.includes(nodeType)
}

const modules = ['ai_chat', 'comment_mining']

const hasActiveModules = computed(() => modules.some(m => isExecuted(m)))
const hasInactiveModules = computed(() => modules.some(m => !isExecuted(m)))
</script>
