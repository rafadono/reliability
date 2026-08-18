<template>
  <div class="space-y-6 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Análisis Deductivo y Causa Raíz (RCA & FTA)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Herramientas estructuradas para el diagnóstico de eventos de fallas mayores e identificación de causas de raíz operacionales. Cumplimiento de las normas IEC 62740 e IEC 61025.</p>
    </div>

    <!-- Módulos Ejecutados -->
    <div v-if="hasActiveModules" class="space-y-8">
      <AnalysisCardWrapper
        v-if="isExecuted('rca')"
        :active="true"
        node-type="rca_fta"
        title="Análisis Causa Raíz RCA (Diagrama de Ishikawa)"
        @navigate="$emit('navigate', $event)"
      >
        <IshikawaCard id="ishikawa-card" :available-equipment="availableEquipment" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('fta')"
        :active="true"
        node-type="rca_fta"
        title="Árbol de Fallas FTA (IEC 61025)"
        @navigate="$emit('navigate', $event)"
      >
        <FtaCanvasCard id="fta-canvas-card" />
      </AnalysisCardWrapper>
    </div>

    <!-- Módulos No Configurados (Grid de Tarjetas Cuadradas) -->
    <div v-if="hasInactiveModules" class="space-y-3 pt-2">
      <div v-if="hasActiveModules" class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
        Módulos No Configurados
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnalysisCardWrapper
          v-if="!isExecuted('rca')"
          :active="false"
          node-type="rca_fta"
          title="Análisis Causa Raíz RCA"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('fta')"
          :active="false"
          node-type="rca_fta"
          title="Árbol de Fallas FTA"
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
import IshikawaCard from './IshikawaCard.vue'
import FtaCanvasCard from './FtaCanvasCard.vue'

defineProps({
  availableEquipment: {
    type: Array,
    required: true
  }
})

defineEmits(['navigate'])

const isExecuted = (nodeType) => {
  const executed = sharedState.executedNodes || []
  return executed.includes(nodeType)
}

const modules = ['rca', 'fta']

const hasActiveModules = computed(() => modules.some(m => isExecuted(m)))
const hasInactiveModules = computed(() => modules.some(m => !isExecuted(m)))
</script>
