<template>
  <div class="space-y-6 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Ingeniería de Mantenimiento (RCM & FMECA)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Implementación de estándares internacionales para la confiabilidad operacional. Flujo RCM según SAE JA1011/12 y evaluación del RPN según IEC 60812.</p>
    </div>

    <!-- Módulos Ejecutados -->
    <div v-if="hasActiveModules" class="space-y-8">
      <AnalysisCardWrapper
        v-if="isExecuted('rcm')"
        :active="true"
        node-type="rcm"
        title="Asistente RCM (SAE JA1011)"
        @navigate="$emit('navigate', $event)"
      >
        <RcmWizardCard :available-equipment="availableEquipment" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('fmeca')"
        :active="true"
        node-type="fmeca"
        title="Matriz FMECA y RPN (IEC 60812)"
        @navigate="$emit('navigate', $event)"
      >
        <FmecaTableCard />
      </AnalysisCardWrapper>
    </div>

    <!-- Módulos No Configurados (Grid de Tarjetas Cuadradas) -->
    <div v-if="hasInactiveModules" class="space-y-3 pt-2">
      <div v-if="hasActiveModules" class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
        Módulos No Configurados
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnalysisCardWrapper
          v-if="!isExecuted('rcm')"
          :active="false"
          node-type="rcm"
          title="Asistente RCM JA1011"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('fmeca')"
          :active="false"
          node-type="fmeca"
          title="Matriz FMECA RPN"
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
import RcmWizardCard from './RcmWizardCard.vue'
import FmecaTableCard from './FmecaTableCard.vue'

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

const modules = ['rcm', 'fmeca']

const hasActiveModules = computed(() => modules.some(m => isExecuted(m)))
const hasInactiveModules = computed(() => modules.some(m => !isExecuted(m)))
</script>
