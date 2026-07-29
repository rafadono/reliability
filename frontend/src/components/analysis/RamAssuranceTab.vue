<template>
  <div class="space-y-6 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Aseguramiento de Producción (RAM)</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Modelamiento predictivo y análisis de confiabilidad, disponibilidad y mantenibilidad para plantas industriales.</p>
    </div>

    <!-- Módulos Ejecutados -->
    <div v-if="hasActiveModules" class="space-y-8">
      <AnalysisCardWrapper
        v-if="isExecuted('ram_sim')"
        :active="true"
        node-type="ram_sim"
        @navigate="$emit('navigate', $event)"
      >
        <RamSimulatorCard :available-equipment="availableEquipment" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('apm')"
        :active="true"
        node-type="apm"
        @navigate="$emit('navigate', $event)"
      >
        <ApmCard :available-equipment="availableEquipment" :available-types="availableTypes" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('trend')"
        :active="true"
        node-type="trend"
        @navigate="$emit('navigate', $event)"
      >
        <TrendCard :available-equipment="availableEquipment" :available-types="availableTypes" />
      </AnalysisCardWrapper>
    </div>

    <!-- Módulos No Configurados (Grid de Tarjetas Cuadradas) -->
    <div v-if="hasInactiveModules" class="space-y-3 pt-2">
      <div v-if="hasActiveModules" class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
        Módulos No Configurados
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnalysisCardWrapper
          v-if="!isExecuted('ram_sim')"
          :active="false"
          node-type="ram_sim"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('apm')"
          :active="false"
          node-type="apm"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('trend')"
          :active="false"
          node-type="trend"
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
  if (nodeType === 'ram_sim') return executed.includes('ram') || executed.includes('ramSimulator')
  return executed.includes(nodeType)
}

const modules = ['ram_sim', 'apm', 'trend']

const hasActiveModules = computed(() => modules.some(m => isExecuted(m)))
const hasInactiveModules = computed(() => modules.some(m => !isExecuted(m)))
</script>
