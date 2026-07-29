<template>
  <div class="space-y-6 animate-fade-in">
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4">
      <h3 class="text-xl font-semibold text-gray-900 dark:text-white">Análisis RAM Cuantitativo</h3>
      <p class="text-sm text-gray-500 dark:text-slate-400">Herramientas estadísticas tradicionales para el modelado de confiabilidad, disponibilidad y análisis de fallas.</p>
    </div>

    <!-- Módulos Ejecutados (Visualización Completa) -->
    <div v-if="hasActiveModules" class="space-y-8">
      <AnalysisCardWrapper
        v-if="isExecuted('pareto')"
        :active="true"
        node-type="pareto"
        @navigate="$emit('navigate', $event)"
      >
        <ParetoCard id="pareto-card" :available-equipment="availableEquipment" :available-types="availableTypes" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('jackknife')"
        :active="true"
        node-type="jackknife"
        @navigate="$emit('navigate', $event)"
      >
        <JackknifeCard id="jackknife-card" :available-equipment="availableEquipment" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('criticality')"
        :active="true"
        node-type="criticality"
        @navigate="$emit('navigate', $event)"
      >
        <CriticalityCard id="criticality-card" :available-equipment="availableEquipment" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('weibull_kijima')"
        :active="true"
        node-type="weibull_kijima"
        @navigate="$emit('navigate', $event)"
      >
        <WeibullKijimaCard id="weibull-kijima-card" :available-equipment="availableEquipment" :available-types="availableTypes" />
      </AnalysisCardWrapper>

      <AnalysisCardWrapper
        v-if="isExecuted('event_plot')"
        :active="true"
        node-type="event_plot"
        @navigate="$emit('navigate', $event)"
      >
        <EventPlotCard id="event-plot-card" :available-equipment="availableEquipment" />
      </AnalysisCardWrapper>
    </div>

    <!-- Módulos No Configurados (Grid de Tarjetas Cuadradas) -->
    <div v-if="hasInactiveModules" class="space-y-3 pt-2">
      <div v-if="hasActiveModules" class="text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
        Módulos No Configurados
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnalysisCardWrapper
          v-if="!isExecuted('pareto')"
          :active="false"
          node-type="pareto"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('jackknife')"
          :active="false"
          node-type="jackknife"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('criticality')"
          :active="false"
          node-type="criticality"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('weibull_kijima')"
          :active="false"
          node-type="weibull_kijima"
          @navigate="$emit('navigate', $event)"
        />
        <AnalysisCardWrapper
          v-if="!isExecuted('event_plot')"
          :active="false"
          node-type="event_plot"
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
  if (nodeType === 'weibull_kijima') {
    return executed.includes('weibull') || executed.includes('kijima')
  }
  return executed.includes(nodeType)
}

const modules = ['pareto', 'jackknife', 'criticality', 'weibull_kijima', 'event_plot']

const hasActiveModules = computed(() => modules.some(m => isExecuted(m)))
const hasInactiveModules = computed(() => modules.some(m => !isExecuted(m)))
</script>
