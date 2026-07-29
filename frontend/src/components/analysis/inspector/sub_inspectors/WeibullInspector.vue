<template>
  <div class="space-y-4">
    <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
      Ajusta la distribución probabilística de Weibull 2P para analizar tiempos entre fallas (TBX) y tiempos de reparación (TTX).
    </p>

    <div class="space-y-2">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">
        Umbral Mínimo TBX (Tiempo entre Fallas)
      </label>
      <div class="flex items-center gap-2">
        <input 
          type="number" 
          v-model.number="node.data.min_tbx" 
          min="0" 
          step="0.5" 
          placeholder="0.0"
          class="w-full text-xs border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none" 
        />
        <span class="text-xs text-gray-500 font-semibold">hrs</span>
      </div>
      <p class="text-[11px] text-gray-400">Descarta eventos o micro-detenciones con TBX menor al umbral.</p>
    </div>

    <div class="space-y-2">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">
        Umbral Mínimo TTX (Duración de Detención)
      </label>
      <div class="flex items-center gap-2">
        <input 
          type="number" 
          v-model.number="node.data.min_ttx" 
          min="0" 
          step="0.1" 
          placeholder="0.0"
          class="w-full text-xs border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none" 
        />
        <span class="text-xs text-gray-500 font-semibold">hrs</span>
      </div>
      <p class="text-[11px] text-gray-400">Descarta paradas con duración menor al umbral especificado.</p>
    </div>
  </div>
</template>

<script setup>
import { watch } from 'vue'

const props = defineProps({
  node: { type: Object, required: true }
})

watch(() => props.node?.id, () => {
  if (props.node && props.node.type === 'weibull') {
    if (props.node.data.min_tbx === undefined) props.node.data.min_tbx = 0.0
    if (props.node.data.min_ttx === undefined) props.node.data.min_ttx = 0.0
  }
}, { immediate: true })
</script>
