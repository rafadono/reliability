<template>
  <div class="space-y-4">
    <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
      Ajusta modelos de reparación imperfecta para estimar el factor de restauración del mantenimiento preventivo y correctivo.
    </p>

    <div class="space-y-2">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">Modelos Kijima a Ajustar</label>
      <div class="space-y-1.5 bg-gray-50 dark:bg-slate-800 p-3 rounded-lg border border-gray-200 dark:border-slate-700 text-xs">
        <label v-for="mOpt in kijimaModelOptions" :key="'kmodel-' + mOpt.value" class="flex items-center gap-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-slate-700/50 p-1 rounded">
          <input 
            type="checkbox" 
            :value="mOpt.value" 
            v-model="node.data.model_types" 
            class="rounded text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5 cursor-pointer"
          />
          <span class="text-gray-800 dark:text-slate-200 font-semibold">{{ mOpt.label }}</span>
        </label>
      </div>
    </div>

    <div class="space-y-2">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">
        Umbral Mínimo TBX (Tiempo entre Eventos)
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
      <p class="text-[11px] text-gray-400">Omite intervalos entre eventos extremadamente pequeños.</p>
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
      <p class="text-[11px] text-gray-400">Omite detenciones de muy corta duración.</p>
    </div>
  </div>
</template>

<script setup>
import { watch } from 'vue'

const props = defineProps({
  node: { type: Object, required: true }
})

const kijimaModelOptions = [
  { value: 1, label: 'Kijima I (Efecto en el último ciclo)' },
  { value: 2, label: 'Kijima II (Efecto acumulado total)' },
  { value: 3, label: 'Kijima I TD (Temporal Exponencial)' },
  { value: 4, label: 'Kijima II TD (Temporal Exponencial)' },
  { value: 5, label: 'Kijima I TD2 (Temporal Logístico)' },
  { value: 6, label: 'Kijima II TD2 (Temporal Logístico)' }
]

watch(() => props.node?.id, () => {
  if (props.node && props.node.type === 'kijima') {
    if (!props.node.data.model_types || !Array.isArray(props.node.data.model_types) || props.node.data.model_types.length === 0) {
      if (props.node.data.model_type) {
        props.node.data.model_types = [props.node.data.model_type]
      } else {
        props.node.data.model_types = [1, 2]
      }
    }
    if (props.node.data.min_tbx === undefined) props.node.data.min_tbx = 0.0
    if (props.node.data.min_ttx === undefined) props.node.data.min_ttx = 0.0
  }
}, { immediate: true })
</script>
