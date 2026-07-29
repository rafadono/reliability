<template>
  <div class="space-y-4">
    <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
      Define el Árbol de Fallas (IEC 61025): un evento tope, la compuerta lógica que combina los eventos básicos, y la probabilidad de cada evento básico.
    </p>

    <div class="space-y-1">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">Evento Tope (Top Event)</label>
      <input
        v-model="node.data.top_event"
        type="text"
        placeholder="Falla funcional del equipo"
        class="w-full text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
      />
    </div>

    <div class="space-y-1">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">Compuerta Lógica</label>
      <div class="flex gap-2">
        <button
          type="button"
          @click="node.data.gate_type = 'OR'"
          class="flex-1 text-xs font-bold py-1.5 rounded-lg border transition-colors cursor-pointer"
          :class="node.data.gate_type !== 'AND' ? 'bg-orange-500 text-white border-orange-400' : 'bg-gray-50 dark:bg-slate-800 text-gray-600 dark:text-slate-300 border-gray-200 dark:border-slate-700'"
        >OR (unión)</button>
        <button
          type="button"
          @click="node.data.gate_type = 'AND'"
          class="flex-1 text-xs font-bold py-1.5 rounded-lg border transition-colors cursor-pointer"
          :class="node.data.gate_type === 'AND' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-gray-50 dark:bg-slate-800 text-gray-600 dark:text-slate-300 border-gray-200 dark:border-slate-700'"
        >AND (intersección)</button>
      </div>
      <p class="text-[11px] text-gray-400">OR: el evento tope ocurre si cualquier evento básico ocurre. AND: requiere que todos ocurran simultáneamente.</p>
    </div>

    <div class="space-y-2">
      <div class="flex items-center justify-between">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Eventos Básicos</label>
        <button
          type="button"
          @click="addEvent"
          class="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer"
        >+ Añadir evento</button>
      </div>
      <div v-if="!node.data.basic_events || node.data.basic_events.length === 0" class="text-[11px] text-gray-400 italic">
        Sin eventos básicos configurados — se usarán 3 eventos de ejemplo por defecto.
      </div>
      <div v-for="(ev, idx) in node.data.basic_events" :key="idx" class="bg-gray-50 dark:bg-slate-900/40 rounded-lg p-2.5 border border-gray-100 dark:border-slate-800 space-y-1.5">
        <div class="flex items-center gap-2">
          <input
            v-model="ev.name"
            type="text"
            placeholder="Nombre del evento básico"
            class="flex-1 text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded px-2 py-1 outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button type="button" @click="removeEvent(idx)" class="text-red-500 hover:text-red-700 font-bold px-1 cursor-pointer">✕</button>
        </div>
        <div class="flex items-center gap-2">
          <input
            type="range" min="0" max="100" step="1"
            v-model.number="ev._probPercent"
            @input="ev.probability = ev._probPercent / 100"
            class="flex-1 h-1 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
          />
          <span class="text-[10px] font-bold text-gray-600 dark:text-slate-300 w-10 text-right">{{ ev._probPercent }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { watch } from 'vue'

const props = defineProps({
  node: { type: Object, required: true }
})

// Keep an internal `_probPercent` (0-100 int) alongside `probability` (0-1 float)
// purely for the slider's UX; `probability` is the only field the backend reads.
watch(() => props.node?.id, () => {
  if (!props.node) return
  if (!props.node.data.gate_type) props.node.data.gate_type = 'OR'
  if (!props.node.data.top_event) props.node.data.top_event = ''
  if (!Array.isArray(props.node.data.basic_events)) props.node.data.basic_events = []
  props.node.data.basic_events.forEach(ev => {
    if (ev._probPercent === undefined) {
      ev._probPercent = Math.round((ev.probability ?? 0) * 100)
    }
  })
}, { immediate: true })

const addEvent = () => {
  if (!Array.isArray(props.node.data.basic_events)) props.node.data.basic_events = []
  props.node.data.basic_events.push({ name: '', probability: 0.1, _probPercent: 10 })
}

const removeEvent = (idx) => {
  props.node.data.basic_events.splice(idx, 1)
}
</script>
