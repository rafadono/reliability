<template>
  <div class="space-y-4">
    <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
      Defina y registre modos de falla del activo para compilar la matriz de RPN.
    </p>

    <div class="space-y-3 bg-gray-50 dark:bg-slate-900/40 rounded-xl p-3 border border-gray-100 dark:border-slate-800 text-xs">
      <div class="font-bold text-gray-700 dark:text-slate-300">Añadir Modo Falla:</div>
      
      <input 
        v-model="newRecord.component" 
        type="text" 
        placeholder="Componente" 
        class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-indigo-500"
      />
      <input 
        v-model="newRecord.mode" 
        type="text" 
        placeholder="Modo de falla" 
        class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-indigo-500"
      />
      <input 
        v-model="newRecord.effect" 
        type="text" 
        placeholder="Efecto de la falla" 
        class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-indigo-500"
      />

      <div class="grid grid-cols-3 gap-2">
        <div>
          <label class="text-[9px] font-semibold text-gray-500 block mb-0.5">Severidad (1-10)</label>
          <input 
            v-model.number="newRecord.severity" 
            type="number" 
            min="1" 
            max="10" 
            class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2 py-1 outline-none"
          />
        </div>
        <div>
          <label class="text-[9px] font-semibold text-gray-500 block mb-0.5">Ocurrencia (1-10)</label>
          <input 
            v-model.number="newRecord.occurrence" 
            type="number" 
            min="1" 
            max="10" 
            class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2 py-1 outline-none"
          />
        </div>
        <div>
          <label class="text-[9px] font-semibold text-gray-500 block mb-0.5">Detección (1-10)</label>
          <input 
            v-model.number="newRecord.detection" 
            type="number" 
            min="1" 
            max="10" 
            class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2 py-1 outline-none"
          />
        </div>
      </div>

      <input 
        v-model="newRecord.action" 
        type="text" 
        placeholder="Acción recomendada" 
        class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-indigo-500"
      />

      <button 
        @click="addFmecaRecord" 
        class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-1.5 rounded-lg transition-colors cursor-pointer"
      >
        Añadir Registro
      </button>
    </div>

    <!-- Lista de registros FMECA actuales -->
    <div v-if="node.data.records && node.data.records.length > 0" class="space-y-2">
      <div class="text-xs font-bold text-gray-700 dark:text-slate-300">Registros ({{ node.data.records.length }}):</div>
      <div class="max-h-40 overflow-y-auto space-y-1.5 pr-1 scrollbar-thin">
        <div 
          v-for="(rec, idx) in node.data.records" 
          :key="idx" 
          class="bg-gray-50 dark:bg-slate-800/80 p-2 rounded border border-gray-200 dark:border-slate-700 text-[11px] flex justify-between items-start"
        >
          <div>
            <div class="font-bold text-gray-800 dark:text-white">{{ rec.component || 'Sin Comp.' }} - {{ rec.mode }}</div>
            <div class="text-gray-500 dark:text-slate-400 text-[10px]">RPN: {{ (rec.severity || 5) * (rec.occurrence || 5) * (rec.detection || 5) }} (S:{{ rec.severity }}, O:{{ rec.occurrence }}, D:{{ rec.detection }})</div>
          </div>
          <button @click="node.data.records.splice(idx, 1)" class="text-red-500 hover:text-red-700 font-bold px-1">✕</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  node: { type: Object, required: true }
})

const newRecord = ref({
  component: '',
  mode: '',
  effect: '',
  severity: 5,
  occurrence: 5,
  detection: 5,
  action: ''
})

const addFmecaRecord = () => {
  if (!props.node.data.records) {
    props.node.data.records = []
  }
  props.node.data.records.push({ ...newRecord.value })
  
  newRecord.value = {
    component: '',
    mode: '',
    effect: '',
    severity: 5,
    occurrence: 5,
    detection: 5,
    action: ''
  }
}
</script>
