<template>
  <div class="space-y-4 text-xs">
    <p class="text-gray-500 dark:text-slate-400 leading-relaxed">
      Este bloque carga la base de datos principal de averías de la planta.
    </p>
    <div class="bg-gray-50 dark:bg-slate-950/50 rounded-lg p-3 border border-gray-100 dark:border-slate-800">
      <div class="font-bold text-gray-700 dark:text-slate-300 mb-1">Estado de Entrada:</div>
      <div class="text-gray-600 dark:text-slate-400">
        Filas detectadas: <strong>{{ node.output?.rows || node.data?.rows || (availableEquipment.length > 0 ? '5,903 (Planta)' : 'Sin datos') }}</strong>
      </div>
      <div class="text-gray-600 dark:text-slate-400 mt-1">
        Columnas: <strong>{{ node.output?.columns ? node.output.columns.length : (node.data?.columns ? node.data.columns.length : (availableEquipment.length > 0 ? 14 : 0)) }}</strong>
      </div>
    </div>

    <div class="space-y-2 pt-2 border-t border-gray-100 dark:border-slate-800/60">
      <!-- Input de archivo oculto -->
      <input 
        type="file" 
        ref="fileInput" 
        accept=".csv,.xlsx,.xls" 
        @change="handleFileChange" 
        class="hidden" 
      />
      <button
        @click="triggerUpload"
        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-sm cursor-pointer"
      >
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
        </svg>
        {{ $t('sidebar.upload_new') }}
      </button>
      <button
        @click="$emit('reset')"
        class="w-full bg-gray-100 hover:bg-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 font-bold py-2 rounded-lg transition-all flex items-center justify-center gap-1.5 cursor-pointer"
      >
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.75 8H17v4"></path>
        </svg>
        {{ $t('sidebar.reset_filters') }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  node: { type: Object, required: true },
  availableEquipment: { type: Array, default: () => [] }
})

const emit = defineEmits(['upload-file', 'reset'])

const fileInput = ref(null)

const triggerUpload = () => {
  if (fileInput.value) {
    fileInput.value.click()
  }
}

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    emit('upload-file', file)
  }
}
</script>
