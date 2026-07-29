<template>
  <div 
    class="fixed inset-y-0 right-0 w-80 bg-white/95 dark:bg-slate-900/95 backdrop-blur-md shadow-2xl border-l border-gray-200 dark:border-slate-800 z-50 transform transition-transform duration-300 p-6 flex flex-col justify-between"
    :class="isOpen ? 'translate-x-0' : 'translate-x-full'"
  >
    <div class="flex-1 overflow-y-auto pr-2 scrollbar-thin">
      <!-- Encabezado del Inspector -->
      <div class="flex items-center justify-between border-b border-gray-100 dark:border-slate-800 pb-4 mb-6">
        <div>
          <h4 class="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">{{ $t('workbench.inspector_title') }}</h4>
          <span class="text-[10px] bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 font-extrabold px-2 py-0.5 rounded mt-1 inline-block">
            {{ $t('workbench.block_id') }}: {{ node.id }}
          </span>
        </div>
        <button 
          type="button"
          @click="$emit('close')" 
          title="Cerrar inspector"
          aria-label="Cerrar inspector"
          class="text-gray-500 hover:text-gray-900 dark:text-slate-400 dark:hover:text-white p-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 transition-all flex items-center justify-center shadow-sm cursor-pointer"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Parámetros específicos por tipo de nodo -->
      <div class="space-y-6">
        <DataSourceInspector 
          v-if="node.type === 'dataSource'" 
          :node="node" 
          :available-equipment="availableEquipment" 
          @upload-file="$emit('upload-file', $event)" 
          @reset="$emit('reset')" 
        />
        
        <FilterInspector 
          v-else-if="node.type === 'filter'" 
          :node="node" 
          :available-equipment="availableEquipment" 
          :filter-options="filterOptions" 
          @filter-changed="$emit('filter-changed', $event)" 
        />
        
        <WeibullInspector 
          v-else-if="node.type === 'weibull'" 
          :node="node" 
        />
        
        <KijimaInspector 
          v-else-if="node.type === 'kijima'" 
          :node="node" 
        />
        
        <FmecaInspector
          v-else-if="node.type === 'fmeca'"
          :node="node"
        />

        <FtaInspector
          v-else-if="node.type === 'fta'"
          :node="node"
        />

        <GenericInspector
          v-else
          :node="node"
          :available-equipment="availableEquipment"
        />
      </div>
    </div>

    <!-- Botones de Acción de Pie de Inspector -->
    <div class="pt-4 border-t border-gray-100 dark:border-slate-800/80 space-y-2">
      <button 
        @click="$emit('run')"
        class="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2.5 rounded-lg transition-all shadow-md flex items-center justify-center gap-2 cursor-pointer"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        {{ $t('workbench.run_pipeline') }}
      </button>
      <button 
        @click="$emit('delete', node.id)"
        class="w-full bg-red-50 hover:bg-red-100 dark:bg-red-950/20 dark:hover:bg-red-950/40 text-red-600 dark:text-red-400 text-xs font-bold py-2 rounded-lg transition-all cursor-pointer"
      >
        {{ $t('workbench.delete_block') }}
      </button>
    </div>
  </div>
</template>

<script setup>
import DataSourceInspector from './inspector/sub_inspectors/DataSourceInspector.vue'
import FilterInspector from './inspector/sub_inspectors/FilterInspector.vue'
import WeibullInspector from './inspector/sub_inspectors/WeibullInspector.vue'
import KijimaInspector from './inspector/sub_inspectors/KijimaInspector.vue'
import FmecaInspector from './inspector/sub_inspectors/FmecaInspector.vue'
import FtaInspector from './inspector/sub_inspectors/FtaInspector.vue'
import GenericInspector from './inspector/sub_inspectors/GenericInspector.vue'

defineProps({
  isOpen: { type: Boolean, required: true },
  node: { type: Object, required: true },
  availableEquipment: { type: Array, required: true },
  filterOptions: { type: Object, default: () => ({ types: [], mdfs: [] }) }
})

defineEmits(['close', 'run', 'delete', 'filter-changed', 'upload-file', 'reset'])
</script>

<style scoped>
.scrollbar-thin::-webkit-scrollbar {
  width: 4px;
}
.scrollbar-thin::-webkit-scrollbar-track {
  background: transparent;
}
.scrollbar-thin::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}
.dark .scrollbar-thin::-webkit-scrollbar-thumb {
  background: #334155;
}
</style>
