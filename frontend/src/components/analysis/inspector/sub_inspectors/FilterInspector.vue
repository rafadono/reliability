<template>
  <div class="space-y-4">
    <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
      Filtre los datos del pipeline por equipos, áreas de fallas o tipo de censura.
    </p>

    <!-- Filtro de Planta -->
    <div v-if="filterOptions?.plants && filterOptions.plants.length > 0" class="space-y-1">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Planta</label>
      <select 
        v-model="node.data.plant" 
        @change="onFilterChange"
        class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
      >
        <option value="">Todas las Plantas</option>
        <option v-for="p in filterOptions.plants" :key="p" :value="p">{{ p }}</option>
      </select>
    </div>

    <!-- Filtro de Equipo -->
    <div class="space-y-1">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Equipo</label>
      <select 
        v-model="node.data.equipment" 
        @change="onFilterChange"
        class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
      >
        <option value="">Todos los Equipos</option>
        <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
      </select>
    </div>

    <!-- Filtro de Tipo de Parada -->
    <div class="space-y-1 relative type-dropdown-container">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Tipo de Falla</label>
      <div>
        <button 
          type="button"
          @click="showTypeDropdown = !showTypeDropdown; showMdfDropdown = false"
          class="w-full text-left text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 flex justify-between items-center transition-all"
        >
          <span class="truncate">{{ selectedTypesText }}</span>
          <svg class="w-4 h-4 ml-2 transition-transform duration-200 shrink-0" :class="{ 'rotate-180': showTypeDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        <div 
          v-if="showTypeDropdown" 
          class="absolute left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto p-2 space-y-1.5 scrollbar-thin"
        >
          <div class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer">
            <input 
              type="checkbox" 
              id="all-types" 
              :checked="!node.data.type || node.data.type.length === 0"
              @change="toggleAllTypes"
              class="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label for="all-types" class="text-xs text-gray-700 dark:text-slate-200 font-bold select-none cursor-pointer flex-1">
              Todos los Tipos
            </label>
          </div>
          <div 
            v-for="t in availableTypesForFailure" 
            :key="t" 
            class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer"
          >
            <input 
              type="checkbox" 
              :id="'type-' + t" 
              :value="t"
              v-model="node.data.type"
              @change="onFilterChange"
              class="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label :for="'type-' + t" class="text-xs text-gray-700 dark:text-slate-200 select-none cursor-pointer flex-1 truncate">
              {{ t }}
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Filtro de Modo de Falla -->
    <div class="space-y-1 relative mdf-dropdown-container">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Modo de Falla</label>
      <div>
        <button 
          type="button"
          @click="showMdfDropdown = !showMdfDropdown; showTypeDropdown = false"
          class="w-full text-left text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 flex justify-between items-center transition-all"
        >
          <span class="truncate">{{ selectedMdfsText }}</span>
          <svg class="w-4 h-4 ml-2 transition-transform duration-200 shrink-0" :class="{ 'rotate-180': showMdfDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        <div 
          v-if="showMdfDropdown" 
          class="absolute left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto p-2 space-y-1.5 scrollbar-thin"
        >
          <div class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer">
            <input 
              type="checkbox" 
              id="all-mdfs" 
              :checked="!node.data.mdf || node.data.mdf.length === 0"
              @change="toggleAllMdfs"
              class="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label for="all-mdfs" class="text-xs text-gray-700 dark:text-slate-200 font-bold select-none cursor-pointer flex-1">
              Todos los Modos
            </label>
          </div>
          <div 
            v-for="m in availableMdfsForFailure" 
            :key="m" 
            class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer"
          >
            <input 
              type="checkbox" 
              :id="'mdf-' + m" 
              :value="m"
              v-model="node.data.mdf"
              class="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label :for="'mdf-' + m" class="text-xs text-gray-700 dark:text-slate-200 select-none cursor-pointer flex-1 truncate">
              {{ m }}
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Tipos a Censurar (Suspensiones) -->
    <div class="space-y-1 relative censored-types-container">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Tipos a Censurar (Suspensiones)</label>
      <div>
        <button 
          type="button"
          @click="showCensoredTypesDropdown = !showCensoredTypesDropdown; showTypeDropdown = false; showMdfDropdown = false; showCensoredMdfsDropdown = false"
          class="w-full text-left text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-amber-500 flex justify-between items-center transition-all"
        >
          <span class="truncate">{{ selectedCensoredTypesText }}</span>
          <svg class="w-4 h-4 ml-2 transition-transform duration-200 shrink-0" :class="{ 'rotate-180': showCensoredTypesDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        <div 
          v-if="showCensoredTypesDropdown" 
          class="absolute left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto p-2 space-y-1.5 scrollbar-thin"
        >
          <div class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer">
            <input 
              type="checkbox" 
              id="all-censored-types" 
              :checked="!node.data.censored_types || node.data.censored_types.length === 0"
              @change="toggleAllCensoredTypes"
              class="rounded border-gray-300 text-amber-600 focus:ring-amber-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label for="all-censored-types" class="text-xs text-gray-700 dark:text-slate-200 font-bold select-none cursor-pointer flex-1">
              Ningún Tipo Censurado
            </label>
          </div>
          <div 
            v-for="t in availableTypesForCensorship" 
            :key="'cen-type-' + t" 
            class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer"
          >
            <input 
              type="checkbox" 
              :id="'cen-type-' + t" 
              :value="t"
              v-model="node.data.censored_types"
              @change="onFilterChange"
              class="rounded border-gray-300 text-amber-600 focus:ring-amber-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label :for="'cen-type-' + t" class="text-xs text-gray-700 dark:text-slate-200 select-none cursor-pointer flex-1 truncate">
              {{ t }}
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Modos a Censurar (Suspensiones) -->
    <div class="space-y-1 relative censored-mdfs-container">
      <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Modos a Censurar (Suspensiones)</label>
      <div>
        <button 
          type="button"
          @click="showCensoredMdfsDropdown = !showCensoredMdfsDropdown; showTypeDropdown = false; showMdfDropdown = false; showCensoredTypesDropdown = false"
          class="w-full text-left text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-amber-500 flex justify-between items-center transition-all"
        >
          <span class="truncate">{{ selectedCensoredMdfsText }}</span>
          <svg class="w-4 h-4 ml-2 transition-transform duration-200 shrink-0" :class="{ 'rotate-180': showCensoredMdfsDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        <div 
          v-if="showCensoredMdfsDropdown" 
          class="absolute left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto p-2 space-y-1.5 scrollbar-thin"
        >
          <div class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer">
            <input 
              type="checkbox" 
              id="all-censored-mdfs" 
              :checked="isAllCensoredMdfsSelected"
              @change="toggleAllCensoredMdfs"
              class="rounded border-gray-300 text-amber-600 focus:ring-amber-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label for="all-censored-mdfs" class="text-xs text-gray-700 dark:text-slate-200 font-bold select-none cursor-pointer flex-1">
              {{ node.data.censored_types && node.data.censored_types.length > 0 ? 'Todos los Modos (' + node.data.censored_types.join(', ') + ')' : 'Ningún Modo Censurado' }}
            </label>
          </div>
          <div 
            v-for="m in availableMdfsForCensorship" 
            :key="'cen-mdf-' + m" 
            class="flex items-center gap-2 p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-700/60 cursor-pointer"
          >
            <input 
              type="checkbox" 
              :id="'cen-mdf-' + m" 
              :value="m"
              v-model="node.data.censored_mdfs"
              class="rounded border-gray-300 text-amber-600 focus:ring-amber-500 h-3.5 w-3.5 cursor-pointer"
            />
            <label :for="'cen-mdf-' + m" class="text-xs text-gray-700 dark:text-slate-200 select-none cursor-pointer flex-1 truncate">
              {{ m }}
            </label>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useFilterCascade } from '../composables/useFilterCascade'

const props = defineProps({
  node: { type: Object, required: true },
  availableEquipment: { type: Array, default: () => [] },
  filterOptions: { type: Object, default: () => ({ types: [], mdfs: [] }) }
})

const emit = defineEmits(['filter-changed'])

const {
  showTypeDropdown,
  showMdfDropdown,
  showCensoredTypesDropdown,
  showCensoredMdfsDropdown,
  selectedTypesText,
  selectedMdfsText,
  selectedCensoredTypesText,
  selectedCensoredMdfsText,
  isAllCensoredMdfsSelected,
  availableTypesForFailure,
  availableTypesForCensorship,
  availableMdfsForFailure,
  availableMdfsForCensorship,
  toggleAllTypes,
  toggleAllMdfs,
  toggleAllCensoredTypes,
  toggleAllCensoredMdfs,
  onFilterChange
} = useFilterCascade(props, emit)
</script>
