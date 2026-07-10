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
          @click="$emit('close')" 
          class="text-gray-400 hover:text-gray-600 dark:hover:text-white p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Parámetros específicos por tipo de nodo -->
      <div class="space-y-6">
        <!-- 1. Nodo DataSource -->
        <div v-if="node.type === 'dataSource'" class="space-y-4 text-xs">
          <p class="text-gray-500 dark:text-slate-400 leading-relaxed">
            Este bloque carga la base de datos principal de averías de la planta.
          </p>
          <div class="bg-gray-50 dark:bg-slate-950/50 rounded-lg p-3 border border-gray-100 dark:border-slate-800">
            <div class="font-bold text-gray-700 dark:text-slate-300 mb-1">Estado de Entrada:</div>
            <div class="text-gray-600 dark:text-slate-400">
              Filas detectadas: <strong>{{ node.data.rows || 'Cargando...' }}</strong>
            </div>
            <div class="text-gray-600 dark:text-slate-400 mt-1">
              Columnas: <strong>{{ node.data.columns ? node.data.columns.length : 0 }}</strong>
            </div>
          </div>

          <div class="space-y-2 pt-2 border-t border-gray-100 dark:border-slate-800/60">
            <button
              @click="triggerUpload"
              class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-sm"
            >
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
              </svg>
              {{ $t('sidebar.upload_new') }}
            </button>
            <button
              @click="$emit('reset')"
              class="w-full bg-gray-100 hover:bg-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 font-bold py-2 rounded-lg transition-all flex items-center justify-center gap-1.5"
            >
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.75 8H17v4"></path>
              </svg>
              {{ $t('sidebar.reset_filters') }}
            </button>
            <input
              type="file"
              ref="fileInput"
              accept=".csv"
              @change="handleFileChange"
              class="hidden"
            />
          </div>
        </div>

        <!-- 2. Nodo Filtro -->
        <div v-if="node.type === 'filter'" class="space-y-4">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Filtre los datos del pipeline por equipos, áreas de fallas o tipo de censura.
          </p>

          <!-- Filtro de Equipo -->
          <div class="space-y-1">
            <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Equipo</label>
            <select 
              v-model="node.data.equipment" 
              @change="onFilterChange"
              class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Planta Completa</option>
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
                <span class="truncate">
                  {{ selectedTypesText }}
                </span>
                <svg class="w-4 h-4 ml-2 transition-transform duration-200 shrink-0" :class="{ 'rotate-180': showTypeDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              <!-- Dropdown Panel -->
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
                  v-for="t in filterOptions.types" 
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
                <span class="truncate">
                  {{ selectedMdfsText }}
                </span>
                <svg class="w-4 h-4 ml-2 transition-transform duration-200 shrink-0" :class="{ 'rotate-180': showMdfDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              <!-- Dropdown Panel -->
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
                  v-for="m in filterOptions.mdfs" 
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

          <!-- Filtro de Censura -->
          <div class="space-y-1">
            <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Censurado</label>
            <select 
              v-model="node.data.censored" 
              class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="all">Ver Todos</option>
              <option value="0">Fallas Correctivas (0)</option>
              <option value="1">Eventos Censurados (1)</option>
            </select>
          </div>
        </div>

        <!-- 3. Nodo Weibull -->
        <div v-if="node.type === 'weibull'" class="space-y-4">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Calcula la confiabilidad estructural ajustando una distribución biparamétrica de Weibull (beta y eta) al conjunto de datos filtrado.
          </p>
          <div class="bg-gray-50 dark:bg-slate-950/50 rounded-lg p-3 border border-gray-100 dark:border-slate-800 text-xs">
            <span class="font-bold text-gray-700 dark:text-slate-300 block mb-1">Método de Ajuste:</span>
            <span class="text-gray-600 dark:text-slate-400">Regresión lineal de rangos (Median Ranks) de mínimos cuadrados sobre curva de distribución acumulada CDF.</span>
          </div>
        </div>

        <!-- 4. Nodo Kijima -->
        <div v-if="node.type === 'kijima'" class="space-y-4">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Ajusta modelos de reparación imperfecta para estimar el factor de restauración del mantenimiento preventivo y correctivo.
          </p>

          <div class="space-y-1">
            <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Modelo Kijima</label>
            <select 
              v-model.number="node.data.model_type" 
              class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="1">Kijima I (Efecto en el último ciclo)</option>
              <option value="2">Kijima II (Efecto acumulado total)</option>
              <option value="3">Kijima I TD (Temporal Exponencial)</option>
              <option value="4">Kijima II TD (Temporal Exponencial)</option>
              <option value="5">Kijima I TD2 (Temporal Logístico)</option>
              <option value="6">Kijima II TD2 (Temporal Logístico)</option>
            </select>
          </div>
        </div>

        <!-- 5. Nodo FMECA -->
        <div v-if="node.type === 'fmeca'" class="space-y-4">
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
              placeholder="Efecto de falla" 
              class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <input 
              v-model="newRecord.action" 
              type="text" 
              placeholder="Acción mitigar" 
              class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-indigo-500"
            />

            <div class="grid grid-cols-3 gap-2">
              <div>
                <span class="text-[9px] text-gray-500 block mb-0.5">S (1-10)</span>
                <input v-model.number="newRecord.severity" type="number" min="1" max="10" class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-center py-1" />
              </div>
              <div>
                <span class="text-[9px] text-gray-500 block mb-0.5">O (1-10)</span>
                <input v-model.number="newRecord.occurrence" type="number" min="1" max="10" class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-center py-1" />
              </div>
              <div>
                <span class="text-[9px] text-gray-500 block mb-0.5">D (1-10)</span>
                <input v-model.number="newRecord.detection" type="number" min="1" max="10" class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded text-center py-1" />
              </div>
            </div>

            <button 
              @click="addFmecaRecord"
              :disabled="!newRecord.component || !newRecord.mode"
              class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-1.5 rounded-lg shadow-sm disabled:opacity-50"
            >
              Registrar Modo
            </button>
          </div>
        </div>

        <!-- 6. Nodo RAM Simulator -->
        <div v-if="node.type === 'ramSimulator'" class="space-y-5">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Simula la disponibilidad operacional futura de planta y el Production Assurance.
          </p>

          <!-- Slider Eficiencia PM -->
          <div class="space-y-1.5">
            <div class="flex justify-between text-xs font-semibold">
              <span class="text-gray-600 dark:text-slate-400">Eficiencia Preventivos (PM):</span>
              <span class="text-indigo-600 dark:text-indigo-400 font-bold">{{ Math.round(node.data.preventive_efficiency * 100) }}%</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              v-model.number="node.data.preventive_efficiency"
              class="w-full h-1 bg-gray-200 dark:bg-slate-700 rounded appearance-none cursor-pointer accent-indigo-600"
            />
          </div>

          <!-- Slider Demora Logística -->
          <div class="space-y-1.5">
            <div class="flex justify-between text-xs font-semibold">
              <span class="text-gray-600 dark:text-slate-400">Demora Logística Promedio:</span>
              <span class="text-indigo-600 dark:text-indigo-400 font-bold">{{ node.data.logistics_delay }} hrs</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="24" 
              step="0.5"
              v-model.number="node.data.logistics_delay"
              class="w-full h-1 bg-gray-200 dark:bg-slate-700 rounded appearance-none cursor-pointer accent-indigo-600"
            />
          </div>
        </div>

        <!-- 7. Nodo Pareto -->
        <div v-if="node.type === 'pareto'" class="space-y-4">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Identifica el 20% de las causas críticas responsables del 80% de las pérdidas o fallas.
          </p>

          <div class="space-y-1">
            <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Agrupar por</label>
            <select 
              v-model="node.data.group_by" 
              class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
            >
              <option value="Equipment">Equipo</option>
              <option value="Type">Tipo de Falla</option>
              <option value="mdf">Modo de Falla</option>
            </select>
          </div>
        </div>

        <!-- 8. Nodo Jackknife -->
        <div v-if="node.type === 'jackknife'" class="space-y-4">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Diagrama costo-riesgo que clasifica activos según la cantidad de fallas (frecuencia) e inactividad total.
          </p>

          <div class="space-y-1">
            <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Comparar por</label>
            <select 
              v-model="node.data.compare_by" 
              class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
            >
              <option value="Equipment">Equipo</option>
              <option value="Type">Tipo de Falla</option>
              <option value="mdf">Modo de Falla</option>
            </select>
          </div>
        </div>

        <!-- 9. Nodo KPI Trend -->
        <div v-if="node.type === 'trend'" class="space-y-4">
          <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
            Muestra el perfil histórico mensual y las tendencias de fallas, MTBF, MTTR y disponibilidad.
          </p>
          <div class="bg-gray-50 dark:bg-slate-950/50 rounded-lg p-3 border border-gray-100 dark:border-slate-800 text-xs text-gray-600 dark:text-slate-400">
            Este bloque no requiere parámetros adicionales y procesará directamente la serie de tiempo del conjunto de datos.
          </div>
        </div>
      </div>
    </div>

    <!-- Acciones en el pie del inspector -->
    <div class="border-t border-gray-100 dark:border-slate-800 pt-4 mt-6 shrink-0 space-y-2">
      <button 
        @click="$emit('run')"
        class="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2.5 rounded-lg shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-1.5"
      >
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        {{ $t('workbench.run_pipeline') }}
      </button>
      <button 
        @click="$emit('delete', node.id)"
        class="w-full bg-red-50 hover:bg-red-100 dark:bg-red-950/20 dark:hover:bg-red-950/40 text-red-600 dark:text-red-400 text-xs font-bold py-2 rounded-lg transition-all"
      >
        {{ $t('workbench.delete_block') }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  isOpen: {
    type: Boolean,
    required: true
  },
  node: {
    type: Object,
    required: true
  },
  availableEquipment: {
    type: Array,
    required: true
  },
  filterOptions: {
    type: Object,
    default: () => ({ types: [], mdfs: [] })
  }
})

const emit = defineEmits(['close', 'run', 'delete', 'filter-changed', 'upload-file', 'reset'])

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

const newRecord = ref({
  component: '',
  mode: '',
  effect: '',
  severity: 5,
  occurrence: 5,
  detection: 5,
  action: ''
})

const onFilterChange = () => {
  emit('filter-changed', {
    equipment: props.node.data.equipment,
    type: props.node.data.type
  })
}

const addFmecaRecord = () => {
  if (!props.node.data.records) {
    props.node.data.records = []
  }
  props.node.data.records.push({ ...newRecord.value })
  
  // Limpiar campos
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

// Custom dropdown refs and computed properties
const showTypeDropdown = ref(false)
const showMdfDropdown = ref(false)

const selectedTypesText = computed(() => {
  const selected = props.node.data?.type || []
  if (selected.length === 0) {
    return 'Todos los Tipos'
  }
  return selected.join(', ')
})

const selectedMdfsText = computed(() => {
  const selected = props.node.data?.mdf || []
  if (selected.length === 0) {
    return 'Todos los Modos'
  }
  return selected.join(', ')
})

const toggleAllTypes = (event) => {
  if (event.target.checked) {
    props.node.data.type = []
  }
  onFilterChange()
}

const toggleAllMdfs = (event) => {
  if (event.target.checked) {
    props.node.data.mdf = []
  }
}

const handleClickOutside = (e) => {
  if (!e.target.closest('.type-dropdown-container')) {
    showTypeDropdown.value = false
  }
  if (!e.target.closest('.mdf-dropdown-container')) {
    showMdfDropdown.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})

watch(() => props.node.id, () => {
  showTypeDropdown.value = false
  showMdfDropdown.value = false

  if (props.node && props.node.type === 'filter') {
    if (!props.node.data.type) {
      props.node.data.type = []
    } else if (typeof props.node.data.type === 'string') {
      props.node.data.type = props.node.data.type ? [props.node.data.type] : []
    }
    
    if (!props.node.data.mdf) {
      props.node.data.mdf = []
    } else if (typeof props.node.data.mdf === 'string') {
      props.node.data.mdf = props.node.data.mdf ? [props.node.data.mdf] : []
    }
  }

  // Resetear formulario si cambia de nodo
  newRecord.value = {
    component: '',
    mode: '',
    effect: '',
    severity: 5,
    occurrence: 5,
    detection: 5,
    action: ''
  }
}, { immediate: true })
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
  border-radius: 2px;
}
.dark .scrollbar-thin::-webkit-scrollbar-thumb {
  background: #475569;
}
input[type="range"]::-webkit-slider-thumb {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #4f46e5;
  cursor: pointer;
  -webkit-appearance: none;
}
</style>
