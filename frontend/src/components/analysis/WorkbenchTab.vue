<template>
  <div class="space-y-8 animate-fade-in">
    <!-- Encabezado de la pestaña -->
    <div class="border-b border-gray-200 dark:border-slate-700 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <h3 class="text-xl font-semibold text-gray-900 dark:text-white">{{ $t('workbench.title') }}</h3>
        <p class="text-sm text-gray-500 dark:text-slate-400">
          {{ $t('workbench.desc') }}
        </p>
      </div>

      <!-- Barra de herramientas superior -->
      <div class="flex flex-wrap items-center gap-2">
        <!-- Selector de Plantilla Rápida -->
        <select 
          @change="loadTemplate($event.target.value)" 
          class="text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-bold"
        >
          <option value="">{{ $t('workbench.load_template') }}</option>
          <option value="criticality">{{ $t('workbench.template_criticality') }}</option>
          <option value="weibull_opt">{{ $t('workbench.template_weibull_opt') }}</option>
          <option value="ram_sim">{{ $t('workbench.template_ram_sim') }}</option>
          <option value="bad_actors_flow">{{ $t('workbench.template_bad_actors') }}</option>
          <option value="trend_flow">{{ $t('workbench.template_trend_flow') }}</option>
        </select>

        <button 
          @click="executePipeline"
          :disabled="loading || nodes.length === 0"
          class="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 dark:disabled:bg-slate-700 disabled:text-gray-500 text-white text-xs font-bold px-4 py-2 rounded-lg shadow-md flex items-center gap-1.5 transition-all"
        >
          <span v-if="loading" class="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
          <svg v-else class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          </svg>
          {{ $t('workbench.run_flow') }}
        </button>

        <button 
          @click="savePipeline"
          :disabled="nodes.length === 0"
          class="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-700 dark:text-slate-300 text-xs font-bold px-3 py-2 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-all"
        >
          {{ $t('workbench.save_flow') }}
        </button>

        <button 
          @click="clearCanvas"
          class="bg-red-50 dark:bg-red-950/20 text-red-600 dark:text-red-400 text-xs font-bold px-3 py-2 rounded-lg hover:bg-red-100 dark:hover:bg-red-950/40 transition-all"
        >
          {{ $t('workbench.clear_canvas') }}
        </button>
      </div>
    </div>

    <!-- Panel Constructor Principal -->
    <div class="grid grid-cols-1 lg:grid-cols-5 gap-6">
      <!-- Caja de Herramientas de Bloques (Caja Izquierda) -->
      <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-5 border border-gray-100 dark:border-slate-800 space-y-4">
        <h5 class="text-xs font-bold text-gray-700 dark:text-slate-300 uppercase tracking-wider mb-2">{{ $t('workbench.toolbox') }}</h5>
        
        <div class="grid grid-cols-1 gap-2.5">
          <button 
            v-for="tool in blockTools" 
            :key="tool.type"
            @click="addBlock(tool.type)"
            class="flex items-center gap-3 p-3 rounded-xl border border-gray-200 dark:border-slate-700/80 bg-white dark:bg-slate-800 hover:border-indigo-500 hover:shadow-md transition-all text-left group"
          >
            <div 
              class="w-8 h-8 rounded-lg flex items-center justify-center text-white font-extrabold shadow-inner"
              :class="tool.bgColor"
            >
              {{ tool.iconText }}
            </div>
            <div>
              <div class="text-xs font-bold text-gray-800 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                {{ tool.name }}
              </div>
              <div class="text-[9px] text-gray-400 dark:text-slate-500 leading-snug mt-0.5">
                {{ tool.desc }}
              </div>
            </div>
          </button>
        </div>
      </div>

      <!-- Lienzo Principal del Workbench (Caja Derecha / Central) -->
      <div class="lg:col-span-4 relative">
        <div 
          class="w-full h-[600px] bg-slate-50 dark:bg-slate-950/80 border border-gray-200 dark:border-slate-800 rounded-2xl overflow-hidden relative cursor-grab active:cursor-grabbing select-none"
          ref="canvasRef"
          @mousemove="onCanvasMouseMove"
          @mouseup="onCanvasMouseUp"
          @mouseleave="onCanvasMouseLeave"
        >
          <!-- Grid Pattern Cuadriculado Moderno -->
          <div class="absolute inset-0 grid-grid-pattern opacity-40 dark:opacity-20 pointer-events-none"></div>

          <!-- Conexiones SVG (Cables) -->
          <svg class="absolute inset-0 w-full h-full pointer-events-none z-10 overflow-visible">
            <!-- Cable Temporal de Creación -->
            <path 
              v-if="drawingEdge"
              :d="drawTempPath"
              stroke="#6366f1"
              stroke-width="2.5"
              stroke-dasharray="5,5"
              fill="none"
            />
            <!-- Cables del Grafo con Animación -->
            <g v-for="edge in edges" :key="edge.id">
              <!-- Línea Base -->
              <path 
                :d="getEdgePath(edge)"
                stroke="#94a3b8"
                stroke-width="3.5"
                fill="none"
                class="dark:stroke-slate-700"
              />
              <!-- Animación de flujo si el pipeline está cargado/activo -->
              <path 
                :d="getEdgePath(edge)"
                stroke="#6366f1"
                stroke-width="2.5"
                fill="none"
                stroke-dasharray="8,8"
                class="animated-flow"
              />
            </g>
          </svg>

          <!-- Nodos Renderizados en el Lienzo -->
          <div 
            v-for="node in nodes" 
            :key="node.id"
            class="absolute bg-white dark:bg-slate-800 border-2 rounded-xl shadow-lg p-4 w-48 text-left transition-shadow z-20"
            :class="[
              selectedNodeId === node.id 
                ? 'border-indigo-500 ring-2 ring-indigo-100 dark:ring-indigo-950/40 shadow-indigo-100 dark:shadow-none' 
                : 'border-gray-200 dark:border-slate-700/80 hover:border-gray-300 dark:hover:border-slate-600'
            ]"
            :style="{ left: `${node.x}px`, top: `${node.y}px` }"
            @mousedown.stop="onNodeMouseDown($event, node)"
          >
            <!-- Cabecera del Nodo -->
            <div class="flex items-center justify-between border-b border-gray-100 dark:border-slate-700/80 pb-2 mb-2">
              <span class="text-[10px] font-bold text-gray-800 dark:text-slate-200 uppercase truncate pr-1">
                {{ getBlockTool(node.type).name }}
              </span>
              <!-- Indicador de Estado de Ejecución -->
              <div 
                class="w-2.5 h-2.5 rounded-full border shadow-sm transition-colors"
                :class="getStatusColor(node.status)"
                :title="`Estado: ${node.status || 'Sin ejecutar'}`"
              ></div>
            </div>

            <!-- Contenido Dinámico / Resultados del Nodo -->
            <div class="text-[10px] text-gray-500 dark:text-slate-400 space-y-1">
              <!-- DataSource -->
              <div v-if="node.type === 'dataSource'">
                <div v-if="node.output && node.output.rows">
                  Registros: <strong class="text-gray-800 dark:text-slate-300">{{ node.output.rows }}</strong>
                </div>
                <div v-else>Listo para conectar</div>
              </div>

              <!-- Filtro -->
              <div v-if="node.type === 'filter'">
                <div v-if="node.output && node.output.rows">
                  Equipo: <strong class="text-gray-800 dark:text-slate-300 truncate block max-w-full">{{ node.output.equipment || 'Planta' }}</strong>
                  Filtrados: <strong class="text-gray-800 dark:text-slate-300">{{ node.output.rows }}</strong>
                </div>
                <div v-else>Configure filtros</div>
              </div>

              <!-- Weibull -->
              <div v-if="node.type === 'weibull'">
                <div v-if="node.output && node.output.beta">
                  Beta: <strong class="text-indigo-600 dark:text-indigo-400 font-extrabold">{{ node.output.beta }}</strong><br>
                  Eta: <strong class="text-indigo-600 dark:text-indigo-400 font-extrabold">{{ node.output.eta }} hrs</strong><br>
                  MTBF: <strong class="text-gray-700 dark:text-slate-300">{{ node.output.mtbf }} hrs</strong>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-indigo-500 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300 font-extrabold">Ver Gráficos →</button>
                  </div>
                </div>
                <div v-else-if="node.output && node.output.error" class="text-red-500 dark:text-red-400 font-semibold leading-tight">
                  {{ node.output.error }}
                </div>
                <div v-else>Ajuste paramétrico</div>
              </div>

              <!-- Kijima -->
              <div v-if="node.type === 'kijima'">
                <div v-if="node.output && node.output.beta">
                  Modelo: <strong class="text-gray-700 dark:text-slate-300">{{ node.output.model_name }}</strong><br>
                  Beta: <strong class="text-amber-600 dark:text-amber-400">{{ node.output.beta }}</strong><br>
                  Restauración: <strong class="text-amber-600 dark:text-amber-400">{{ node.output.ar || node.output.ap || '--' }}</strong>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-amber-600 hover:text-amber-800 dark:text-amber-400 dark:hover:text-amber-300 font-extrabold">Ver Curvas →</button>
                  </div>
                </div>
                <div v-else-if="node.output && node.output.error" class="text-red-500 dark:text-red-400 font-semibold leading-tight">
                  {{ node.output.error }}
                </div>
                <div v-else>Reparación imperfecta</div>
              </div>

              <!-- FMECA -->
              <div v-if="node.type === 'fmeca'">
                <div v-if="node.output && node.output.records">
                  Registros FMEA: <strong class="text-gray-800 dark:text-slate-300">{{ node.output.records.length }}</strong>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'rcm_fmea')" class="text-[9px] text-rose-500 hover:text-rose-700 dark:text-rose-400 dark:hover:text-rose-300 font-extrabold">Abrir Matriz →</button>
                  </div>
                </div>
                <div v-else>
                  Matriz de RPN
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'rcm_fmea')" class="text-[9px] text-rose-500 hover:text-rose-700 dark:text-rose-400 dark:hover:text-rose-300 font-extrabold">Diseñar FMECA →</button>
                  </div>
                </div>
              </div>

              <!-- RAM Simulator -->
              <div v-if="node.type === 'ramSimulator'">
                <div v-if="node.output && node.output.availability">
                  Disponibilidad: <strong class="text-emerald-600 dark:text-emerald-400 font-black">{{ node.output.availability }}%</strong><br>
                  Assurance: <strong class="text-blue-600 dark:text-blue-400 font-black">{{ node.output.production_assurance }}%</strong>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'ram')" class="text-[9px] text-emerald-500 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300 font-extrabold">Ver Simulación →</button>
                  </div>
                </div>
                <div v-else>
                  Simulación de disponibilidad
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'ram')" class="text-[9px] text-emerald-500 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300 font-extrabold">Abrir Simulador →</button>
                  </div>
                </div>
              </div>

              <!-- Pareto -->
              <div v-if="node.type === 'pareto'">
                <div v-if="node.output && node.output.vital_few">
                  Vitales ({{ node.output.group_by }}):
                  <ul class="list-disc list-inside mt-1 font-bold text-gray-700 dark:text-slate-300">
                    <li v-for="item in node.output.vital_few" :key="item.name" class="truncate">
                      {{ item.name }}: {{ Math.round(item.percentage) }}%
                    </li>
                  </ul>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-orange-500 hover:text-orange-700 dark:text-orange-400 dark:hover:text-orange-300 font-extrabold">Ver Pareto →</button>
                  </div>
                </div>
                <div v-else>Distribución 80/20</div>
              </div>

              <!-- Jackknife -->
              <div v-if="node.type === 'jackknife'">
                <div v-if="node.output && node.output.critical_count !== undefined">
                  Críticos: <strong class="text-red-500 font-extrabold">{{ node.output.critical_count }}</strong><br>
                  Crónicos: <strong class="text-orange-500 font-bold">{{ node.output.chronic_count }}</strong><br>
                  Agudos: <strong class="text-blue-500 font-bold">{{ node.output.acute_count }}</strong>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-teal-500 hover:text-teal-700 dark:text-teal-400 dark:hover:text-teal-300 font-extrabold">Ver Diagrama →</button>
                  </div>
                </div>
                <div v-else>Clasificación costo-riesgo</div>
              </div>

              <!-- Trend -->
              <div v-if="node.type === 'trend'">
                <div v-if="node.output && node.output.failures !== undefined">
                  Disponibilidad: <strong class="text-cyan-600 dark:text-cyan-400 font-black">{{ node.output.availability }}%</strong><br>
                  MTBF: <strong class="text-gray-700 dark:text-slate-300">{{ node.output.mtbf }} hrs</strong><br>
                  Fallas: <strong class="text-gray-700 dark:text-slate-300">{{ node.output.failures }}</strong>
                  <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                    <button @click.stop="$emit('navigate', 'ram')" class="text-[9px] text-cyan-500 hover:text-cyan-700 dark:text-cyan-400 dark:hover:text-cyan-300 font-extrabold">Ver Tendencia →</button>
                  </div>
                </div>
                <div v-else>Tendencia temporal</div>
              </div>
            </div>

            <!-- Puerto de Entrada (Círculo Izquierdo) -->
            <div 
              v-if="node.type !== 'dataSource'"
              class="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-gray-200 dark:bg-slate-700 border-2 border-gray-400 rounded-full cursor-pointer hover:bg-indigo-500 hover:border-white transition-colors"
              @mouseup.stop="onPortMouseUp($event, node.id, 'input')"
              title="Puerto de Entrada"
            ></div>

            <!-- Puerto de Salida (Círculo Derecho) -->
            <div 
              class="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-gray-200 dark:bg-slate-700 border-2 border-gray-400 rounded-full cursor-pointer hover:bg-indigo-500 hover:border-white transition-colors"
              @mousedown.stop="onPortMouseDown($event, node.id, 'output')"
              title="Puerto de Salida"
            ></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Inspector lateral deslizable -->
    <NodeInspector 
      v-if="inspectorOpen"
      :is-open="inspectorOpen"
      :node="selectedNode"
      :available-equipment="availableEquipment"
      :filter-options="currentFilterOptions"
      @close="inspectorOpen = false"
      @delete="deleteBlock"
      @run="executePipeline"
      @filter-changed="updateDropdownCascade"
      @upload-file="$emit('upload-file', $event)"
      @reset="$emit('reset')"
    />
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { apiService } from '../../api'
import NodeInspector from './NodeInspector.vue'
import { sharedState } from '../../sharedState'

const props = defineProps({
  availableEquipment: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['navigate', 'upload-file', 'reset'])

// Variables del lienzo
const nodes = ref([])
const edges = ref([])
const selectedNodeId = ref('')
const loading = ref(false)
const canvasRef = ref(null)

// Variables de arrastre y dibujo de cables
const draggingNode = ref(null)
const dragOffset = ref({ x: 0, y: 0 })
const drawingEdge = ref(null)
const tempMousePos = ref({ x: 0, y: 0 })

// Inspector lateral
const inspectorOpen = ref(false)
const selectedNode = computed(() => nodes.value.find(n => n.id === selectedNodeId.value) || {})

// Cascading filters cache para dropdowns del filter node
const currentFilterOptions = ref({ types: [], mdfs: [] })

watch(selectedNodeId, async (newId) => {
  if (newId) {
    const node = nodes.value.find(n => n.id === newId)
    if (node && node.type === 'filter') {
      await updateDropdownCascade(node.data)
    }
  }
})

// Herramientas / Tipos de bloque disponibles
const blockTools = [
  { type: 'dataSource', name: 'Ingesta de Datos', desc: 'Carga el historial técnico de averías.', iconText: 'IN', bgColor: 'bg-blue-600' },
  { type: 'filter', name: 'Filtro Jerárquico', desc: 'Filtra por equipo, modo y censura.', iconText: 'FL', bgColor: 'bg-indigo-600' },
  { type: 'pareto', name: 'Análisis Pareto', desc: 'Identifica los equipos con mayor impacto.', iconText: 'PA', bgColor: 'bg-orange-600' },
  { type: 'jackknife', name: 'Diagrama Jackknife', desc: 'Diagrama costo-riesgo MTBF vs MTTR.', iconText: 'JK', bgColor: 'bg-teal-600' },
  { type: 'weibull', name: 'Curvas Weibull', desc: 'Ajusta parámetros de ciclo de vida.', iconText: 'WB', bgColor: 'bg-violet-600' },
  { type: 'kijima', name: 'Imperfect Repair', desc: 'Ajusta modelos de edad virtual.', iconText: 'KJ', bgColor: 'bg-amber-600' },
  { type: 'fmeca', name: 'Matriz FMECA', desc: 'Matriz inductiva de prioridad de riesgo.', iconText: 'FM', bgColor: 'bg-rose-600' },
  { type: 'ramSimulator', name: 'Simulador RAM', desc: 'Simulación de producción.', iconText: 'RM', bgColor: 'bg-emerald-600' },
  { type: 'trend', name: 'Tendencia de KPIs', desc: 'Tendencia temporal de MTBF y disponibilidad.', iconText: 'TR', bgColor: 'bg-cyan-600' }
]

const getBlockTool = (type) => blockTools.find(t => t.type === type) || {}

const getStatusColor = (status) => {
  switch (status) {
    case 'success': return 'bg-green-500 border-green-300 animate-pulse'
    case 'error': return 'bg-red-500 border-red-300'
    case 'ready': return 'bg-yellow-500 border-yellow-300'
    default: return 'bg-gray-300 dark:bg-slate-600 border-gray-400'
  }
}

// 1. Agregar bloque
const addBlock = (type) => {
  const id = `${type}-${Date.now()}`
  
  // Parámetros por defecto según el tipo
  const data = {}
  if (type === 'filter') {
    data.equipment = ''
    data.type = []
    data.mdf = []
    data.censored = 'all'
  } else if (type === 'kijima') {
    data.model_type = 1
  } else if (type === 'ramSimulator') {
    data.preventive_efficiency = 0.8
    data.logistics_delay = 4.0
  } else if (type === 'fmeca') {
    data.records = [
      { component: 'Rodamiento Mecánico', mode: 'Desgaste', effect: 'Recalentamiento', severity: 8, occurrence: 4, detection: 3, action: 'Reemplazo' }
    ]
  } else if (type === 'pareto') {
    data.group_by = 'Equipment'
  } else if (type === 'jackknife') {
    data.compare_by = 'Equipment'
  } else if (type === 'trend') {
    data.dummy = true
  }

  nodes.value.push({
    id,
    type,
    data,
    x: 40 + (nodes.value.length * 30) % 200,
    y: 80 + (nodes.value.length * 40) % 200,
    status: 'ready',
    output: null
  })
}

// 2. Eliminar bloque
const deleteBlock = (id) => {
  nodes.value = nodes.value.filter(n => n.id !== id)
  edges.value = edges.value.filter(e => e.source !== id && e.target !== id)
  selectedNodeId.value = ''
  inspectorOpen.value = false
}

// 3. Arrastre de Nodos
const onNodeMouseDown = (e, node) => {
  draggingNode.value = node
  dragOffset.value = {
    x: e.clientX - node.x,
    y: e.clientY - node.y
  }
  selectedNodeId.value = node.id
  inspectorOpen.value = true
}

// 4. Dibujo de Cables (Mousedown Puerto Salida)
const onPortMouseDown = (e, nodeId, portType) => {
  if (canvasRef.value) {
    const rect = canvasRef.value.getBoundingClientRect()
    tempMousePos.value = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    }
  }
  drawingEdge.value = {
    source: nodeId,
    sourcePort: portType
  }
}

// 5. Destino del Cable (Mouseup Puerto Entrada)
const onPortMouseUp = (e, nodeId, portType) => {
  if (drawingEdge.value && drawingEdge.value.source !== nodeId) {
    // Validar si ya existe conexión entrante (1 a 1 en targets)
    const exists = edges.value.some(edge => edge.target === nodeId)
    if (!exists) {
      edges.value.push({
        id: `edge-${Date.now()}`,
        source: drawingEdge.value.source,
        target: nodeId
      })
      // Marcar nodo destino como listo
      const targetNode = nodes.value.find(n => n.id === nodeId)
      if (targetNode) targetNode.status = 'ready'
    }
  }
  drawingEdge.value = null
}

// 6. Eventos de Movimiento en el Lienzo
const onCanvasMouseMove = (e) => {
  if (draggingNode.value) {
    draggingNode.value.x = Math.max(0, Math.min(1000, e.clientX - dragOffset.value.x))
    draggingNode.value.y = Math.max(0, Math.min(600, e.clientY - dragOffset.value.y))
  } else if (drawingEdge.value && canvasRef.value) {
    const rect = canvasRef.value.getBoundingClientRect()
    tempMousePos.value = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    }
  }
}

const onCanvasMouseUp = () => {
  draggingNode.value = null
  drawingEdge.value = null
}

const onCanvasMouseLeave = () => {
  draggingNode.value = null
  drawingEdge.value = null
}

// 7. Resolver curvas Bézier cúbicas SVG de los cables
const getEdgePath = (edge) => {
  const sourceNode = nodes.value.find(n => n.id === edge.source)
  const targetNode = nodes.value.find(n => n.id === edge.target)
  if (!sourceNode || !targetNode) return ''

  // Puerto de salida a la derecha del origen
  const x1 = sourceNode.x + 192 // ancho del nodo
  const y1 = sourceNode.y + 40  // centrado vertical aproximado

  // Puerto de entrada a la izquierda del destino
  const x2 = targetNode.x
  const y2 = targetNode.y + 40

  const controlX1 = x1 + 80
  const controlX2 = x2 - 80

  return `M ${x1} ${y1} C ${controlX1} ${y1}, ${controlX2} ${y2}, ${x2} ${y2}`
}

const drawTempPath = computed(() => {
  if (!drawingEdge.value) return ''
  const sourceNode = nodes.value.find(n => n.id === drawingEdge.value.source)
  if (!sourceNode) return ''

  const x1 = sourceNode.x + 192
  const y1 = sourceNode.y + 40
  const x2 = tempMousePos.value.x
  const y2 = tempMousePos.value.y

  const controlX1 = x1 + 50
  const controlX2 = x2 - 50

  return `M ${x1} ${y1} C ${controlX1} ${y1}, ${controlX2} ${y2}, ${x2} ${y2}`
})

// 8. Actualizar Cascading Dropdowns para Filtros de la UI
const updateDropdownCascade = async (filters) => {
  try {
    const response = await apiService.getAvailableFilters()
    currentFilterOptions.value.types = response.data.types || []
    currentFilterOptions.value.mdfs = response.data.failure_modes || []
  } catch (err) {
    console.error(err)
  }
}

// 9. Ejecutar Pipeline
const executePipeline = async () => {
  loading.value = true
  
  // Marcar todos en ejecución
  nodes.value.forEach(n => {
    n.status = 'running'
  })

  try {
    const payload = {
      nodes: nodes.value.map(n => ({
        id: n.id,
        type: n.type,
        data: n.data,
        x: n.x,
        y: n.y
      })),
      edges: edges.value.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target
      }))
    }

    const response = await apiService.executeWorkbenchPipeline(payload)
    if (response.data.status === 'success') {
      const results = response.data.results
      nodes.value.forEach(n => {
        const nodeRes = results[n.id]
        if (nodeRes) {
          n.status = nodeRes.status
          if (nodeRes.status === 'success') {
            n.output = nodeRes.output
            
            // Sync to global sharedState
            if (n.type === 'weibull') {
              sharedState.weibull = {
                beta: nodeRes.output.beta,
                eta: nodeRes.output.eta,
                mtbf: nodeRes.output.mtbf,
                mttr: nodeRes.output.mttr,
                aic: nodeRes.output.aic,
                bic: nodeRes.output.bic,
                ks_p_value: nodeRes.output.ks_p_value
              }
            } else if (n.type === 'kijima') {
              sharedState.kijima = {
                model_type: n.data.model_type,
                model_name: nodeRes.output.model_name,
                beta: nodeRes.output.beta,
                eta: nodeRes.output.eta,
                ar: nodeRes.output.ar,
                ap: nodeRes.output.ap,
                r2: nodeRes.output.r2
              }
            } else if (n.type === 'fmeca') {
              sharedState.fmecaRecords = nodeRes.output.records
            } else if (n.type === 'ramSimulator') {
              sharedState.ram = {
                preventiveEfficiency: n.data.preventive_efficiency,
                logisticsDelay: n.data.logistics_delay,
                results: nodeRes.output
              }
            } else if (n.type === 'filter') {
              sharedState.filters = {
                equipment: n.data.equipment,
                type: n.data.type,
                mdf: n.data.mdf,
                censored: n.data.censored
              }
            }
          } else {
            n.output = { error: nodeRes.error || 'Fallo de ejecución' }
          }
        }
      })
      
      // Actualizar nodos ejecutados con éxito
      const executed = []
      nodes.value.forEach(n => {
        if (n.status === 'success') {
          executed.push(n.type)
        }
      })
      sharedState.executedNodes = executed
    }
  } catch (error) {
    console.error('Error running pipeline:', error)
    nodes.value.forEach(n => {
      n.status = 'error'
      n.output = { error: 'Error de red / API' }
    })
  } finally {
    loading.value = false
  }
}

// 10. Persistencia: Guardar y Cargar
const savePipeline = async () => {
  const name = prompt('Ingrese el nombre para este pipeline de análisis:')
  if (!name) return

  try {
    const payload = {
      name,
      nodes: nodes.value,
      edges: edges.value
    }
    const response = await apiService.saveWorkbenchPipeline(payload)
    if (response.data.status === 'success') {
      alert(`¡Pipeline '${name}' guardado correctamente!`)
    }
  } catch (error) {
    console.error('Error saving pipeline:', error)
    alert('Fallo al guardar el pipeline.')
  }
}

const clearCanvas = () => {
  nodes.value = []
  edges.value = []
  selectedNodeId.value = ''
  inspectorOpen.value = false
  sharedState.executedNodes = []
}

// 11. Cargar Plantillas Preconfiguradas
const loadTemplate = (type) => {
  if (!type) return
  clearCanvas()

  if (type === 'criticality') {
    nodes.value = [
      { id: 'source-1', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
      { id: 'filter-1', type: 'filter', data: { equipment: '', type: '', mdf: '', censored: 'all' }, x: 290, y: 150, status: 'ready', output: null },
      { id: 'fmeca-1', type: 'fmeca', data: { records: [{ component: 'Sello Principal', mode: 'Fuga de aceite', effect: 'Parada hidráulica', severity: 9, occurrence: 4, detection: 2, action: 'Cambio preventivo' }] }, x: 530, y: 150, status: 'ready', output: null }
    ]
    edges.value = [
      { id: 'edge-1', source: 'source-1', target: 'filter-1' },
      { id: 'edge-2', source: 'filter-1', target: 'fmeca-1' }
    ]
  } else if (type === 'weibull_opt') {
    nodes.value = [
      { id: 'source-2', type: 'dataSource', data: {}, x: 50, y: 180, status: 'ready', output: null },
      { id: 'filter-2', type: 'filter', data: { equipment: '', type: '', mdf: '', censored: 'all' }, x: 290, y: 180, status: 'ready', output: null },
      { id: 'weibull-2', type: 'weibull', data: {}, x: 530, y: 80, status: 'ready', output: null },
      { id: 'kijima-2', type: 'kijima', data: { model_type: 1 }, x: 530, y: 280, status: 'ready', output: null }
    ]
    edges.value = [
      { id: 'edge-3', source: 'source-2', target: 'filter-2' },
      { id: 'edge-4', source: 'filter-2', target: 'weibull-2' },
      { id: 'edge-5', source: 'filter-2', target: 'kijima-2' }
    ]
  } else if (type === 'ram_sim') {
    nodes.value = [
      { id: 'source-3', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
      { id: 'filter-3', type: 'filter', data: { equipment: '', type: [], mdf: [], censored: 'all' }, x: 290, y: 150, status: 'ready', output: null },
      { id: 'ram-3', type: 'ramSimulator', data: { preventive_efficiency: 0.8, logistics_delay: 4.0 }, x: 530, y: 150, status: 'ready', output: null }
    ]
    edges.value = [
      { id: 'edge-6', source: 'source-3', target: 'filter-3' },
      { id: 'edge-7', source: 'filter-3', target: 'ram-3' }
    ]
  } else if (type === 'bad_actors_flow') {
    nodes.value = [
      { id: 'source-4', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
      { id: 'filter-4', type: 'filter', data: { equipment: '', type: [], mdf: [], censored: 'all' }, x: 250, y: 150, status: 'ready', output: null },
      { id: 'pareto-4', type: 'pareto', data: { group_by: 'Equipment' }, x: 450, y: 50, status: 'ready', output: null },
      { id: 'jackknife-4', type: 'jackknife', data: { compare_by: 'Equipment' }, x: 450, y: 250, status: 'ready', output: null }
    ]
    edges.value = [
      { id: 'edge-8', source: 'source-4', target: 'filter-4' },
      { id: 'edge-9', source: 'filter-4', target: 'pareto-4' },
      { id: 'edge-10', source: 'filter-4', target: 'jackknife-4' }
    ]
  } else if (type === 'trend_flow') {
    nodes.value = [
      { id: 'source-5', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
      { id: 'filter-5', type: 'filter', data: { equipment: '', type: [], mdf: [], censored: 'all' }, x: 250, y: 150, status: 'ready', output: null },
      { id: 'trend-5', type: 'trend', data: {}, x: 450, y: 50, status: 'ready', output: null },
      { id: 'weibull-5', type: 'weibull', data: {}, x: 450, y: 250, status: 'ready', output: null }
    ]
    edges.value = [
      { id: 'edge-11', source: 'source-5', target: 'filter-5' },
      { id: 'edge-12', source: 'filter-5', target: 'trend-5' },
      { id: 'edge-13', source: 'filter-5', target: 'weibull-5' }
    ]
  }
  
  // Ejecutar el pipeline de la plantilla de inmediato
  executePipeline()
}
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.grid-grid-pattern {
  background-size: 20px 20px;
  background-image: 
    linear-gradient(to right, #cbd5e1 1px, transparent 1px),
    linear-gradient(to bottom, #cbd5e1 1px, transparent 1px);
}
.dark .grid-grid-pattern {
  background-image: 
    linear-gradient(to right, #334155 1px, transparent 1px),
    linear-gradient(to bottom, #334155 1px, transparent 1px);
}

.animated-flow {
  stroke-dashoffset: 100;
  animation: dash 12s linear infinite;
}

@keyframes dash {
  to {
    stroke-dashoffset: 0;
  }
}
</style>
