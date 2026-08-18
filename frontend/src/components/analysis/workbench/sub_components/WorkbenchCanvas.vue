<template>
  <div
    class="workbench-canvas flex-1 min-h-[400px] relative overflow-auto bg-slate-50 dark:bg-slate-950 cursor-grab active:cursor-grabbing select-none"
    @mousedown="startCanvasPan"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
    @wheel="handleWheel"
  >
    <!-- Contenedor Transformable (Pan & Zoom) -->
    <div 
      class="absolute inset-0 origin-top-left transition-transform duration-75"
      :style="{
        transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`
      }"
    >
      <!-- Cuadrícula SVG -->
      <svg class="absolute inset-0 w-[4000px] h-[4000px] pointer-events-none opacity-40 dark:opacity-20">
        <defs>
          <pattern id="grid-pattern" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" class="text-gray-300 dark:text-slate-700" stroke-width="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid-pattern)" />

        <!-- Renderizado de Bordes / Conexiones -->
        <g v-for="edge in edges" :key="edge.id">
          <!-- Capa de Resplandor (Glow Background) para Alta Visibilidad en Modo Oscuro -->
          <path 
            :d="calculateEdgePath(edge)" 
            fill="none" 
            stroke="currentColor" 
            class="text-indigo-500/20 dark:text-indigo-400/40"
            stroke-width="8"
          />
          <!-- Línea Principal Punteada con Alto Contraste -->
          <path 
            :d="calculateEdgePath(edge)" 
            fill="none" 
            stroke="currentColor" 
            class="text-indigo-600 dark:text-indigo-300 transition-all"
            stroke-width="3.5"
            stroke-dasharray="8,4"
          />
        </g>

        <!-- Línea activa durante arrastre de conexión -->
        <path 
          v-if="isConnecting"
          :d="activeConnectionPath"
          fill="none"
          stroke="#f59e0b"
          stroke-width="3"
          stroke-dasharray="4,4"
        />
      </svg>

      <!-- Tarjetas de Nodos -->
      <div
        v-for="node in nodes"
        :key="node.id"
        class="node-card absolute w-52 bg-white dark:bg-slate-900 border rounded-xl shadow-lg transition-shadow duration-200 cursor-move"
        :class="[
          selectedNodeId === node.id ? 'border-indigo-500 ring-2 ring-indigo-500/30 shadow-xl' : 'border-gray-200 dark:border-slate-800 hover:border-gray-300 dark:hover:border-slate-700',
          node.status === 'running' ? 'animate-pulse border-amber-500' : ''
        ]"
        :style="{ left: `${node.x}px`, top: `${node.y}px` }"
        tabindex="0"
        @mousedown.stop="startNodeDrag(node.id, $event)"
        @click.stop="$emit('select-node', node.id)"
        @focus="$emit('select-node', node.id)"
        @keydown="handleNodeKeydown(node.id, $event)"
      >
        <!-- Cabecera del Nodo -->
        <div class="px-3 py-2 border-b border-gray-100 dark:border-slate-800 flex items-center justify-between bg-gray-50/50 dark:bg-slate-800/50 rounded-t-xl">
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full shrink-0" :class="getNodeBadgeColor(node.type)"></span>
            <span class="text-xs font-bold text-gray-800 dark:text-slate-200 truncate max-w-[110px]">{{ getNodeTitle(node.type) }}</span>
          </div>
          <span class="text-[9px] font-extrabold px-1.5 py-0.5 rounded uppercase tracking-wider" :class="getStatusBadgeClass(node.status)">
            {{ node.status }}
          </span>
        </div>

        <!-- Cuerpo / Resumen del Nodo -->
        <div class="p-3 text-[11px] text-gray-600 dark:text-slate-400 min-h-[60px] flex flex-col justify-center">
          <div v-if="node.type === 'dataSource'">
            <div>Filas: <strong class="text-gray-900 dark:text-white">{{ node.output?.rows || node.data?.rows || (availableEquipment.length > 0 ? '5,903' : '0') }}</strong></div>
            <div class="text-[10px] text-gray-400">Planta completa</div>
          </div>

          <div v-if="node.type === 'filter'">
            <div>Equipo: <strong class="text-gray-900 dark:text-white">{{ node.data?.equipment || 'Todos' }}</strong></div>
            <div>Tipos: <span class="text-indigo-600 dark:text-indigo-400 font-semibold">{{ node.data?.type?.length ? node.data.type.join(', ') : 'Todos' }}</span></div>
            <div>Censura: <span class="text-amber-600 dark:text-amber-400 font-semibold">{{ node.data?.censored_types?.length ? node.data.censored_types.join(', ') : 'Ninguno' }}</span></div>
          </div>

          <div v-if="node.type === 'weibull'">
            <div v-if="node.output && node.output.beta">
              Beta: <strong class="text-indigo-600 dark:text-indigo-400">{{ node.output.beta }}</strong> | Eta: <strong class="text-indigo-600 dark:text-indigo-400">{{ node.output.eta }}</strong>
              <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 font-extrabold">Ver Curvas →</button>
              </div>
            </div>
            <div v-else-if="node.output && node.output.error" class="text-red-500 font-semibold leading-tight">
              {{ node.output.error }}
            </div>
            <div v-else>Ajuste paramétrico</div>
          </div>

          <div v-if="node.type === 'kijima'">
            <div v-if="node.output && (node.output.models || node.output.beta)">
              <div v-if="node.output.models && node.output.models.length > 1">
                Modelos: <strong class="text-gray-700 dark:text-slate-300">{{ node.output.models.length }}</strong>
                <div v-for="m in node.output.models.slice(0, 2)" :key="m.model_name" class="truncate text-[10px]">
                  {{ m.model_name }}: β={{ m.beta }}
                </div>
              </div>
              <div v-else>
                Modelo: <strong class="text-gray-700 dark:text-slate-300">{{ node.output.model_name }}</strong><br>
                Beta: <strong class="text-amber-600 dark:text-amber-400">{{ node.output.beta }}</strong>
              </div>
              <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-amber-600 hover:text-amber-800 dark:text-amber-400 font-extrabold">Ver Curvas →</button>
              </div>
            </div>
            <div v-else-if="node.output && node.output.error" class="text-red-500 font-semibold leading-tight">
              {{ node.output.error }}
            </div>
            <div v-else>Reparación imperfecta</div>
          </div>

          <div v-if="node.type === 'pareto'">
            <div v-if="node.output && node.output.error" class="text-red-500 font-semibold leading-tight">
              {{ node.output.error }}
            </div>
            <div v-else-if="node.output && node.output.vital_few">
              Vitales: <strong class="text-emerald-600 dark:text-emerald-400">{{ node.output.vital_few.length }}</strong>
              <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-emerald-600 hover:text-emerald-800 dark:text-emerald-400 font-extrabold">Ver Resultados →</button>
              </div>
            </div>
            <div v-else>Priorización 80/20</div>
          </div>

          <div v-if="node.type === 'jackknife'">
            <div v-if="node.output && node.output.error" class="text-red-500 font-semibold leading-tight">
              {{ node.output.error }}
            </div>
            <div v-else-if="node.output && node.output.critical_count !== undefined">
              Críticos: <strong class="text-pink-600 dark:text-pink-400">{{ node.output.critical_count }}</strong> | Crónicos: <strong>{{ node.output.chronic_count }}</strong>
              <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-pink-600 hover:text-pink-800 dark:text-pink-400 font-extrabold">Ver Resultados →</button>
              </div>
            </div>
            <div v-else>Comparar: {{ node.data?.compare_by || 'Equipo' }}</div>
          </div>

          <div v-if="node.type === 'fmeca'">
            <div>Modos: <strong>{{ node.data?.records?.length || 0 }}</strong></div>
            <div v-if="node.output && node.output.records" class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
              <button @click.stop="$emit('navigate', 'rcm_fmea')" class="text-[9px] text-cyan-600 hover:text-cyan-800 dark:text-cyan-400 font-extrabold">Ver Matriz →</button>
            </div>
          </div>

          <div v-if="node.type === 'trend'">
            <div v-if="node.output && node.output.error" class="text-red-500 font-semibold leading-tight">
              {{ node.output.error }}
            </div>
            <div v-else-if="node.output && node.output.availability !== undefined">
              MTBF: <strong class="text-teal-600 dark:text-teal-400">{{ node.output.mtbf }}</strong> | Disp: <strong>{{ node.output.availability }}%</strong>
              <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                <button @click.stop="$emit('navigate', 'quant')" class="text-[9px] text-teal-600 hover:text-teal-800 dark:text-teal-400 font-extrabold">Ver Tendencia →</button>
              </div>
            </div>
            <div v-else>Perfil mensual KPI</div>
          </div>

          <div v-if="node.type === 'ram'">
            <div v-if="node.output && node.output.availability">
              Disponibilidad: <strong class="text-emerald-600 dark:text-emerald-400">{{ node.output.availability }}%</strong>
              <div class="mt-2 pt-1 border-t border-gray-100 dark:border-slate-700/60 flex justify-end">
                <button @click.stop="$emit('navigate', 'ram')" class="text-[9px] text-orange-600 hover:text-orange-800 dark:text-orange-400 font-extrabold">Ver Simulación →</button>
              </div>
            </div>
            <div v-else-if="node.output && node.output.error" class="text-red-500 font-semibold leading-tight">
              {{ node.output.error }}
            </div>
            <div v-else>Monte Carlo (RAM)</div>
          </div>
        </div>

        <!-- Conector Entrada (Izquierda) -->
        <div 
          class="node-connector absolute -left-2.5 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-white dark:bg-slate-800 border-2 border-indigo-500 flex items-center justify-center hover:scale-125 transition-transform cursor-pointer shadow-sm"
          title="Conectar entrada"
          @mouseup.stop="$emit('complete-connection', node.id)"
        >
          <div class="w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
        </div>

        <!-- Conector Salida (Derecha) -->
        <div 
          class="node-connector absolute -right-2.5 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-white dark:bg-slate-800 border-2 border-amber-500 flex items-center justify-center hover:scale-125 transition-transform cursor-pointer shadow-sm"
          title="Arrastrar para conectar salida"
          @mousedown.stop="$emit('start-connection', node.id, $event)"
        >
          <div class="w-1.5 h-1.5 rounded-full bg-amber-500"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  nodes: { type: Array, required: true },
  edges: { type: Array, required: true },
  selectedNodeId: { type: String, default: null },
  availableEquipment: { type: Array, default: () => [] },
  zoomLevel: { type: Number, default: 1 },
  panOffset: { type: Object, default: () => ({ x: 0, y: 0 }) },
  isConnecting: { type: Boolean, default: false },
  connectingSourceId: { type: String, default: null },
  connectionCursor: { type: Object, default: () => ({ x: 0, y: 0 }) },
  calculateEdgePath: { type: Function, required: true }
})

const emit = defineEmits([
  'select-node',
  'start-node-drag',
  'on-node-drag',
  'stop-node-drag',
  'start-canvas-pan',
  'on-canvas-pan',
  'stop-canvas-pan',
  'handle-wheel',
  'start-connection',
  'update-connection-cursor',
  'complete-connection',
  'end-connection',
  'navigate',
  'delete-node',
  'nudge-node'
])

// Accesibilidad por teclado: con el nodo enfocado (tabindex="0"), Delete/Backspace
// lo elimina (con confirmación, vía el mismo deleteBlock del composable) y las
// flechas lo desplazan una cantidad fija de píxeles, reutilizando la misma
// ruta de mutación de posición que el arrastre con mouse.
const NUDGE_STEP = 10

const handleNodeKeydown = (nodeId, event) => {
  if (event.key === 'Delete' || event.key === 'Backspace') {
    event.preventDefault()
    emit('delete-node', nodeId)
    return
  }

  const arrowDeltas = {
    ArrowUp: { dx: 0, dy: -NUDGE_STEP },
    ArrowDown: { dx: 0, dy: NUDGE_STEP },
    ArrowLeft: { dx: -NUDGE_STEP, dy: 0 },
    ArrowRight: { dx: NUDGE_STEP, dy: 0 }
  }
  const delta = arrowDeltas[event.key]
  if (delta) {
    event.preventDefault()
    emit('nudge-node', { id: nodeId, dx: delta.dx, dy: delta.dy })
  }
}

const activeConnectionPath = computed(() => {
  const sourceNode = props.nodes.find(n => n.id === props.connectingSourceId)
  if (!sourceNode) return ''

  const sx = sourceNode.x + 200
  const sy = sourceNode.y + 45
  const tx = props.connectionCursor.x
  const ty = props.connectionCursor.y
  const dx = Math.abs(tx - sx) / 2

  return `M ${sx} ${sy} C ${sx + dx} ${sy}, ${tx - dx} ${ty}, ${tx} ${ty}`
})

const startCanvasPan = (e) => emit('start-canvas-pan', e)

const onMouseMove = (e) => {
  emit('on-canvas-pan', e)
  emit('on-node-drag', e)
  emit('update-connection-cursor', e)
}

const onMouseUp = (e) => {
  emit('stop-canvas-pan')
  emit('stop-node-drag')
  emit('end-connection')
}

const handleWheel = (e) => emit('handle-wheel', e)

const startNodeDrag = (nodeId, e) => emit('start-node-drag', nodeId, e)

const getNodeTitle = (type) => {
  const titles = {
    dataSource: 'Fuente de Datos',
    filter: 'Filtro de Datos',
    weibull: 'Ajuste Weibull 2P',
    kijima: 'Reparación Kijima',
    pareto: 'Pareto 80/20',
    jackknife: 'Diagrama Jackknife',
    criticality: 'Matriz Criticidad',
    comments: ' NLP Comentarios',
    fmeca: 'FMECA / RPN',
    trend: 'Tendencia KPI',
    ram: 'Simulador RAM'
  }
  return titles[type] || type
}

const getNodeBadgeColor = (type) => {
  const colors = {
    dataSource: 'bg-blue-500',
    filter: 'bg-indigo-500',
    weibull: 'bg-amber-500',
    kijima: 'bg-purple-500',
    pareto: 'bg-emerald-500',
    jackknife: 'bg-pink-500',
    criticality: 'bg-red-500',
    comments: 'bg-orange-500',
    fmeca: 'bg-cyan-500',
    trend: 'bg-teal-500',
    ram: 'bg-orange-500'
  }
  return colors[type] || 'bg-gray-500'
}

const getStatusBadgeClass = (status) => {
  const classes = {
    ready: 'bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-400',
    running: 'bg-amber-100 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400',
    completed: 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400',
    error: 'bg-red-100 dark:bg-red-950/60 text-red-600 dark:text-red-400'
  }
  return classes[status] || classes.ready
}
</script>

<style scoped>
/* Baseline keyboard-accessibility focus indicator for nodes (tabindex="0"). */
.node-card:focus {
  outline: 2px solid #6366f1;
  outline-offset: 2px;
}
</style>
