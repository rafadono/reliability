<template>
  <div class="h-[70vh] min-h-[400px] flex flex-col relative overflow-hidden bg-slate-50 dark:bg-slate-950 rounded-2xl border border-gray-200 dark:border-slate-800 shadow-md">
    <!-- Barra de Herramientas Superior -->
    <WorkbenchToolbar 
      :loading="loading" 
      :is-console-open="isConsoleOpen"
      :error-count="errorCount"
      @execute="executePipeline" 
      @add-block="addBlock" 
      @load-template="loadPipelineTemplate" 
      @reset-zoom="resetZoom" 
      @toggle-console="toggleConsole"
    />

    <!-- Superficie Principal de Dibujo / Canvas -->
    <WorkbenchCanvas 
      :nodes="nodes" 
      :edges="edges" 
      :selected-node-id="selectedNodeId" 
      :available-equipment="availableEquipment" 
      :zoom-level="zoomLevel" 
      :pan-offset="panOffset" 
      :is-connecting="isConnecting" 
      :connecting-source-id="connectingSourceId" 
      :connection-cursor="connectionCursor" 
      :calculate-edge-path="calculateEdgePath" 
      @select-node="selectNode" 
      @start-node-drag="startNodeDrag" 
      @on-node-drag="onNodeDrag" 
      @stop-node-drag="stopNodeDrag" 
      @start-canvas-pan="startCanvasPan" 
      @on-canvas-pan="onCanvasPan" 
      @stop-canvas-pan="stopCanvasPan" 
      @handle-wheel="handleWheel" 
      @start-connection="startConnection"
      @update-connection-cursor="updateConnectionCursor"
      @complete-connection="completeConnection"
      @end-connection="endConnection"
      @navigate="$emit('navigate', $event)"
      @delete-node="deleteBlock"
      @nudge-node="({ id, dx, dy }) => nudgeNode(id, dx, dy)"
    />

    <!-- Consola de Logs & Diagnósticos en la parte inferior -->
    <WorkbenchLogConsole 
      :is-open="isConsoleOpen" 
      :logs="workbenchLogs" 
      @close="isConsoleOpen = false" 
      @clear="clearLogs" 
    />

    <!-- Inspector lateral deslizable -->
    <NodeInspector 
      v-if="inspectorOpen && selectedNode"
      :is-open="inspectorOpen"
      :node="selectedNode"
      :available-equipment="availableEquipment"
      :filter-options="currentFilterOptions"
      @close="inspectorOpen = false"
      @delete="deleteBlock"
      @run="executePipeline"
      @filter-changed="updateDropdownCascade"
      @upload-file="handleFileUpload"
      @reset="handleReset"
    />
  </div>
</template>

<script setup>
import { useWorkbenchGraph } from './workbench/composables/useWorkbenchGraph'
import WorkbenchToolbar from './workbench/sub_components/WorkbenchToolbar.vue'
import WorkbenchCanvas from './workbench/sub_components/WorkbenchCanvas.vue'
import WorkbenchLogConsole from './workbench/sub_components/WorkbenchLogConsole.vue'
import NodeInspector from './NodeInspector.vue'

const emit = defineEmits(['upload-file', 'reset', 'navigate'])

const {
  loading,
  nodes,
  edges,
  availableEquipment,
  currentFilterOptions,
  inspectorOpen,
  selectedNodeId,
  selectedNode,
  zoomLevel,
  panOffset,
  isConnecting,
  connectingSourceId,
  connectionCursor,
  selectNode,
  addBlock,
  deleteBlock,
  nudgeNode,
  startConnection,
  updateConnectionCursor,
  completeConnection,
  endConnection,
  startNodeDrag,
  onNodeDrag,
  stopNodeDrag,
  startCanvasPan,
  onCanvasPan,
  stopCanvasPan,
  handleWheel,
  resetZoom,
  calculateEdgePath,
  updateDropdownCascade,
  executePipeline,
  loadPipelineTemplate,
  getSummaryStats,
  workbenchLogs,
  isConsoleOpen,
  errorCount,
  toggleConsole,
  clearLogs
} = useWorkbenchGraph(emit)

const handleFileUpload = async (file) => {
  emit('upload-file', file)
  setTimeout(async () => {
    await getSummaryStats()
    workbenchLogs.value.unshift({
      id: `upload-${Date.now()}`,
      timestamp: new Date().toLocaleTimeString(),
      node_id: 'dataSource',
      node_type: 'dataSource',
      level: 'INFO',
      message: 'Nuevo archivo subido exitosamente. Configura los bloques (Filtro, Weibull, Kijima, etc.) y presiona "Ejecutar Pipeline" cuando estés listo.'
    })
  }, 1000)
}

const handleReset = async () => {
  emit('reset')
  setTimeout(async () => {
    await getSummaryStats()
  }, 1000)
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
</style>
