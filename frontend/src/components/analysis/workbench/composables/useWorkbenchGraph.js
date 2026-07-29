import { ref, computed, onMounted } from 'vue'
import { apiService } from '../../../../api'
import { sharedState } from '../../../../sharedState'

export function useWorkbenchGraph(emit) {
  const loading = ref(false)
  const availableEquipment = ref([])
  const currentFilterOptions = ref({ plants: [], equipment: [], types: [], mdfs: [] })

  // Cascading dropdowns
  const updateDropdownCascade = async (filters) => {
    try {
      const plant = filters?.plant || undefined
      const eq = filters?.equipment || undefined
      const types = filters?.type ? (Array.isArray(filters.type) ? filters.type : [filters.type]) : undefined
      const response = await apiService.getFilters(plant, eq, types)
      if (response.data) {
        currentFilterOptions.value.plants = response.data.plants || []
        if (response.data.equipment && response.data.equipment.length > 0) {
          currentFilterOptions.value.equipment = response.data.equipment
        }
        currentFilterOptions.value.types = response.data.types || []
        currentFilterOptions.value.mdfs = response.data.failure_modes || []
      }
    } catch (err) {
      console.error('Cascade filter error:', err)
    }
  }

  // Grafo por defecto
  const nodes = ref([
    { id: 'source-1', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
    { id: 'filter-1', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 280, y: 150, status: 'ready', output: null },
    { id: 'weibull-1', type: 'weibull', data: {}, x: 520, y: 80, status: 'ready', output: null },
    { id: 'kijima-1', type: 'kijima', data: { model_types: [1, 2] }, x: 520, y: 250, status: 'ready', output: null }
  ])

  const edges = ref([
    { id: 'edge-1', source: 'source-1', target: 'filter-1' },
    { id: 'edge-2', source: 'filter-1', target: 'weibull-1' },
    { id: 'edge-3', source: 'filter-1', target: 'kijima-1' }
  ])

  const inspectorOpen = ref(false)
  const selectedNodeId = ref(null)

  const selectedNode = computed(() => {
    return nodes.value.find(n => n.id === selectedNodeId.value) || null
  })

  // Drag & Drop
  const isDragging = ref(false)
  const draggedNodeId = ref(null)
  const dragOffset = ref({ x: 0, y: 0 })

  // Pan & Zoom
  const zoomLevel = ref(1)
  const panOffset = ref({ x: 0, y: 0 })
  const isPanning = ref(false)
  const panStart = ref({ x: 0, y: 0 })

  // Conexión manual de nodos
  const isConnecting = ref(false)
  const connectingSourceId = ref(null)
  const connectionCursor = ref({ x: 0, y: 0 })

  const getSummaryStats = async () => {
    try {
      const summaryRes = await apiService.getAvailableFilters()
      if (summaryRes.data) {
        availableEquipment.value = summaryRes.data.equipment || []
        currentFilterOptions.value.types = summaryRes.data.types || []
        currentFilterOptions.value.mdfs = summaryRes.data.failure_modes || []
      }

      const statsRes = await apiService.getSummaryStats().catch(() => null)
      const totalRows = statsRes?.data?.total_records || 0
      nodes.value.forEach(n => {
        if (n.type === 'dataSource') {
          n.output = { rows: totalRows, columns: ['Equipment', 'Type', 'mdf', 'Days', 'Censored', 'Date'] }
          n.data = { rows: totalRows, columns: ['Equipment', 'Type', 'mdf', 'Days', 'Censored', 'Date'] }
          n.status = 'completed'
        }
      })
    } catch (e) {
      console.error('Error in getSummaryStats:', e)
    }
  }

  const selectNode = (nodeId) => {
    selectedNodeId.value = nodeId
    inspectorOpen.value = true
    const targetNode = nodes.value.find(n => n.id === nodeId)
    if (targetNode && targetNode.type === 'filter') {
      updateDropdownCascade(targetNode.data || {})
    }
  }

  const defaultNodeData = (type) => {
    switch (type) {
      case 'filter':
        return { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }
      case 'kijima':
        return { model_types: [1, 2], min_tbx: 0.0, min_ttx: 0.0 }
      case 'weibull':
        return { min_tbx: 0.0, min_ttx: 0.0 }
      case 'pareto':
        return { group_by: 'Equipment' }
      case 'jackknife':
        return { compare_by: 'Equipment' }
      case 'criticality':
        return { compare_by: 'mode', metric_x: 'count' }
      case 'apm':
        return { compare_by: 'equipment' }
      case 'ram':
      case 'ramSimulator':
        return { preventive_efficiency: 0.8, logistics_delay: 4.0 }
      case 'rcm':
      case 'rca':
        return { equipment: '' }
      case 'fta':
        return { top_event: '', gate_type: 'OR', basic_events: [] }
      case 'fmeca':
        // Re-use whatever FMECA records were last edited (e.g. in the RCM & FMECA tab),
        // so adding a new fmeca block doesn't discard the user's existing matrix.
        return { records: sharedState.nodeConfigs?.fmeca?.records ? [...sharedState.nodeConfigs.fmeca.records] : [] }
      default:
        return {}
    }
  }

  const addBlock = (type) => {
    const newId = `${type}-${Date.now().toString().slice(-4)}`
    const newNode = {
      id: newId,
      type,
      data: defaultNodeData(type),
      x: 100 + (nodes.value.length * 30) % 400,
      y: 100 + (nodes.value.length * 30) % 300,
      status: 'ready',
      output: null
    }
    nodes.value.push(newNode)
    selectNode(newId)
  }

  const deleteBlock = (nodeId) => {
    if (!window.confirm('¿Seguro que quieres eliminar este bloque? Esta acción no se puede deshacer.')) {
      return
    }
    nodes.value = nodes.value.filter(n => n.id !== nodeId)
    edges.value = edges.value.filter(e => e.source !== nodeId && e.target !== nodeId)
    if (selectedNodeId.value === nodeId) {
      inspectorOpen.value = false
      selectedNodeId.value = null
    }
  }

  // Nudge de posición por teclado (accesibilidad): mueve un nodo por un delta
  // en px, reutilizando el mismo mecanismo de mutación que el arrastre con mouse.
  const nudgeNode = (nodeId, dx, dy) => {
    const node = nodes.value.find(n => n.id === nodeId)
    if (node) {
      node.x = Math.round(node.x + dx)
      node.y = Math.round(node.y + dy)
    }
  }

  // Métodos para Conexión manual
  const startConnection = (sourceId, event) => {
    isConnecting.value = true
    connectingSourceId.value = sourceId
    const canvasEl = event.currentTarget.closest('.workbench-canvas')
    if (canvasEl) {
      const rect = canvasEl.getBoundingClientRect()
      connectionCursor.value = {
        x: (event.clientX - rect.left - panOffset.value.x) / zoomLevel.value,
        y: (event.clientY - rect.top - panOffset.value.y) / zoomLevel.value
      }
    }
  }

  const updateConnectionCursor = (event) => {
    if (!isConnecting.value) return
    const canvasEl = event.currentTarget
    const rect = canvasEl.getBoundingClientRect()
    connectionCursor.value = {
      x: (event.clientX - rect.left - panOffset.value.x) / zoomLevel.value,
      y: (event.clientY - rect.top - panOffset.value.y) / zoomLevel.value
    }
  }

  const completeConnection = (targetId) => {
    if (isConnecting.value && connectingSourceId.value && connectingSourceId.value !== targetId) {
      const exists = edges.value.some(e => e.source === connectingSourceId.value && e.target === targetId)
      if (!exists) {
        edges.value.push({
          id: `edge-${Date.now().toString().slice(-4)}`,
          source: connectingSourceId.value,
          target: targetId
        })
      }
    }
    endConnection()
  }

  const endConnection = () => {
    isConnecting.value = false
    connectingSourceId.value = null
  }

  // Métodos de Arrastre de Nodos
  const startNodeDrag = (nodeId, event) => {
    isDragging.value = true
    draggedNodeId.value = nodeId
    const node = nodes.value.find(n => n.id === nodeId)
    if (node) {
      dragOffset.value = {
        x: event.clientX / zoomLevel.value - node.x,
        y: event.clientY / zoomLevel.value - node.y
      }
    }
  }

  const onNodeDrag = (event) => {
    if (isDragging.value && draggedNodeId.value) {
      const node = nodes.value.find(n => n.id === draggedNodeId.value)
      if (node) {
        node.x = Math.round(event.clientX / zoomLevel.value - dragOffset.value.x)
        node.y = Math.round(event.clientY / zoomLevel.value - dragOffset.value.y)
      }
    }
  }

  const stopNodeDrag = () => {
    isDragging.value = false
    draggedNodeId.value = null
  }

  // Métodos de Pan & Zoom
  const startCanvasPan = (event) => {
    if (event.target.closest('.node-card') || event.target.closest('.node-connector')) return
    isPanning.value = true
    panStart.value = {
      x: event.clientX - panOffset.value.x,
      y: event.clientY - panOffset.value.y
    }
  }

  const onCanvasPan = (event) => {
    if (isPanning.value) {
      panOffset.value = {
        x: event.clientX - panStart.value.x,
        y: event.clientY - panStart.value.y
      }
    }
  }

  const stopCanvasPan = () => {
    isPanning.value = false
  }

  const handleWheel = (event) => {
    event.preventDefault()
    const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9
    const newZoom = Math.min(Math.max(0.4, zoomLevel.value * zoomFactor), 2.0)
    zoomLevel.value = newZoom
  }

  const resetZoom = () => {
    zoomLevel.value = 1
    panOffset.value = { x: 0, y: 0 }
  }

  // Cálculo de Rutas SVG Curvas
  const calculateEdgePath = (edge) => {
    const sourceNode = nodes.value.find(n => n.id === edge.source)
    const targetNode = nodes.value.find(n => n.id === edge.target)
    if (!sourceNode || !targetNode) return ''

    const sx = sourceNode.x + 200
    const sy = sourceNode.y + 45
    const tx = targetNode.x
    const ty = targetNode.y + 45
    const dx = Math.abs(tx - sx) / 2

    return `M ${sx} ${sy} C ${sx + dx} ${sy}, ${tx - dx} ${ty}, ${tx} ${ty}`
  }

  const workbenchLogs = ref([])
  const isConsoleOpen = ref(false)

  const toggleConsole = () => {
    isConsoleOpen.value = !isConsoleOpen.value
  }

  const clearLogs = () => {
    workbenchLogs.value = []
  }

  const errorCount = computed(() => {
    return workbenchLogs.value.filter(l => l.level === 'ERROR').length
  })

  const getNodeTitle = (node) => {
    const titles = {
      dataSource: 'Fuente de Datos',
      filter: 'Filtro Jerárquico',
      weibull: 'Weibull 2P',
      kijima: 'Kijima I / II',
      pareto: 'Pareto 80/20',
      jackknife: 'Jackknife RCM',
      criticality: 'Matriz Criticidad 3D',
      event_plot: 'Línea de Eventos',
      ram: 'Simulador RAM',
      apm: 'Bad Actors APM',
      trend: 'Tendencia KPI',
      rcm: 'Asistente RCM',
      fmeca: 'Matriz FMECA',
      rca: 'Causa Raíz RCA',
      fta: 'Árbol de Fallas FTA',
      comment_mining: 'Minería NLP'
    }
    return titles[node.type] || node.type
  }

  const validatePipelineGraph = () => {
    const errors = []

    const sourceNodes = nodes.value.filter(n => n.type === 'dataSource')
    if (sourceNodes.length === 0) {
      errors.push({
        node_id: 'pipeline',
        message: 'El flujo de trabajo debe contener al menos un bloque de "Fuente de Datos".'
      })
    }

    const parentMap = {}
    edges.value.forEach(e => {
      if (!parentMap[e.target]) parentMap[e.target] = []
      parentMap[e.target].push(e.source)
    })

    const isReachableFromSource = (nodeId, visited = new Set()) => {
      if (visited.has(nodeId)) return false
      visited.add(nodeId)

      const node = nodes.value.find(n => n.id === nodeId)
      if (!node) return false
      if (node.type === 'dataSource') return true

      const parents = parentMap[nodeId] || []
      return parents.some(pId => isReachableFromSource(pId, new Set(visited)))
    }

    nodes.value.forEach(node => {
      if (node.type !== 'dataSource') {
        const parents = parentMap[node.id] || []
        if (parents.length === 0) {
          node.status = 'error'
          errors.push({
            node_id: node.id,
            message: `El bloque "${getNodeTitle(node)}" (${node.id}) está desconectado. Debe conectarse a un filtro o fuente de datos aguas arriba.`
          })
        } else if (!isReachableFromSource(node.id)) {
          node.status = 'error'
          errors.push({
            node_id: node.id,
            message: `El bloque "${getNodeTitle(node)}" (${node.id}) no posee una ruta válida que provenga de una Fuente de Datos.`
          })
        }
      }

      if (node.type === 'kijima') {
        const models = node.data?.model_types || []
        if (!Array.isArray(models) || models.length === 0) {
          node.status = 'error'
          errors.push({
            node_id: node.id,
            message: `El bloque Kijima (${node.id}) requiere seleccionar al menos un tipo de modelo (Kijima I o II) en su inspector.`
          })
        }
      }

      if (node.type === 'fta') {
        const events = node.data?.basic_events || []
        const invalid = events.some(e => !e.name || !e.name.trim())
        if (invalid) {
          node.status = 'error'
          errors.push({
            node_id: node.id,
            message: `El bloque FTA (${node.id}) tiene un evento básico sin nombre. Completa o elimina el evento en su inspector.`
          })
        }
      }
    })

    return errors
  }

  // Ejecutar Pipeline DAG
  const executePipeline = async () => {
    const validationErrors = validatePipelineGraph()
    if (validationErrors.length > 0) {
      isConsoleOpen.value = true
      validationErrors.forEach(err => {
        workbenchLogs.value.unshift({
          id: `val-err-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
          timestamp: new Date().toLocaleTimeString(),
          node_id: err.node_id,
          node_type: 'error',
          level: 'ERROR',
          message: `Configuración incompleta: ${err.message}`
        })
      })
      return
    }

    loading.value = true
    nodes.value.forEach(n => { n.status = 'running' })

    try {
      await getSummaryStats()
      const payload = {
        nodes: nodes.value.map(n => ({
          id: n.id,
          type: n.type,
          data: n.data || {},
          x: n.x || 0,
          y: n.y || 0
        })),
        edges: edges.value.map(e => ({
          id: e.id,
          source: e.source,
          target: e.target
        }))
      }

      const res = await apiService.executeWorkbenchPipeline(payload)
      if (res.data) {
        if (res.data.results) {
          const executedTypes = new Set(sharedState.executedNodes || [])
          nodes.value.forEach(n => {
            if (res.data.results[n.id]) {
              const resObj = res.data.results[n.id]
              n.output = resObj.output || resObj
              n.status = resObj.status === 'success' ? 'completed' : 'error'
              if (resObj.status === 'success') {
                executedTypes.add(n.type)

                // Store node data & outputs in sharedState
                sharedState.nodeConfigs[n.type] = n.data || {}
                sharedState.nodeConfigs[n.id] = n.data || {}
                sharedState.nodeOutputs[n.type] = resObj.output || resObj
                sharedState.nodeOutputs[n.id] = resObj.output || resObj

                if (n.type === 'filter') {
                  const d = n.data || {}
                  sharedState.filters.plant = d.plant || ''
                  sharedState.filters.equipment = d.equipment || ''
                  sharedState.filters.type = Array.isArray(d.type) ? d.type : (d.type ? [d.type] : [])
                  sharedState.filters.mdf = Array.isArray(d.mdf) ? d.mdf : (d.mdf ? [d.mdf] : [])
                  sharedState.filters.censored_types = d.censored_types || []
                  sharedState.filters.censored_mdfs = d.censored_mdfs || []
                }

                if (n.type === 'weibull') {
                  sharedState.weibull = resObj.output
                  executedTypes.add('weibull_kijima')
                }
                if (n.type === 'kijima') {
                  sharedState.kijima = resObj.output
                  executedTypes.add('weibull_kijima')
                }
                if (n.type === 'pareto') {
                  sharedState.pareto = resObj.output
                }
                if (n.type === 'jackknife') {
                  sharedState.jackknife = resObj.output
                }
                if (n.type === 'criticality') {
                  sharedState.criticality = resObj.output
                }
                if (n.type === 'event_plot') {
                  sharedState.event_plot = resObj.output
                }
                if (n.type === 'ram' || n.type === 'ramSimulator') {
                  sharedState.ram = resObj.output
                  executedTypes.add('ram_sim')
                }
                if (n.type === 'apm') {
                  sharedState.apm = resObj.output
                }
                if (n.type === 'trend') {
                  sharedState.trend = resObj.output
                }
                if (n.type === 'rcm') {
                  sharedState.rcm = resObj.output
                }
                if (n.type === 'fmeca') {
                  sharedState.fmecaRecords = resObj.output
                  executedTypes.add('fmeca')
                }
                if (n.type === 'rca') {
                  sharedState.rca = resObj.output
                }
                if (n.type === 'fta') {
                  sharedState.fta = resObj.output
                }
                if (n.type === 'comment_mining') {
                  sharedState.comment_mining = resObj.output
                }
              }
            } else {
              n.status = 'ready'
            }
          })
          sharedState.executedNodes = Array.from(executedTypes)
        }
        if (res.data.logs && Array.isArray(res.data.logs)) {
          workbenchLogs.value = [...res.data.logs, ...workbenchLogs.value].slice(0, 100)
        }
      }
    } catch (err) {
      console.error('Error executing pipeline:', err)
      nodes.value.forEach(n => { n.status = 'error' })
      workbenchLogs.value.unshift({
        id: `err-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        node_id: 'pipeline',
        node_type: 'error',
        level: 'ERROR',
        message: `Error general en ejecución: ${err.message || err}`
      })
    } finally {
      loading.value = false
    }
  }

  // Cargar Plantillas de Flujo
  const loadPipelineTemplate = (type) => {
    if (type === 'basic_weibull') {
      nodes.value = [
        { id: 'source-1', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-1', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 280, y: 150, status: 'ready', output: null },
        { id: 'weibull-1', type: 'weibull', data: {}, x: 520, y: 150, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-1', source: 'source-1', target: 'filter-1' },
        { id: 'edge-2', source: 'filter-1', target: 'weibull-1' }
      ]
    } else if (type === 'kijima_repair') {
      nodes.value = [
        { id: 'source-2', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-2', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 280, y: 150, status: 'ready', output: null },
        { id: 'kijima-2', type: 'kijima', data: { model_types: [1, 2, 3] }, x: 520, y: 150, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-3', source: 'source-2', target: 'filter-2' },
        { id: 'edge-4', source: 'filter-2', target: 'kijima-2' }
      ]
    } else if (type === 'ram_simulation') {
      nodes.value = [
        { id: 'source-3', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-3', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 250, y: 150, status: 'ready', output: null },
        { id: 'weibull-3', type: 'weibull', data: {}, x: 450, y: 50, status: 'ready', output: null },
        { id: 'ram-3', type: 'ram', data: {}, x: 650, y: 150, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-5', source: 'source-3', target: 'filter-3' },
        { id: 'edge-6', source: 'filter-3', target: 'weibull-3' },
        { id: 'edge-7', source: 'weibull-3', target: 'ram-3' }
      ]
    } else if (type === 'pareto_analysis') {
      nodes.value = [
        { id: 'source-4', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-4', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 250, y: 150, status: 'ready', output: null },
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
        { id: 'filter-5', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 250, y: 150, status: 'ready', output: null },
        { id: 'trend-5', type: 'trend', data: {}, x: 450, y: 50, status: 'ready', output: null },
        { id: 'weibull-5', type: 'weibull', data: {}, x: 450, y: 250, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-11', source: 'source-5', target: 'filter-5' },
        { id: 'edge-12', source: 'filter-5', target: 'trend-5' },
        { id: 'edge-13', source: 'filter-5', target: 'weibull-5' }
      ]
    } else if (type === 'rcm_fmeca_flow') {
      nodes.value = [
        { id: 'source-6', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-6', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 250, y: 150, status: 'ready', output: null },
        { id: 'rcm-6', type: 'rcm', data: defaultNodeData('rcm'), x: 480, y: 50, status: 'ready', output: null },
        { id: 'fmeca-6', type: 'fmeca', data: defaultNodeData('fmeca'), x: 480, y: 250, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-14', source: 'source-6', target: 'filter-6' },
        { id: 'edge-15', source: 'filter-6', target: 'rcm-6' },
        { id: 'edge-16', source: 'filter-6', target: 'fmeca-6' }
      ]
    } else if (type === 'rca_fta_flow') {
      nodes.value = [
        { id: 'source-7', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-7', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 250, y: 150, status: 'ready', output: null },
        { id: 'rca-7', type: 'rca', data: defaultNodeData('rca'), x: 480, y: 50, status: 'ready', output: null },
        { id: 'fta-7', type: 'fta', data: defaultNodeData('fta'), x: 480, y: 250, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-17', source: 'source-7', target: 'filter-7' },
        { id: 'edge-18', source: 'filter-7', target: 'rca-7' },
        { id: 'edge-19', source: 'filter-7', target: 'fta-7' }
      ]
    } else if (type === 'criticality_apm_flow') {
      nodes.value = [
        { id: 'source-8', type: 'dataSource', data: {}, x: 50, y: 150, status: 'ready', output: null },
        { id: 'filter-8', type: 'filter', data: { equipment: '', type: [], mdf: [], censored_types: [], censored_mdfs: [] }, x: 250, y: 150, status: 'ready', output: null },
        { id: 'criticality-8', type: 'criticality', data: defaultNodeData('criticality'), x: 480, y: 50, status: 'ready', output: null },
        { id: 'apm-8', type: 'apm', data: defaultNodeData('apm'), x: 480, y: 250, status: 'ready', output: null }
      ]
      edges.value = [
        { id: 'edge-20', source: 'source-8', target: 'filter-8' },
        { id: 'edge-21', source: 'filter-8', target: 'criticality-8' },
        { id: 'edge-22', source: 'filter-8', target: 'apm-8' }
      ]
    }
    executePipeline()
  }

  onMounted(() => {
    getSummaryStats()
  })

  return {
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
  }
}
