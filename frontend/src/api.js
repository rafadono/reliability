import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 0 // No timeout to allow long-running ML operations
})

api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

export const apiService = {
  upload(file) {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  getFilters(equipment, failureType) {
    return api.get('/filters', {
      params: { equipment, failure_type: failureType }
    })
  },

  setFilters(equipment, failureType, failureMode) {
    return api.post('/filters/set', {
      equipment,
      failure_type: failureType,
      failure_mode: failureMode
    })
  },

  getAvailableFilters() {
    return api.get('/data/available-filters')
  },

  resetFilters() {
    return api.get('/data/reset-filters')
  },

  getParetoAnalysis(groupBy, equipment, failureType) {
    return api.post('/analysis/pareto', {
      group_by: groupBy,
      equipment,
      failure_type: failureType
    })
  },

  getJackknifeAnalysis(equipment, failureType, compareBy = 'equipment', typesToUse = null) {
    return api.post('/analysis/jackknife-plot', {
      equipment,
      failure_type: failureType,
      compare_by: compareBy,
      types_to_use: typesToUse
    })
  },

  getCriticalityAnalysis(equipment, failureType, compareBy = 'mode', metricX = 'count') {
    return api.post('/analysis/criticality-plot', {
      equipment,
      failure_type: failureType,
      compare_by: compareBy,
      metric_x: metricX
    })
  },

  fitData(equipment, failureType, typesToFit = null, censoredFailureTypes = null, targetColumn = 'TBX', minTbx = 0.0, excludedIndices = null) {
    return api.post('/analysis/fit', {
      equipment,
      failure_type: failureType,
      types_to_fit: typesToFit,
      censored_failure_types: censoredFailureTypes,
      target_column: targetColumn,
      min_tbx: minTbx,
      excluded_indices: excludedIndices
    })
  },

  fitKijima(equipment, failureType, typesToFit = null, censoredFailureTypes = null, minTbx = 0.0, excludedIndices = null) {
    return api.post('/analysis/kijima-fit', {
      equipment,
      failure_type: failureType,
      types_to_fit: typesToFit,
      censored_failure_types: censoredFailureTypes,
      min_tbx: minTbx,
      excluded_indices: excludedIndices
    })
  },

  getBadActors(equipment, failureType, compareBy = 'equipment', typesToUse = null) {
    return api.post('/analysis/bad-actors', {
      equipment,
      failure_type: failureType,
      compare_by: compareBy,
      types_to_use: typesToUse
    })
  },

  getGrowthAnalysis(equipment, failureType, typesToUse = null) {
    return api.post('/analysis/growth', {
      equipment,
      failure_type: failureType,
      types_to_use: typesToUse
    })
  },

  getEventPlot(equipment, failureType) {
    return api.post('/analysis/event-plot', {
      equipment,
      failure_type: failureType
    })
  },

  getOptimalPm(filters, costPm, costFailure) {
    return api.post('/analysis/optimal-pm', {
      ...filters,
      cost_pm: costPm,
      cost_failure: costFailure
    })
  },

  getConditionalReliability(filters, currentAge, missionTime) {
    return api.post('/analysis/conditional-reliability', {
      ...filters,
      current_age: currentAge,
      mission_time: missionTime
    })
  },

  getKpiTrend(equipment, failureType, typesToUse = null) {
    return api.post('/analysis/kpi-trend', {
      equipment,
      failure_type: failureType,
      types_to_use: typesToUse
    })
  },

  getCommentMining(equipment, failureType, modelName) {
    return api.post('/analysis/comment-mining', {
      equipment,
      failure_type: failureType,
      types_to_use: modelName ? [modelName] : null
    })
  },

  getSummaryStats() {
    return api.get('/stats/summary')
  },

  getModelsStatus() {
    return api.get('/analysis/models-status')
  },

  downloadModels(modelName) {
    return api.post('/analysis/download-model', {
      types_to_use: modelName ? [modelName] : null
    })
  },

  getRcmSuggestions(equipment) {
    return api.post('/analysis/rcm/suggest', { equipment })
  },

  calculateFmeaRpn(severity, occurrence, detection) {
    return api.post('/analysis/fmea/calculate-rpn', { severity, occurrence, detection })
  },

  simulateRam(equipment, preventiveEfficiency, logisticsDelay) {
    return api.post('/analysis/ram/simulate', {
      equipment,
      preventive_efficiency: preventiveEfficiency,
      logistics_delay: logisticsDelay
    })
  },

  getRcaSuggestions(equipment, failureEventDate = null) {
    return api.post('/analysis/rca/suggest', {
      equipment,
      failure_event_date: failureEventDate
    })
  }
}
