import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000
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
    return api.post('/analysis/jackknife', {
      equipment,
      failure_type: failureType,
      compare_by: compareBy,
      types_to_use: typesToUse
    })
  },

  fitData(equipment, failureType, typesToFit = null, censoredFailureTypes = null, targetColumn = 'TBX') {
    return api.post('/analysis/fit', {
      equipment,
      failure_type: failureType,
      types_to_fit: typesToFit,
      censored_failure_types: censoredFailureTypes,
      target_column: targetColumn
    })
  },

  getBadActors(equipment, failureType, compareBy = 'equipment') {
    return api.post('/analysis/bad-actors', {
      equipment,
      failure_type: failureType,
      compare_by: compareBy
    })
  },

  getGrowthAnalysis(equipment, failureType) {
    return api.post('/analysis/growth', {
      equipment,
      failure_type: failureType
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

  getKpiTrend(equipment, failureType) {
    return api.post('/analysis/kpi-trend', {
      equipment,
      failure_type: failureType
    })
  },

  getCommentMining(equipment, failureType) {
    return api.post('/analysis/comment-mining', {
      equipment,
      failure_type: failureType
    })
  },

  getSummaryStats() {
    return api.get('/stats/summary')
  }
}
