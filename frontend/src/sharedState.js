import { reactive } from 'vue'

export const sharedState = reactive({
  weibull: null,
  kijima: null,
  fmecaRecords: null,
  ram: null,
  pareto: null,
  jackknife: null,
  criticality: null,
  event_plot: null,
  apm: null,
  trend: null,
  rcm: null,
  rca: null,
  fta: null,
  comment_mining: null,
  executedNodes: [],
  nodeConfigs: {},
  nodeOutputs: {},
  filters: {
    plant: '',
    equipment: '',
    type: [],
    mdf: [],
    censored_types: [],
    censored_mdfs: []
  }
})
