import { reactive } from 'vue'

export const sharedState = reactive({
  weibull: null,
  kijima: null,
  fmecaRecords: null,
  ram: null,
  executedNodes: [],
  filters: {
    equipment: '',
    type: '',
    mdf: '',
    censored: 'all'
  }
})

