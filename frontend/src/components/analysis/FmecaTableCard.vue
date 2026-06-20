<template>
  <div class="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 p-6 transition-all duration-300">
    <div class="border-b border-gray-100 dark:border-slate-700 pb-4 mb-6">
      <h4 class="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <svg class="w-5 h-5 text-indigo-600 dark:text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        {{ $t('charts.iso_analysis.fmeca_title') }}
      </h4>
      <p class="text-xs text-gray-500 dark:text-slate-400">{{ $t('charts.iso_analysis.fmeca_desc') }}</p>
    </div>

    <!-- Controles de Fila Nueva -->
    <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-4 border border-gray-100 dark:border-slate-800/80 mb-6 space-y-4">
      <h5 class="text-xs font-bold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.register_new_mode') }}</h5>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.component_system') }}</label>
          <input 
            v-model="newRecord.component" 
            type="text" 
            :placeholder="$t('charts.iso_analysis.placeholder_component')" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.failure_mode') }}</label>
          <input 
            v-model="newRecord.mode" 
            type="text" 
            :placeholder="$t('charts.iso_analysis.placeholder_mode')" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.operational_effect') }}</label>
          <input 
            v-model="newRecord.effect" 
            type="text" 
            :placeholder="$t('charts.iso_analysis.placeholder_effect')" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.mitigating_action') }}</label>
          <input 
            v-model="newRecord.action" 
            type="text" 
            :placeholder="$t('charts.iso_analysis.placeholder_action')" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.severity') }}</label>
          <select 
            v-model.number="newRecord.severity" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
          </select>
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.occurrence') }}</label>
          <select 
            v-model.number="newRecord.occurrence" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
          </select>
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.detection') }}</label>
          <select 
            v-model.number="newRecord.detection" 
            class="text-xs bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
          </select>
        </div>
        <button 
          @click="addFmecaRow" 
          :disabled="!newRecord.component || !newRecord.mode"
          class="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 dark:disabled:bg-slate-700 disabled:text-gray-500 text-white text-xs font-bold py-2 rounded-lg shadow-sm transition-all"
        >
          {{ $t('charts.iso_analysis.add_record') }}
        </button>
      </div>
    </div>

    <!-- Tabla de Registros -->
    <div class="overflow-x-auto border border-gray-100 dark:border-slate-700 rounded-xl">
      <table class="w-full text-left text-xs border-collapse">
        <thead>
          <tr class="bg-gray-50 dark:bg-slate-900/60 border-b border-gray-100 dark:border-slate-700">
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.table_component') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.table_mode') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.table_effect') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300 text-center">{{ $t('charts.iso_analysis.table_s') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300 text-center">{{ $t('charts.iso_analysis.table_o') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300 text-center">{{ $t('charts.iso_analysis.table_d') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300 text-center">{{ $t('charts.iso_analysis.table_rpn') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.table_criticality') }}</th>
            <th class="p-3 font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.table_action') }}</th>
            <th class="p-3 text-center"></th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100 dark:divide-slate-800">
          <tr 
            v-for="(row, idx) in tableData" 
            :key="idx" 
            class="hover:bg-gray-50/50 dark:hover:bg-slate-700/30 transition-colors"
          >
            <td class="p-3 text-gray-900 dark:text-white font-medium">{{ row.component }}</td>
            <td class="p-3 text-gray-600 dark:text-slate-300">{{ row.mode }}</td>
            <td class="p-3 text-gray-600 dark:text-slate-400 max-w-xs truncate" :title="row.effect">{{ row.effect }}</td>
            <td class="p-3 text-center">
              <select v-model.number="row.severity" @change="recalculateRow(idx)" class="bg-transparent text-center border-b border-dashed border-gray-300 dark:border-slate-600 text-gray-900 dark:text-white py-0.5 outline-none focus:border-indigo-500">
                <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
              </select>
            </td>
            <td class="p-3 text-center">
              <select v-model.number="row.occurrence" @change="recalculateRow(idx)" class="bg-transparent text-center border-b border-dashed border-gray-300 dark:border-slate-600 text-gray-900 dark:text-white py-0.5 outline-none focus:border-indigo-500">
                <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
              </select>
            </td>
            <td class="p-3 text-center">
              <select v-model.number="row.detection" @change="recalculateRow(idx)" class="bg-transparent text-center border-b border-dashed border-gray-300 dark:border-slate-600 text-gray-900 dark:text-white py-0.5 outline-none focus:border-indigo-500">
                <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
              </select>
            </td>
            <td class="p-3 text-center font-bold text-gray-955 dark:text-white">{{ row.rpn }}</td>
            <td class="p-3">
              <span 
                class="px-2 py-0.5 text-[10px] font-bold rounded-full border"
                :class="getBadgeStyles(row.category)"
              >
                {{ translateCategory(row.category) }} ({{ row.rpn }})
              </span>
            </td>
            <td class="p-3 text-gray-600 dark:text-slate-400">{{ row.action }}</td>
            <td class="p-3 text-center">
              <button 
                @click="removeRow(idx)" 
                class="text-red-500 hover:text-red-700 transition-colors p-1"
                :title="$t('charts.iso_analysis.delete_record')"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </td>
          </tr>
          <tr v-if="tableData.length === 0">
            <td colspan="10" class="text-center p-8 text-gray-400 dark:text-slate-500">
              {{ $t('charts.iso_analysis.no_modes_entered') }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'

const { t } = useI18n()

const tableData = ref([
  { component: 'Rodamiento Principal', mode: 'Desgaste mecánico', effect: 'Vibraciones altas y recalentamiento', severity: 8, occurrence: 4, detection: 3, rpn: 96, category: 'Medio', action: 'Monitoreo mensual de vibración' },
  { component: 'Sello de Eje', mode: 'Desgaste de elastómero', effect: 'Fuga de lubricante sintético', severity: 6, occurrence: 6, detection: 2, rpn: 72, category: 'Medio', action: 'Reemplazo preventivo en PM2' },
  { component: 'Motor de Tracción', mode: 'Cortocircuito devanado', effect: 'Parada súbita de equipo y bloqueo', severity: 9, occurrence: 2, detection: 2, rpn: 36, category: 'Bajo', action: 'Pruebas de aislamiento anuales' }
])

const newRecord = ref({
  component: '',
  mode: '',
  effect: '',
  severity: 5,
  occurrence: 5,
  detection: 5,
  action: ''
})

const translateCategory = (cat) => {
  if (!cat) return ''
  const normalized = cat.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
  if (normalized === 'critico' || normalized === 'critical') return t('charts.iso_analysis.criticidad_critico')
  if (normalized === 'alto' || normalized === 'high') return t('charts.iso_analysis.criticidad_alto')
  if (normalized === 'medio' || normalized === 'medium') return t('charts.iso_analysis.criticidad_medio')
  return t('charts.iso_analysis.criticidad_bajo')
}

const getCategoryAndRpn = async (severity, occurrence, detection) => {
  try {
    const response = await apiService.calculateFmeaRpn(severity, occurrence, detection)
    if (response.data.status === 'success') {
      return {
        rpn: response.data.rpn,
        category: response.data.category
      }
    }
  } catch (error) {
    console.error('Error calculando RPN:', error)
  }
  // Fallback local calculations
  const rpn = severity * occurrence * detection
  let category = 'Bajo'
  if (rpn >= 300) category = 'Crítico'
  else if (rpn >= 150) category = 'Alto'
  else if (rpn >= 50) category = 'Medio'
  return { rpn, category }
}

const addFmecaRow = async () => {
  const result = await getCategoryAndRpn(newRecord.value.severity, newRecord.value.occurrence, newRecord.value.detection)
  tableData.value.push({
    component: newRecord.value.component,
    mode: newRecord.value.mode,
    effect: newRecord.value.effect,
    severity: newRecord.value.severity,
    occurrence: newRecord.value.occurrence,
    detection: newRecord.value.detection,
    rpn: result.rpn,
    category: result.category,
    action: newRecord.value.action || 'Pendiente definición'
  })
  
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

const recalculateRow = async (idx) => {
  const row = tableData.value[idx]
  const result = await getCategoryAndRpn(row.severity, row.occurrence, row.detection)
  row.rpn = result.rpn
  row.category = result.category
}

const removeRow = (idx) => {
  tableData.value.splice(idx, 1)
}

const getBadgeStyles = (category) => {
  if (!category) return 'bg-green-50 text-green-700 border-green-200'
  const normalized = category.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
  switch (normalized) {
    case 'critico':
    case 'critical':
      return 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-800'
    case 'alto':
    case 'high':
      return 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-800'
    case 'medio':
    case 'medium':
      return 'bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-950/40 dark:text-yellow-300 dark:border-yellow-800'
    default:
      return 'bg-green-50 text-green-700 border-green-200 dark:bg-green-950/40 dark:text-green-300 dark:border-green-800'
  }
}
</script>
