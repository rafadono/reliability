<template>
  <div class="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 p-6 transition-all duration-300">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-100 dark:border-slate-700 pb-4 mb-6">
      <div>
        <h4 class="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <svg class="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
          </svg>
          {{ $t('charts.iso_analysis.rcm_title') }}
        </h4>
        <p class="text-xs text-gray-500 dark:text-slate-400">{{ $t('charts.iso_analysis.rcm_desc') }}</p>
      </div>

      <!-- Selector de Equipo -->
      <div class="flex items-center gap-2">
        <label class="text-xs font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.equipment_label') }}</label>
        <select 
          v-model="selectedEquipment"
          class="text-sm bg-gray-50 dark:bg-slate-700 border border-gray-200 dark:border-slate-600 text-gray-900 dark:text-white rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-blue-500 outline-none"
        >
          <option value="" disabled>{{ $t('charts.iso_analysis.select_equipment') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <button 
          @click="generateRcmSuggestions" 
          :disabled="!selectedEquipment || loading"
          class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-slate-700 disabled:text-gray-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-sm transition-all flex items-center gap-1.5"
        >
          <span v-if="loading" class="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
          {{ $t('charts.iso_analysis.generate_ai') }}
        </button>
      </div>
    </div>

    <!-- No Data State -->
    <div v-if="rcmSheets.length === 0" class="text-center py-12">
      <div class="w-16 h-16 bg-blue-50 dark:bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600 dark:text-blue-400">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <h5 class="text-sm font-semibold text-gray-800 dark:text-slate-200">{{ $t('charts.iso_analysis.no_active_analysis') }}</h5>
      <p class="text-xs text-gray-500 dark:text-slate-400 max-w-xs mx-auto mt-1">{{ $t('charts.iso_analysis.rcm_instruction') }}</p>
    </div>

    <!-- Stepper Wizard -->
    <div v-else class="space-y-6">
      <!-- Selector de Modos Sugeridos -->
      <div class="bg-gray-50 dark:bg-slate-700/50 rounded-lg p-3 flex items-center justify-between border border-gray-100 dark:border-slate-700">
        <div class="text-xs">
          <span class="font-bold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.failure_mode_analyzed') }}</span>
          <span class="ml-2 text-blue-600 dark:text-blue-400 font-medium">{{ currentSheet.mode }}</span>
        </div>
        <div class="flex gap-1">
          <button 
            v-for="(sheet, idx) in rcmSheets" 
            :key="idx" 
            @click="activeSheetIndex = idx; currentStep = 0"
            class="px-2.5 py-1 text-xs rounded transition-all"
            :class="activeSheetIndex === idx ? 'bg-blue-600 text-white font-semibold' : 'bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-300 hover:bg-gray-300 dark:hover:bg-slate-600'"
          >
            {{ $t('charts.iso_analysis.mode_label') }} {{ idx + 1 }}
          </button>
        </div>
      </div>

      <!-- Pasos del Stepper -->
      <div class="relative flex items-center justify-between w-full">
        <div class="absolute left-0 right-0 top-1/2 h-0.5 bg-gray-200 dark:bg-slate-700 -translate-y-1/2 z-0"></div>
        <div 
          v-for="(step, idx) in steps" 
          :key="idx" 
          class="relative z-10 flex flex-col items-center"
        >
          <button 
            @click="currentStep = idx"
            class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 border-2"
            :class="[
              currentStep === idx 
                ? 'bg-blue-600 text-white border-blue-600 ring-4 ring-blue-100 dark:ring-blue-900/30' 
                : idx < currentStep 
                  ? 'bg-green-500 text-white border-green-500' 
                  : 'bg-white dark:bg-slate-800 text-gray-400 dark:text-slate-500 border-gray-200 dark:border-slate-700 hover:border-blue-300'
            ]"
          >
            <span v-if="idx < currentStep">✓</span>
            <span v-else>{{ idx + 1 }}</span>
          </button>
          <span class="text-[10px] font-medium text-gray-500 dark:text-slate-400 mt-1.5 hidden md:block">{{ step.title }}</span>
        </div>
      </div>

      <!-- Área de Contenido del Paso -->
      <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-5 border border-gray-100 dark:border-slate-800/80 min-h-[160px] flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-3">
            <span class="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider">{{ $t('charts.iso_analysis.question_prefix') }} {{ currentStep + 1 }}: {{ steps[currentStep].question }}</span>
            <span class="text-xs text-gray-400 dark:text-slate-500">{{ steps[currentStep].title }}</span>
          </div>

          <!-- Inputs interactivos según el paso -->
          <div class="space-y-2">
            <label class="block text-xs font-semibold text-gray-600 dark:text-slate-300 mb-1">
              {{ steps[currentStep].description }}
            </label>
            <textarea 
              v-model="currentSheet[steps[currentStep].key]"
              rows="3"
              class="w-full bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-sm text-gray-900 dark:text-white rounded-lg p-3 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent outline-none transition-all resize-none shadow-inner"
            ></textarea>
          </div>
        </div>

        <div class="flex justify-between items-center mt-6 pt-4 border-t border-gray-100 dark:border-slate-800">
          <button 
            @click="prevStep" 
            :disabled="currentStep === 0"
            class="px-4 py-1.5 text-xs font-bold text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800 disabled:opacity-50 rounded-lg transition-all"
          >
            {{ $t('charts.iso_analysis.back') }}
          </button>
          
          <button 
            v-if="currentStep < steps.value.length - 1" 
            @click="nextStep"
            class="bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-4 py-1.5 rounded-lg shadow transition-all"
          >
            {{ $t('charts.iso_analysis.next') }}
          </button>
          
          <button 
            v-else 
            @click="saveRcmSheet"
            class="bg-green-600 hover:bg-green-700 text-white text-xs font-bold px-5 py-1.5 rounded-lg shadow-md transition-all flex items-center gap-1"
          >
            {{ $t('charts.iso_analysis.save_sheet') }}
          </button>
        </div>
      </div>

      <!-- Resumen de las 7 preguntas -->
      <div class="bg-blue-50/50 dark:bg-slate-900/20 border border-blue-100/50 dark:border-slate-700/50 rounded-lg p-4">
        <h5 class="text-xs font-bold text-blue-800 dark:text-blue-300 mb-2.5">{{ $t('charts.iso_analysis.consolidated_sheet') }}</h5>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div v-for="(step, idx) in steps" :key="idx" class="border-b border-dashed border-gray-200 dark:border-slate-800 pb-2">
            <span class="font-semibold text-gray-700 dark:text-slate-300">{{ idx + 1 }}. {{ step.title }}:</span>
            <p class="text-gray-600 dark:text-slate-400 mt-0.5 line-clamp-2" :title="currentSheet[step.key]">
              {{ currentSheet[step.key] || 'No definido' }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'

const props = defineProps({
  availableEquipment: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['rcm-saved'])

const { t } = useI18n()

const selectedEquipment = ref('')
const loading = ref(false)
const activeSheetIndex = ref(0)
const currentStep = ref(0)
const rcmSheets = ref([])

const steps = computed(() => [
  { key: 'function', title: t('charts.iso_analysis.step_title_function'), question: t('charts.iso_analysis.step_q_function'), description: t('charts.iso_analysis.step_d_function') },
  { key: 'functional_failure', title: t('charts.iso_analysis.step_title_failure'), question: t('charts.iso_analysis.step_q_failure'), description: t('charts.iso_analysis.step_d_failure') },
  { key: 'mode', title: t('charts.iso_analysis.step_title_mode'), question: t('charts.iso_analysis.step_q_mode'), description: t('charts.iso_analysis.step_d_mode') },
  { key: 'effect', title: t('charts.iso_analysis.step_title_effect'), question: t('charts.iso_analysis.step_q_effect'), description: t('charts.iso_analysis.step_d_effect') },
  { key: 'consequence', title: t('charts.iso_analysis.step_title_consequence'), question: t('charts.iso_analysis.step_q_consequence'), description: t('charts.iso_analysis.step_d_consequence') },
  { key: 'proactive_task', title: t('charts.iso_analysis.step_title_task'), question: t('charts.iso_analysis.step_q_task'), description: t('charts.iso_analysis.step_d_task') },
  { key: 'alternative_action', title: t('charts.iso_analysis.step_title_alt'), question: t('charts.iso_analysis.step_q_alt'), description: t('charts.iso_analysis.step_d_alt') }
])

const currentSheet = computed(() => {
  if (rcmSheets.value.length === 0) return {}
  return rcmSheets.value[activeSheetIndex.value]
})

const generateRcmSuggestions = async () => {
  if (!selectedEquipment.value) return
  loading.value = true
  try {
    const response = await apiService.getRcmSuggestions(selectedEquipment.value)
    if (response.data.status === 'success') {
      rcmSheets.value = response.data.rcm_sheets.map(sheet => ({
        function: sheet.function || '',
        functional_failure: sheet.functional_failure || '',
        mode: sheet.mode || '',
        effect: sheet.effect || '',
        consequence: sheet.consequence || '',
        proactive_task: sheet.proactive_task || '',
        alternative_action: sheet.alternative_action || 'Operar hasta la falla / Evaluación de rediseño'
      }))
      activeSheetIndex.value = 0
      currentStep.value = 0
    }
  } catch (error) {
    console.error('Error al generar RCM:', error)
  } finally {
    loading.value = false
  }
}

const nextStep = () => {
  if (currentStep.value < steps.value.length - 1) {
    currentStep.value++
  }
}

const prevStep = () => {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

const saveRcmSheet = () => {
  emit('rcm-saved', {
    equipment: selectedEquipment.value,
    sheet: { ...currentSheet.value }
  })
  alert(t('charts.iso_analysis.saved_success'))
}
</script>
