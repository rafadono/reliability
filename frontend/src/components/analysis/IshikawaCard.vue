<template>
  <div class="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 p-6 transition-all duration-300">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-100 dark:border-slate-700 pb-4 mb-6">
      <div>
        <h4 class="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <svg class="w-5 h-5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          {{ $t('charts.iso_analysis.rca_title') }}
        </h4>
        <p class="text-xs text-gray-500 dark:text-slate-400">{{ $t('charts.iso_analysis.rca_desc') }}</p>
      </div>

      <div class="flex items-center gap-2">
        <label class="text-xs font-semibold text-gray-700 dark:text-slate-300">{{ $t('charts.iso_analysis.equipment_label') }}</label>
        <select 
          v-model="selectedEquipment"
          class="text-sm bg-gray-50 dark:bg-slate-700 border border-gray-200 dark:border-slate-600 text-gray-900 dark:text-white rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-emerald-500 outline-none"
        >
          <option value="" disabled>{{ $t('charts.iso_analysis.select_equipment') }}</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
        <button 
          @click="generateRca" 
          :disabled="!selectedEquipment || loading"
          class="bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-300 dark:disabled:bg-slate-700 disabled:text-gray-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-sm transition-all flex items-center gap-1.5"
        >
          <span v-if="loading" class="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
          {{ $t('charts.iso_analysis.run_rca_suggest') }}
        </button>
      </div>
    </div>

    <!-- No Data State -->
    <div v-if="!hasData" class="text-center py-12">
      <div class="w-16 h-16 bg-emerald-50 dark:bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4 text-emerald-600 dark:text-emerald-400">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      </div>
      <h5 class="text-sm font-semibold text-gray-800 dark:text-slate-200">{{ $t('charts.iso_analysis.no_rca_active') }}</h5>
      <p class="text-xs text-gray-500 dark:text-slate-400 max-w-xs mx-auto mt-1">{{ $t('charts.iso_analysis.rca_instruction') }}</p>
    </div>

    <!-- Contenido del RCA -->
    <div v-else class="space-y-8 animate-fade-in">
      <!-- Los 5 Porqués -->
      <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-5 border border-gray-100 dark:border-slate-800">
        <h5 class="text-sm font-bold text-gray-800 dark:text-slate-200 mb-4 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          {{ $t('charts.iso_analysis.five_whys_title') }}
        </h5>
        
        <div class="relative pl-6 border-l border-gray-200 dark:border-slate-700 space-y-4">
          <div 
            v-for="(item, idx) in fiveWhys" 
            :key="idx"
            class="relative"
          >
            <!-- Pin indicador -->
            <div class="absolute -left-[31px] top-1 w-4.5 h-4.5 rounded-full bg-white dark:bg-slate-800 border-2 border-emerald-500 flex items-center justify-center text-[9px] font-bold text-emerald-600 dark:text-emerald-400">
              {{ idx + 1 }}
            </div>
            
            <div class="text-xs">
              <span class="font-bold text-gray-900 dark:text-white block">{{ item.question }}</span>
              <span class="text-gray-600 dark:text-slate-400 mt-0.5 block italic">{{ item.answer }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Diagrama de Ishikawa -->
      <div class="bg-gray-50 dark:bg-slate-900/40 rounded-xl p-6 border border-gray-100 dark:border-slate-800">
        <h5 class="text-sm font-bold text-gray-800 dark:text-slate-200 mb-6 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          {{ $t('charts.iso_analysis.ishikawa_title') }}
        </h5>

        <div class="ishikawa-container relative py-4 overflow-x-auto min-w-[700px]">
          <!-- Eje central -->
          <div class="absolute left-6 right-32 top-1/2 h-1 bg-emerald-600 dark:bg-emerald-500 -translate-y-1/2"></div>
          <!-- Cabeza de pescado (Efecto) -->
          <div class="absolute right-6 top-1/2 -translate-y-1/2 bg-emerald-600 dark:bg-emerald-500 text-white font-bold text-xs px-4 py-3 rounded-lg shadow-md w-28 text-center">
            {{ selectedEquipment }}<br><span class="text-[10px] font-normal">{{ $t('charts.iso_analysis.falla_label') }}</span>
          </div>

          <!-- Estructura de Espinas -->
          <div class="grid grid-cols-2 gap-x-20 gap-y-16">
            <!-- Espinas Superiores (Maquinaria y Método) -->
            <div class="relative border-b-2 border-dashed border-gray-300 dark:border-slate-700 pb-2">
              <div class="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider mb-2">1. {{ $t('charts.iso_analysis.machinery') }}</div>
              <ul class="space-y-1.5 pl-4 list-disc text-[11px] text-gray-600 dark:text-slate-300">
                <li v-for="(cause, i) in ishikawa.machinery" :key="i">{{ cause }}</li>
              </ul>
            </div>

            <div class="relative border-b-2 border-dashed border-gray-300 dark:border-slate-700 pb-2">
              <div class="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider mb-2">2. {{ $t('charts.iso_analysis.method') }}</div>
              <ul class="space-y-1.5 pl-4 list-disc text-[11px] text-gray-600 dark:text-slate-300">
                <li v-for="(cause, i) in ishikawa.method" :key="i">{{ cause }}</li>
              </ul>
            </div>

            <!-- Espinas Inferiores (Mano de Obra y Medio Ambiente) -->
            <div class="relative border-t-2 border-dashed border-gray-300 dark:border-slate-700 pt-2">
              <div class="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider mb-2">3. {{ $t('charts.iso_analysis.workforce') }}</div>
              <ul class="space-y-1.5 pl-4 list-disc text-[11px] text-gray-600 dark:text-slate-300">
                <li v-for="(cause, i) in ishikawa.workforce" :key="i">{{ cause }}</li>
              </ul>
            </div>

            <div class="relative border-t-2 border-dashed border-gray-300 dark:border-slate-700 pt-2">
              <div class="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider mb-2">4. {{ $t('charts.iso_analysis.environment') }}</div>
              <ul class="space-y-1.5 pl-4 list-disc text-[11px] text-gray-600 dark:text-slate-300">
                <li v-for="(cause, i) in ishikawa.environment" :key="i">{{ cause }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { apiService } from '../../api'

const props = defineProps({
  availableEquipment: {
    type: Array,
    required: true
  }
})

const selectedEquipment = ref('')
const loading = ref(false)
const hasData = ref(false)
const fiveWhys = ref([])
const ishikawa = ref({
  machinery: [],
  method: [],
  workforce: [],
  environment: []
})

const generateRca = async () => {
  if (!selectedEquipment.value) return
  loading.value = true
  try {
    const response = await apiService.getRcaSuggestions(selectedEquipment.value)
    if (response.data.status === 'success') {
      fiveWhys.value = response.data.five_whys || []
      ishikawa.value = response.data.ishikawa || { machinery: [], method: [], workforce: [], environment: [] }
      hasData.value = true
    }
  } catch (error) {
    console.error('Error generating RCA suggestions:', error)
  } finally {
    loading.value = false
  }
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
.ishikawa-container {
  scrollbar-width: thin;
}
</style>
