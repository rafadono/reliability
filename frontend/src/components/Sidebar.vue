<template>
  <aside class="w-64 bg-white dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 p-6 overflow-y-auto transition-colors duration-300">
    <div class="space-y-6">
      <div class="bg-blue-50 dark:bg-slate-900/50 p-4 rounded-lg">
        <h3 class="font-semibold text-gray-900 dark:text-white mb-2">{{ $t('sidebar.quick_actions') }}</h3>
        <button
          @click="triggerUpload"
          :disabled="isLoading"
          class="w-full btn-primary text-sm mb-2"
          :class="isLoading ? 'opacity-50' : ''"
        >
          {{ $t('sidebar.upload_new') }}
        </button>
        <button
          @click="$emit('reset')"
          :disabled="isLoading"
          class="w-full btn-secondary text-sm"
          :class="isLoading ? 'opacity-50' : ''"
        >
          {{ $t('sidebar.reset_filters') }}
        </button>
        <input
          type="file"
          ref="fileInput"
          accept=".csv"
          @change="handleFileChange"
          class="hidden"
        />
      </div>

      <div class="bg-gradient-to-br from-blue-600 to-indigo-700 text-white p-4 rounded-xl shadow-md space-y-4">
        <h3 class="font-bold text-xs uppercase tracking-wider border-b border-white/20 pb-2">Secciones de Análisis</h3>
        
        <!-- Pestaña 1: Análisis Cuantitativo -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">1. Cuantitativo</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('quant', 'pareto-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Análisis de Pareto
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'jackknife-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Diagrama Jackknife
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'criticality-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Matriz de Criticidad
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'weibull-kijima-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Weibull & Kijima
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'event-plot-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Línea de Eventos
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 2: RCM & FMECA -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">2. RCM & FMECA</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('rcm_fmea', 'rcm-wizard-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Asistente RCM (JA1011)
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('rcm_fmea', 'fmeca-table-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Matriz FMECA (IEC 60812)
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 3: RCA & FTA -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">3. RCA & FTA</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('rca_fta', 'ishikawa-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Causa Raíz RCA (IEC 62740)
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('rca_fta', 'fta-canvas-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Árbol de Fallas FTA (IEC 61025)
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 4: Aseguramiento RAM -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">4. Aseguramiento RAM</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('ram', 'ram-simulator-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Simulador RAM (ISO 20815)
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('ram', 'apm-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Bad Actors APM
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('ram', 'trend-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Tendencia de KPIs
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 5: Copiloto IA -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">5. Copiloto IA</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('copilot', 'ai-chat-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Chat Copiloto IA
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('copilot', 'ai-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                Minería de Texto NLP
              </button>
            </li>
          </ul>
        </div>
      </div>

      <div>
        <h3 class="font-semibold text-gray-900 dark:text-white mb-3">Navigation</h3>
        <nav class="space-y-2">
          <a href="#" class="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 font-medium">
            Dashboard
          </a>
          <a href="#" @click.prevent="$emit('export-pdf')" class="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            {{ $t('sidebar.export_pdf') }}
          </a>
          <a href="#" @click.prevent="$emit('notify', 'Settings module coming soon')" class="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
            Settings
          </a>
        </nav>
      </div>

      <div class="pt-4 border-t border-gray-200 dark:border-slate-700">
        <p class="text-xs text-gray-600 dark:text-slate-400">
          <strong>Version:</strong> 1.0<br>
          <strong>Status:</strong> Ready
        </p>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

defineProps({
  isLoading: Boolean
})

const emit = defineEmits(['upload-file', 'reset', 'notify', 'export-pdf', 'select-tab-card'])

const fileInput = ref(null)

const triggerUpload = () => {
  fileInput.value.click()
}

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    emit('upload-file', file)
  }
}

const selectTabAndScroll = (tabId, cardId) => {
  emit('select-tab-card', { tabId, cardId })
}
</script>
