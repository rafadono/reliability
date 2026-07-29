<template>
  <aside class="w-64 bg-white dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 p-6 overflow-y-auto transition-colors duration-300">
    <div class="space-y-6">


      <div class="bg-gradient-to-br from-blue-600 to-indigo-700 text-white p-4 rounded-xl shadow-md space-y-4">
        <h3 class="font-bold text-xs uppercase tracking-wider border-b border-white/20 pb-2">{{ $t('sidebar.analysis_sections') }}</h3>

        <!-- Pestaña 1: Análisis Cuantitativo -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">{{ $t('sidebar.section_quant') }}</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('quant', 'pareto-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.pareto') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'jackknife-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.jackknife') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'criticality-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.criticality') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'weibull-kijima-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_weibull_kijima') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('quant', 'event-plot-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_event_plot') }}
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 2: RCM & FMECA -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">{{ $t('sidebar.section_rcm_fmeca') }}</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('rcm_fmea', 'rcm-wizard-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_rcm_wizard') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('rcm_fmea', 'fmeca-table-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_fmeca_matrix') }}
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 3: RCA & FTA -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">{{ $t('sidebar.section_rca_fta') }}</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('rca_fta', 'ishikawa-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_rca') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('rca_fta', 'fta-canvas-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_fta') }}
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 4: Aseguramiento RAM -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">{{ $t('sidebar.section_ram') }}</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('ram', 'ram-simulator-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_ram_sim') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('ram', 'apm-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_apm') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('ram', 'trend-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_kpi_trend') }}
              </button>
            </li>
          </ul>
        </div>

        <!-- Pestaña 5: Copiloto IA -->
        <div class="space-y-1">
          <div class="text-[10px] font-extrabold uppercase text-blue-200 tracking-wider">{{ $t('sidebar.section_copilot') }}</div>
          <ul class="space-y-0.5 text-xs pl-2 border-l border-white/10">
            <li>
              <button @click="selectTabAndScroll('copilot', 'ai-chat-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_ai_chat') }}
              </button>
            </li>
            <li>
              <button @click="selectTabAndScroll('copilot', 'ai-card')" class="w-full text-left py-1 px-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none">
                {{ $t('sidebar.link_nlp_mining') }}
              </button>
            </li>
          </ul>
        </div>
      </div>

      <div>
        <h3 class="font-semibold text-gray-900 dark:text-white mb-3">{{ $t('sidebar.navigation') }}</h3>
        <nav class="space-y-2">
          <a href="#" @click.prevent="$emit('export-pdf')" class="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            {{ $t('sidebar.export_pdf') }}
          </a>
          <a href="#" @click.prevent="$emit('notify', t('sidebar.settings_coming_soon'))" class="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
            {{ $t('sidebar.settings') }}
          </a>
        </nav>
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

const emit = defineEmits(['notify', 'export-pdf', 'select-tab-card'])

const selectTabAndScroll = (tabId, cardId) => {
  emit('select-tab-card', { tabId, cardId })
}
</script>
