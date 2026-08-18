<template>
  <div class="bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-gray-200/80 dark:border-slate-800/80 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 z-20 shadow-xs">
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-indigo-50 dark:bg-indigo-950/50 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <div>
          <h2 class="text-base font-extrabold text-gray-900 dark:text-white leading-tight">{{ $t('workbench.header_title') }}</h2>
          <p class="text-[11px] text-gray-500 dark:text-slate-400">{{ $t('workbench.header_subtitle') }}</p>
        </div>
      </div>
    </div>

    <!-- Acciones Principales -->
    <div class="flex items-center gap-2">
      <!-- Botón Ejecutar Pipeline -->
      <button
        @click="$emit('execute')"
        :disabled="loading || validationErrors.length > 0"
        :title="validationErrors.length > 0 ? validationErrors.map(e => e.message).join('\n') : ''"
        class="bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-bold px-4 py-2 rounded-lg text-xs shadow-md hover:shadow-indigo-500/25 flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
      >
        <svg v-if="!loading" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <svg v-else class="animate-spin w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span>{{ loading ? $t('workbench.executing') : $t('workbench.run_pipeline') }}</span>
        <span v-if="!loading && validationErrors.length > 0" class="bg-white/20 text-white text-[9px] font-extrabold px-1.5 py-0.2 rounded-full">
          {{ validationErrors.length }}
        </span>
      </button>

      <!-- Menú Desplegable: Añadir Bloque -->
      <div class="relative group">
        <button class="bg-gray-100 dark:bg-slate-800 hover:bg-gray-200 dark:hover:bg-slate-700 text-gray-800 dark:text-slate-200 font-bold px-3 py-2 rounded-lg text-xs flex items-center gap-1.5 transition-all cursor-pointer">
          <svg class="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          <span>{{ $t('workbench.add_block') }}</span>
          <svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        <div class="absolute right-0 mt-1 w-64 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl shadow-xl hidden group-hover:block z-50 p-2 space-y-1 animate-fade-in max-h-96 overflow-y-auto scrollbar-thin">
          <div class="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-2 py-0.5">{{ $t('workbench.cat_data_filters') }}</div>
          <button @click="$emit('add-block', 'dataSource')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-blue-500"></span> {{ $t('workbench.block_data_source') }}
          </button>
          <button @click="$emit('add-block', 'filter')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-indigo-500"></span> {{ $t('workbench.block_filter') }}
          </button>

          <div class="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-2 py-0.5 pt-1">{{ $t('workbench.cat_quant') }}</div>
          <button @click="$emit('add-block', 'pareto')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span> {{ $t('workbench.block_pareto') }}
          </button>
          <button @click="$emit('add-block', 'jackknife')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-pink-500"></span> {{ $t('workbench.block_jackknife') }}
          </button>
          <button @click="$emit('add-block', 'criticality')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-rose-500"></span> {{ $t('workbench.block_criticality') }}
          </button>
          <button @click="$emit('add-block', 'weibull')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-amber-500"></span> {{ $t('workbench.block_weibull') }}
          </button>
          <button @click="$emit('add-block', 'kijima')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-purple-500"></span> {{ $t('workbench.block_kijima') }}
          </button>
          <button @click="$emit('add-block', 'event_plot')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-sky-500"></span> {{ $t('workbench.block_event_plot') }}
          </button>

          <div class="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-2 py-0.5 pt-1">{{ $t('workbench.cat_ram') }}</div>
          <button @click="$emit('add-block', 'ram')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-orange-500"></span> {{ $t('workbench.block_ram') }}
          </button>
          <button @click="$emit('add-block', 'apm')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-red-500"></span> {{ $t('workbench.block_apm') }}
          </button>
          <button @click="$emit('add-block', 'trend')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-teal-500"></span> {{ $t('workbench.block_trend') }}
          </button>

          <div class="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-2 py-0.5 pt-1">{{ $t('workbench.cat_rcm_rca') }}</div>
          <button @click="$emit('add-block', 'rcm')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-lime-500"></span> {{ $t('workbench.block_rcm') }}
          </button>
          <button @click="$emit('add-block', 'fmeca')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-cyan-500"></span> {{ $t('workbench.block_fmeca') }}
          </button>
          <button @click="$emit('add-block', 'rca')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-yellow-500"></span> {{ $t('workbench.block_rca') }}
          </button>
          <button @click="$emit('add-block', 'fta')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-amber-600"></span> {{ $t('workbench.block_fta') }}
          </button>

          <div class="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-2 py-0.5 pt-1">{{ $t('workbench.cat_ai_copilot') }}</div>
          <button @click="$emit('add-block', 'comment_mining')" class="w-full text-left px-2.5 py-1.5 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex items-center gap-2 transition-colors cursor-pointer">
            <span class="w-2 h-2 rounded-full bg-violet-500"></span> {{ $t('workbench.block_comment_mining') }}
          </button>
        </div>
      </div>

      <!-- Menú Desplegable: Plantillas -->
      <div class="relative group">
        <button class="bg-gray-100 dark:bg-slate-800 hover:bg-gray-200 dark:hover:bg-slate-700 text-gray-800 dark:text-slate-200 font-bold px-3 py-2 rounded-lg text-xs flex items-center gap-1.5 transition-all cursor-pointer">
          <svg class="w-4 h-4 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
          <span>{{ $t('workbench.templates_btn') }}</span>
          <svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        <div class="absolute right-0 mt-1 w-64 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl shadow-xl hidden group-hover:block z-50 p-1.5 space-y-1 animate-fade-in">
          <div class="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-2 py-1">{{ $t('workbench.templates_predefined') }}</div>
          <button @click="$emit('load-template', 'basic_weibull')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_basic_weibull_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_basic_weibull_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'kijima_repair')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_kijima_repair_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_kijima_repair_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'ram_simulation')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_ram_simulation_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_ram_simulation_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'pareto_analysis')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_pareto_analysis_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_pareto_analysis_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'trend_flow')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_trend_flow_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_trend_flow_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'rcm_fmeca_flow')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_rcm_fmeca_flow_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_rcm_fmeca_flow_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'rca_fta_flow')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_rca_fta_flow_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_rca_fta_flow_desc') }}</span>
          </button>
          <button @click="$emit('load-template', 'criticality_apm_flow')" class="w-full text-left px-2.5 py-2 text-xs text-gray-700 dark:text-slate-200 hover:bg-indigo-50 dark:hover:bg-slate-700/60 rounded-lg flex flex-col transition-colors cursor-pointer">
            <span class="font-bold text-gray-900 dark:text-white">{{ $t('workbench.template_criticality_apm_flow_name') }}</span>
            <span class="text-[10px] text-gray-500">{{ $t('workbench.template_criticality_apm_flow_desc') }}</span>
          </button>
        </div>
      </div>

      <!-- Reset Zoom / Fit -->
      <button 
        @click="$emit('reset-zoom')" 
        :title="$t('workbench.reset_zoom_title')"
        class="bg-gray-100 dark:bg-slate-800 hover:bg-gray-200 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 p-2 rounded-lg text-xs transition-all cursor-pointer"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-2V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>

      <!-- Toggle Consola de Registros / Logs -->
      <button 
        @click="$emit('toggle-console')" 
        :title="$t('workbench.logs_console_title')"
        class="bg-gray-100 dark:bg-slate-800 hover:bg-gray-200 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300 px-2.5 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer relative"
        :class="{ 'ring-2 ring-indigo-500': isConsoleOpen }"
      >
        <svg class="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <span>{{ $t('workbench.logs') }}</span>
        <span v-if="errorCount > 0" class="bg-red-500 text-white text-[9px] font-extrabold px-1.5 py-0.2 rounded-full">
          {{ errorCount }}
        </span>
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({
  loading: { type: Boolean, default: false },
  isConsoleOpen: { type: Boolean, default: false },
  errorCount: { type: Number, default: 0 },
  validationErrors: { type: Array, default: () => [] }
})

defineEmits(['execute', 'add-block', 'load-template', 'reset-zoom', 'toggle-console'])
</script>
