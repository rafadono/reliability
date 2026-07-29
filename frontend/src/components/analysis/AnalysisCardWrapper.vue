<template>
  <div v-if="active" class="w-full">
    <slot />
  </div>

  <div 
    v-else 
    class="bg-slate-50/90 dark:bg-slate-900/50 border border-dashed border-gray-300 dark:border-slate-800 rounded-xl p-4 flex flex-col justify-between min-h-[135px] transition-all duration-300 shadow-sm hover:border-indigo-400/60 dark:hover:border-indigo-500/50 hover:shadow-md group"
  >
    <div class="space-y-2">
      <div class="flex items-center justify-between gap-2">
        <div class="w-8 h-8 bg-amber-500/10 dark:bg-amber-500/15 border border-amber-500/20 text-amber-600 dark:text-amber-400 rounded-lg flex items-center justify-center shrink-0 shadow-sm group-hover:scale-105 transition-transform">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <span class="px-2 py-0.5 text-[10px] font-bold bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 rounded-full shrink-0">
          {{ $t('workbench.not_configured') }}
        </span>
      </div>

      <div>
        <h4 class="text-xs font-bold text-gray-800 dark:text-slate-200 line-clamp-1">
          {{ title || defaultTitles[nodeType] || $t('workbench.default_module') }}
        </h4>
        <p class="text-[11px] text-gray-500 dark:text-slate-400 line-clamp-2 mt-1 leading-snug">
          {{ $t('workbench.locked_card_desc') }}
        </p>
      </div>
    </div>

    <button
      @click="$emit('navigate', 'workbench')"
      class="mt-3 w-full py-1.5 px-3 bg-indigo-50 dark:bg-indigo-950/40 hover:bg-indigo-100 dark:hover:bg-indigo-900/60 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800/60 rounded-lg text-[11px] font-semibold transition-all active:scale-95 flex items-center justify-center gap-1.5"
    >
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
      {{ $t('workbench.go_to_workbench') }}
    </button>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

defineProps({
  active: {
    type: Boolean,
    required: true
  },
  nodeType: {
    type: String,
    required: true
  },
  title: {
    type: String,
    default: ''
  }
})

defineEmits(['navigate'])

const defaultTitles = computed(() => ({
  pareto: t('workbench.default_titles.pareto'),
  jackknife: t('workbench.default_titles.jackknife'),
  criticality: t('workbench.default_titles.criticality'),
  weibull_kijima: t('workbench.default_titles.weibull_kijima'),
  event_plot: t('workbench.default_titles.event_plot'),
  ram_sim: t('workbench.default_titles.ram_sim'),
  apm: t('workbench.default_titles.apm'),
  trend: t('workbench.default_titles.trend'),
  rcm_fmeca: t('workbench.default_titles.rcm_fmeca'),
  rca_fta: t('workbench.default_titles.rca_fta')
}))
</script>
