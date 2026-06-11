<template>
  <div class="w-full">
    <canvas ref="chartContainer" class="max-h-96"></canvas>
    <div v-if="data.analysis" class="mt-4 grid grid-cols-2 gap-4">
      <div class="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg border border-blue-100/10">
        <p class="text-xs text-gray-600 dark:text-slate-400">{{ $t('charts.pareto.vital') }}</p>
        <p class="text-lg font-bold text-blue-600 dark:text-blue-400">{{ data.analysis.stats.vital_count }}</p>
      </div>
      <div class="bg-gray-50 dark:bg-slate-900/50 p-3 rounded-lg border border-gray-200/10">
        <p class="text-xs text-gray-600 dark:text-slate-400">{{ $t('charts.pareto.trivial') }}</p>
        <p class="text-lg font-bold text-gray-600 dark:text-slate-300">{{ data.analysis.stats.trivial_count }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { Chart as ChartJS, registerables } from 'chart.js'

ChartJS.register(...registerables)

const { t, locale } = useI18n()

const props = defineProps({
  data: Object
})

const emit = defineEmits(['bar-click'])

const chartContainer = ref(null)
let chartInstance = null

const createChart = () => {
  if (!props.data?.pareto) return

  const ctx = chartContainer.value?.getContext('2d')
  if (!ctx) return

  if (chartInstance) {
    chartInstance.destroy()
  }

  const pareto = props.data.pareto

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  chartInstance = new ChartJS(ctx, {
    type: 'bar',
    data: {
      labels: pareto.items,
      datasets: [
        {
          label: t('charts.pareto.count'),
          data: pareto.counts,
          backgroundColor: '#3b82f6',
          borderColor: '#1d4ed8',
          borderWidth: 1,
          yAxisID: 'y'
        },
        {
          label: t('charts.pareto.cumulative'),
          data: pareto.cumsum_pct,
          type: 'line',
          borderColor: '#a855f7',
          backgroundColor: 'rgba(168, 85, 247, 0.1)',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: '#a855f7',
          yAxisID: 'y1',
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: textColor }
        },
        title: {
          display: true,
          text: `${t('charts.pareto.title')} (${props.data.group_by})`,
          color: textColor
        }
      },
      onClick: (event, elements, chart) => {
        if (elements.length > 0) {
          const index = elements[0].index
          const labelName = chart.data.labels[index]
          emit('bar-click', labelName)
        }
      },
      scales: {
        x: {
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: {
            display: true,
            text: t('charts.pareto.causes'),
            color: textColor
          }
        },
        y: {
          position: 'left',
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: {
            display: true,
            text: t('charts.pareto.count'),
            color: textColor
          }
        },
        y1: {
          position: 'right',
          ticks: { color: textColor },
          grid: { drawOnChartArea: false, color: gridColor },
          title: {
            display: true,
            text: t('charts.pareto.cumulative'),
            color: textColor
          },
          min: 0,
          max: 100
        }
      }
    }
  })
}

// Handle theme-changed custom event
const handleThemeChange = () => {
  createChart()
}

watch(() => props.data, () => {
  createChart()
}, { deep: true })

watch(locale, () => {
  createChart()
})

onMounted(() => {
  createChart()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>
