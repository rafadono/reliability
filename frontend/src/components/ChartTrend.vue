<template>
  <div class="w-full">
    <div class="h-80 relative">
      <canvas ref="chartContainer" class="max-h-80"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Chart as ChartJS, registerables } from 'chart.js'

ChartJS.register(...registerables)

const props = defineProps({
  data: Array,
  selectedKpi: String,
  kpiLabel: String
})

const chartContainer = ref(null)
let chartInstance = null

const getKpiColor = (kpi) => {
  switch (kpi) {
    case 'failures':
      return { border: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)' }
    case 'downtime':
      return { border: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' }
    case 'mtbf':
      return { border: '#3b82f6', bg: 'rgba(59, 130, 246, 0.1)' }
    case 'mttr':
      return { border: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.1)' }
    case 'availability':
      return { border: '#10b981', bg: 'rgba(16, 185, 129, 0.1)' }
    default:
      return { border: '#6366f1', bg: 'rgba(99, 102, 241, 0.1)' }
  }
}

const createChart = () => {
  if (!props.data || props.data.length === 0) return

  const ctx = chartContainer.value?.getContext('2d')
  if (!ctx) return

  if (chartInstance) {
    chartInstance.destroy()
  }

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const labels = props.data.map(item => item.month)
  const dataPoints = props.data.map(item => item[props.selectedKpi] || 0)
  const colors = getKpiColor(props.selectedKpi)

  chartInstance = new ChartJS(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: props.kpiLabel,
          data: dataPoints,
          borderColor: colors.border,
          backgroundColor: colors.bg,
          borderWidth: 2,
          fill: true,
          tension: 0.2,
          pointBackgroundColor: colors.border,
          pointHoverRadius: 6,
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: textColor }
        },
        tooltip: {
          mode: 'index',
          intersect: false
        }
      },
      scales: {
        x: {
          ticks: { color: textColor },
          grid: { color: gridColor }
        },
        y: {
          ticks: { color: textColor },
          grid: { color: gridColor },
          beginAtZero: true
        }
      }
    }
  })
}

watch(() => props.data, createChart, { deep: true })
watch(() => props.selectedKpi, createChart)

const handleThemeChange = () => {
  createChart()
}

onMounted(() => {
  createChart()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.destroy()
  }
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>
