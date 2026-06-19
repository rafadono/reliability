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
  data: Object, // Expects { "Global": [...], "Asset A": [...] }
  selectedKpi: String,
  kpiLabel: String
})

const chartContainer = ref(null)
let chartInstance = null

const palette = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#14b8a6', // teal
  '#6366f1'  // indigo
]

const getKpiColor = (kpi) => {
  switch (kpi) {
    case 'failures':
      return { border: '#ef4444', bg: 'rgba(239, 68, 68, 0.05)' }
    case 'downtime':
      return { border: '#f97316', bg: 'rgba(249, 115, 22, 0.05)' }
    case 'mtbf':
      return { border: '#3b82f6', bg: 'rgba(59, 130, 246, 0.05)' }
    case 'mttr':
      return { border: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.05)' }
    case 'availability':
      return { border: '#10b981', bg: 'rgba(16, 185, 129, 0.05)' }
    default:
      return { border: '#6366f1', bg: 'rgba(99, 102, 241, 0.05)' }
  }
}

const createChart = () => {
  if (!props.data || Object.keys(props.data).length === 0) return

  const ctx = chartContainer.value?.getContext('2d')
  if (!ctx) return

  if (chartInstance) {
    chartInstance.destroy()
  }

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const keys = Object.keys(props.data)
  const firstKey = keys[0]
  const labels = props.data[firstKey].map(item => item.month)

  const datasets = []
  
  keys.forEach((key, index) => {
    const dataPoints = props.data[key].map(item => item[props.selectedKpi] || 0)
    
    let borderColor, bgColor, borderWidth, fill, pointRadius, pointHoverRadius
    
    if (key === 'Global') {
      const colors = getKpiColor(props.selectedKpi)
      borderColor = colors.border
      bgColor = colors.bg
      borderWidth = 3
      fill = true
      pointRadius = 4
      pointHoverRadius = 6
    } else {
      const color = palette[index % palette.length]
      borderColor = color
      bgColor = 'transparent'
      borderWidth = 1.75
      fill = false
      pointRadius = 2
      pointHoverRadius = 4
    }

    datasets.push({
      label: key === 'Global' ? `${props.kpiLabel} (Global)` : key,
      data: dataPoints,
      borderColor: borderColor,
      backgroundColor: bgColor,
      borderWidth: borderWidth,
      fill: fill,
      tension: 0.2,
      pointBackgroundColor: borderColor,
      pointHoverRadius: pointHoverRadius,
      pointRadius: pointRadius
    })
  })

  chartInstance = new ChartJS(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
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
