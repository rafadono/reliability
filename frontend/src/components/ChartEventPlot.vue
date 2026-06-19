<template>
  <div>
    <div :style="{ height: computedHeight + 'px' }" class="relative w-full bg-white dark:bg-slate-800 p-2 rounded-lg border border-gray-100 dark:border-slate-700 transition-colors duration-300">
      <canvas ref="chartCanvas"></canvas>
    </div>

    <div v-if="props.colorMode === 'type' && typeLegendItems.length > 0" class="mt-4 flex flex-wrap gap-3 justify-center text-xs">
      <span class="text-gray-500 dark:text-slate-400 font-medium self-center mr-1">Event Types:</span>
      <div v-for="item in typeLegendItems" :key="item.type" class="flex items-center gap-1.5 bg-gray-50 dark:bg-slate-900/50 px-2 py-1 rounded border border-gray-200 dark:border-slate-700">
        <span class="w-3 h-3 rounded" :style="{ backgroundColor: item.color }"></span>
        <span class="font-medium text-gray-700 dark:text-slate-300">{{ item.type }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import { Chart as ChartJS, registerables } from 'chart.js'
import 'chartjs-adapter-date-fns'

ChartJS.register(...registerables)

const props = defineProps({
  data: Object,
  dateFrom: { type: String, default: '' },
  dateTo: { type: String, default: '' },
  minDuration: { type: Number, default: 0.1 },
  colorMode: { type: String, default: 'type' },
  chartHeight: { type: Number, default: 400 },
  barPercentage: { type: Number, default: 0.6 },
  categoryPercentage: { type: Number, default: 0.8 }
})

const computedHeight = computed(() => {
  if (!props.data || Object.keys(props.data).length === 0) return props.chartHeight
  let count = 0
  for (const eq in props.data) {
    const hasValid = props.data[eq].some(evt => {
      const dur = (new Date(evt.end).getTime() - new Date(evt.start).getTime()) / 3600000
      return dur >= props.minDuration
    })
    if (hasValid) count++
  }
  if (count === 0) return props.chartHeight
  const calculated = count * 52 + 80
  return Math.min(props.chartHeight, Math.max(160, calculated))
})

const chartCanvas = ref(null)
let chartInstance = null
const typeLegendItems = ref([])

const COLORS = [
  'rgba(239, 68, 68, 0.75)',
  'rgba(59, 130, 246, 0.75)',
  'rgba(16, 185, 129, 0.75)',
  'rgba(245, 158, 11, 0.75)',
  'rgba(139, 92, 246, 0.75)',
  'rgba(236, 72, 153, 0.75)',
  'rgba(20, 184, 166, 0.75)',
  'rgba(100, 116, 139, 0.75)',
  'rgba(234, 179, 8, 0.75)',
  'rgba(6, 182, 212, 0.75)'
]

const getTypeColors = (types) => {
  const map = {}
  types.forEach((t, i) => {
    const hue = i < COLORS.length ? null : (i * 137.5) % 360
    map[t] = hue !== null ? COLORS[i] : `hsla(${hue}, 70%, 55%, 0.75)`
  })
  return map
}

const getEquipmentColors = (equipments) => {
  const map = {}
  equipments.forEach((eq, i) => {
    map[eq] = COLORS[i % COLORS.length]
  })
  return map
}

const createChart = () => {
  if (!props.data || Object.keys(props.data).length === 0 || !chartCanvas.value) return
  if (chartInstance) chartInstance.destroy()

  const allTypes = new Set()
  for (const eq in props.data) {
    for (const evt of props.data[eq]) {
      const dur = (new Date(evt.end).getTime() - new Date(evt.start).getTime()) / 3600000
      if (dur >= props.minDuration) allTypes.add(evt.type || 'Unknown')
    }
  }

  const sortedTypes = Array.from(allTypes).sort()
  const typeColorMap = {}
  sortedTypes.forEach((t, i) => {
    typeColorMap[t] = i < COLORS.length ? COLORS[i] : `hsla(${(i * 137.5) % 360}, 70%, 55%, 0.75)`
  })

  const equipmentList = Object.keys(props.data)
  const eqColorMap = getEquipmentColors(equipmentList)

  typeLegendItems.value = props.colorMode === 'type'
    ? sortedTypes.map(t => ({ type: t, color: typeColorMap[t] }))
    : []

  // Single flat dataset — one bar entry per event, grouped by equipment label on y-axis
  const dataPoints = []
  const bgColors = []
  const borderColors = []
  let minTimestamp = Infinity

  for (const eq in props.data) {
    for (const evt of props.data[eq]) {
      const start = new Date(evt.start).getTime()
      const end = new Date(evt.end).getTime()
      const dur = (end - start) / 3600000
      if (dur < props.minDuration) continue
      if (start < minTimestamp) minTimestamp = start

      const color = props.colorMode === 'type'
        ? typeColorMap[evt.type || 'Unknown']
        : eqColorMap[eq]

      dataPoints.push({ x: [start, end], y: eq, type: evt.type, mode: evt.mode })
      bgColors.push(color)
      borderColors.push(color.replace('0.75', '1'))
    }
  }

  const axisMin = props.dateFrom ? new Date(props.dateFrom).getTime() : (minTimestamp !== Infinity ? minTimestamp : undefined)
  const axisMax = props.dateTo ? new Date(props.dateTo).getTime() : undefined

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.05)'

  chartInstance = new ChartJS(chartCanvas.value.getContext('2d'), {
    type: 'bar',
    data: {
      datasets: [{
        label: 'Events',
        data: dataPoints,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1,
        // Single dataset means Chart.js allocates the full category slot to one bar
        barPercentage: props.barPercentage,
        categoryPercentage: props.categoryPercentage,
        maxBarThickness: 28
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(context) {
              const raw = context.raw
              const fmt = ts => new Date(ts).toLocaleString('en-US', {
                day: '2-digit', month: '2-digit', year: 'numeric',
                hour: '2-digit', minute: '2-digit'
              })
              const diffHrs = ((raw.x[1] - raw.x[0]) / 3600000).toFixed(2)
              return [
                `Asset: ${raw.y}`,
                `Type: ${raw.type}`,
                `Mode: ${raw.mode}`,
                `Duration: ${diffHrs} hrs`,
                `Start: ${fmt(raw.x[0])}`,
                `End: ${fmt(raw.x[1])}`
              ]
            }
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          min: axisMin,
          max: axisMax,
          time: { unit: 'month', tooltipFormat: 'dd MMM yyyy' },
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: 'Downtime Timeline', color: textColor }
        },
        y: {
          type: 'category',
          offset: true,
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: 'Equipment', color: textColor }
        }
      }
    }
  })
}

const handleThemeChange = () => createChart()

watch(
  () => [props.data, props.dateFrom, props.dateTo, props.minDuration, props.colorMode, props.chartHeight, props.barPercentage, props.categoryPercentage],
  createChart,
  { deep: true }
)

onMounted(() => {
  createChart()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>