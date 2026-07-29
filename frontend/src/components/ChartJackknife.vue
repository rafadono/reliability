<template>
  <div class="w-full">
    <div class="flex justify-between items-center mb-1 flex-wrap gap-2">
      <span class="inline-flex items-center gap-1 text-[10px] font-semibold text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-950/20 border border-red-100 dark:border-red-900/30 rounded-full px-2 py-0.5">
        ▲ {{ $t('charts.jackknife.acute_chronic_badge') }}
      </span>
      <div class="flex gap-2">
        <button @click="handleResetZoom" type="button" class="text-[10px] px-2 py-1 rounded bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors">
          {{ $t('charts.common.reset_zoom') }}
        </button>
        <button @click="handleExport" type="button" class="text-[10px] px-2 py-1 rounded bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors">
          {{ $t('charts.common.export_png') }}
        </button>
      </div>
    </div>
    <div class="h-[400px] relative w-full bg-white dark:bg-slate-800 p-2 rounded-lg border border-gray-100 dark:border-slate-700 transition-colors duration-300">
      <canvas ref="chartCanvas"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Chart as ChartJS, registerables } from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import { useI18n } from 'vue-i18n'
import { downloadChartImage } from '../utils/chartExport'

ChartJS.register(...registerables, zoomPlugin)

const { t, locale } = useI18n()

const props = defineProps({
  data: Object,
  scaleX: { type: String, default: 'linear' },
  scaleY: { type: String, default: 'linear' },
  metricY: { type: String, default: 'total' }
})

const chartCanvas = ref(null)
let chartInstance = null

const handleResetZoom = () => {
  if (chartInstance) chartInstance.resetZoom()
}

const handleExport = () => {
  downloadChartImage(chartInstance, 'jackknife-chart.png')
}

// Native custom plugin to draw averages lines
const quadrantPlugin = {
  id: 'quadrant',
  beforeDraw(chart, args, options) {
    const { ctx, chartArea: { left, top, right, bottom }, scales: { x, y } } = chart;
    const avgX = x.getPixelForValue(options.avgX);
    const avgY = y.getPixelForValue(options.avgY);
    
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)'; // Semi-transparent red
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    // Vertical line
    if (avgX >= left && avgX <= right) {
      ctx.moveTo(avgX, top);
      ctx.lineTo(avgX, bottom);
    }
    // Horizontal line
    if (avgY >= top && avgY <= bottom) {
      ctx.moveTo(left, avgY);
      ctx.lineTo(right, avgY);
    }
    
    ctx.stroke();
    ctx.restore();
  }
}

const createChart = () => {
  if (!props.data?.scatter_data || !chartCanvas.value) return

  if (chartInstance) {
    chartInstance.destroy()
  }

  const ctx = chartCanvas.value.getContext('2d')
  const scatterData = props.data.scatter_data.map(d => ({ x: d.x, y: props.metricY === 'average' ? d.y_avg : d.y_total, label: d.name }))

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  chartInstance = new ChartJS(ctx, {
    type: 'scatter',
    plugins: [quadrantPlugin],
    data: {
      datasets: [{
        label: t('charts.jackknife.dataset_label'),
        data: scatterData,
        backgroundColor: '#2563eb',
        borderColor: '#1d4ed8',
        pointRadius: 6,
        pointHoverRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: textColor }
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const point = context.raw
              const yLabel = props.metricY === 'average' ? t('charts.jackknife.hrs_avg') : t('charts.jackknife.hrs_total')
              return `${point.label}: ${point.x} ${t('charts.jackknife.failures')}, ${point.y.toFixed(1)} ${yLabel}`
            }
          }
        },
        quadrant: {
          avgX: props.data.averages?.failures || 0,
          avgY: props.metricY === 'average' ? (props.data.averages?.avg_downtime || 0) : (props.data.averages?.total_downtime || 0)
        },
        zoom: {
          pan: { enabled: true, mode: 'xy' },
          zoom: {
            wheel: { enabled: true },
            pinch: { enabled: true },
            mode: 'xy'
          }
        }
      },
      scales: {
        x: {
          type: props.scaleX,
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: t('charts.jackknife.x_axis'), color: textColor },
          beginAtZero: props.scaleX === 'linear'
        },
        y: {
          type: props.scaleY,
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: props.metricY === 'average' ? t('charts.jackknife.y_axis_avg') : t('charts.jackknife.y_axis_total'), color: textColor },
          beginAtZero: props.scaleY === 'linear'
        }
      }
    }
  })
}

// Handle theme-changed custom event
const handleThemeChange = () => {
  createChart()
}

watch(() => [props.data, props.scaleX, props.scaleY, props.metricY, locale.value], createChart, { deep: true })

onMounted(() => {
  createChart()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>
