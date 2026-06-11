<template>
  <div class="w-full">
    <div class="h-[400px] relative w-full bg-white dark:bg-slate-800 p-2 rounded-lg border border-gray-100 dark:border-slate-700 transition-colors duration-300">
      <canvas ref="chartCanvas"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Chart as ChartJS, registerables } from 'chart.js'

ChartJS.register(...registerables)

const props = defineProps({
  data: Object,
  scaleX: { type: String, default: 'linear' },
  scaleY: { type: String, default: 'linear' },
  metricY: { type: String, default: 'total' }
})

const chartCanvas = ref(null)
let chartInstance = null

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
        label: 'Equipment/Types',
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
              const yLabel = props.metricY === 'average' ? 'hrs/failure (Avg)' : 'hrs Downtime (Total)'
              return `${point.label}: ${point.x} Failures, ${point.y.toFixed(1)} ${yLabel}`
            }
          }
        },
        quadrant: {
          avgX: props.data.averages?.failures || 0,
          avgY: props.metricY === 'average' ? (props.data.averages?.avg_downtime || 0) : (props.data.averages?.total_downtime || 0)
        }
      },
      scales: {
        x: {
          type: props.scaleX,
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: 'Number of Failures (Frequency)', color: textColor },
          beginAtZero: props.scaleX === 'linear'
        },
        y: {
          type: props.scaleY,
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: { display: true, text: props.metricY === 'average' ? 'Average Downtime (MTTR)' : 'Total Downtime', color: textColor },
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

watch(() => [props.data, props.scaleX, props.scaleY, props.metricY], createChart, { deep: true })

onMounted(() => {
  createChart()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>
