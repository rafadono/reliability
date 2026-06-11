<template>
  <div class="h-[400px] relative w-full bg-white dark:bg-slate-800 p-2 rounded-lg border border-gray-100 dark:border-slate-700 transition-colors duration-300">
    <canvas ref="chartCanvas"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Chart as ChartJS, registerables } from 'chart.js'
import 'chartjs-adapter-date-fns'; // Adapter for time scale

ChartJS.register(...registerables);

const props = defineProps({
  data: Object, // Expects { "Equipo A": ["date1", "date2"], ... }
  dateFrom: { type: String, default: '' },
  dateTo: { type: String, default: '' },
  minDuration: { type: Number, default: 0.1 }
});

const chartCanvas = ref(null);
let chartInstance = null;

const createChart = () => {
  if (!props.data || Object.keys(props.data).length === 0 || !chartCanvas.value) return;

  if (chartInstance) {
    chartInstance.destroy();
  }

  const datasets = [];
  const colors = [
    'rgba(59, 130, 246, 0.7)', 
    'rgba(239, 68, 68, 0.7)', 
    'rgba(16, 185, 129, 0.7)', 
    'rgba(245, 158, 11, 0.7)', 
    'rgba(139, 92, 246, 0.7)', 
    'rgba(236, 72, 153, 0.7)'
  ];
  let colorIndex = 0;
  let minTimestamp = Infinity;

  for (const equipmentName in props.data) {
    const events = props.data[equipmentName];
    const dataPoints = [];

    for (const evt of events) {
      const start = new Date(evt.start).getTime();
      const end = new Date(evt.end).getTime();
      const durationHrs = (end - start) / 3600000;

      if (durationHrs < props.minDuration) {
        continue;
      }

      if (start < minTimestamp) minTimestamp = start;
      dataPoints.push({ 
        x: [start, end], 
        y: equipmentName,
        mode: evt.mode,
        type: evt.type
      });
    }

    if (dataPoints.length > 0) {
      datasets.push({
        label: equipmentName,
        data: dataPoints,
        backgroundColor: colors[colorIndex % colors.length],
        borderColor: colors[colorIndex % colors.length].replace('0.7', '1'),
        borderWidth: 1,
        barPercentage: 0.6,
        categoryPercentage: 0.8
      });
      colorIndex++;
    }
  }
  
  const axisMin = props.dateFrom ? new Date(props.dateFrom).getTime() : (minTimestamp !== Infinity ? minTimestamp : undefined);
  const axisMax = props.dateTo ? new Date(props.dateTo).getTime() : undefined;

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const ctx = chartCanvas.value.getContext('2d');
  chartInstance = new ChartJS(ctx, {
    type: 'bar',
    data: {
      datasets: datasets
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: textColor }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const raw = context.raw;
              const start = new Date(raw.x[0]).toLocaleString('es-ES', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
              const end = new Date(raw.x[1]).toLocaleString('es-ES', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
              const diffHrs = ((raw.x[1] - raw.x[0]) / 3600000).toFixed(2);
              return [
                `Equipo: ${context.dataset.label}`,
                `Tipo Falla: ${raw.type}`,
                `Modo Falla (mdf): ${raw.mode}`,
                `Duración: ${diffHrs} hrs`,
                `Inicio: ${start}`,
                `Fin: ${end}`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          min: axisMin,
          max: axisMax,
          time: {
            unit: 'month',
            tooltipFormat: 'dd MMM yyyy'
          },
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: {
            display: true,
            text: 'Downtime Timeline',
            color: textColor
          }
        },
        y: {
          type: 'category',
          ticks: { color: textColor },
          grid: { color: gridColor },
          title: {
            display: true,
            text: 'Equipment',
            color: textColor
          },
          offset: true,
        }
      }
    }
  });
};

const handleThemeChange = () => {
  createChart()
}

watch(() => [props.data, props.dateFrom, props.dateTo, props.minDuration], createChart, { deep: true });

onMounted(() => {
  createChart()
  window.addEventListener('theme-changed', handleThemeChange)
});

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>