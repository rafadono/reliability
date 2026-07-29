// Small shared helper to export a Chart.js chart instance as a downloadable PNG image.
// Reused by the dense/multi-point charts (Pareto, Criticality, Jackknife, Event Plot)
// so the download logic isn't duplicated in each component.
export function downloadChartImage(chartInstance, filename = 'chart.png') {
  if (!chartInstance) return
  const url = chartInstance.toBase64Image('image/png', 1)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
