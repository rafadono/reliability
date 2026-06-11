<template>
  <aside class="w-64 bg-white dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 p-6 overflow-y-auto transition-colors duration-300">
    <div class="space-y-6">
      <div class="bg-blue-50 dark:bg-slate-900/50 p-4 rounded-lg">
        <h3 class="font-semibold text-gray-900 dark:text-white mb-2">Quick Actions</h3>
        <button
          @click="triggerUpload"
          :disabled="isLoading"
          class="w-full btn-primary text-sm mb-2"
          :class="isLoading ? 'opacity-50' : ''"
        >
          Upload New Data
        </button>
        <button
          @click="$emit('reset')"
          :disabled="isLoading"
          class="w-full btn-secondary text-sm"
          :class="isLoading ? 'opacity-50' : ''"
        >
          Reset Filters
        </button>
        <input
          type="file"
          ref="fileInput"
          accept=".csv"
          @change="handleFileChange"
          class="hidden"
        />
      </div>

      <div class="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-4 rounded-lg shadow-md">
        <h3 class="font-semibold mb-3">Analysis Types</h3>
        <ul class="space-y-1 text-sm font-medium">
          <li>
            <button 
              @click="scrollToCard('pareto-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Pareto Analysis
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('jackknife-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Jackknife Diagram
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('criticality-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Criticality Matrix
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('weibull-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Weibull Fitting
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('proactive-pm-section')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Proactive PM
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('event-plot-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Event Timeline
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('apm-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Bad Actors APM
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('trend-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              Historical KPIs
            </button>
          </li>
          <li>
            <button 
              @click="scrollToCard('nlp-card')" 
              class="w-full text-left flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/15 transition-colors focus:outline-none focus:ring-1 focus:ring-white/30"
            >
              NLP Comment Mining
            </button>
          </li>
        </ul>
      </div>

      <div>
        <h3 class="font-semibold text-gray-900 dark:text-white mb-3">Navigation</h3>
        <nav class="space-y-2">
          <a href="#" class="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 font-medium">
            Dashboard
          </a>
          <a href="#" @click.prevent="$emit('export-pdf')" class="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            Download PDF
          </a>
          <a href="#" @click.prevent="$emit('notify', 'Settings module coming soon')" class="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
            Settings
          </a>
        </nav>
      </div>

      <div class="pt-4 border-t border-gray-200 dark:border-slate-700">
        <p class="text-xs text-gray-600 dark:text-slate-400">
          <strong>Version:</strong> 1.0<br>
          <strong>Status:</strong> Ready
        </p>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  isLoading: Boolean
})

const emit = defineEmits(['upload-file', 'reset', 'notify', 'export-pdf'])

const fileInput = ref(null)

const triggerUpload = () => {
  fileInput.value.click()
}

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    emit('upload-file', file)
  }
}

const scrollToCard = (id) => {
  const el = document.getElementById(id)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  } else if (id === 'proactive-pm-section') {
    const wEl = document.getElementById('weibull-card')
    if (wEl) {
      wEl.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }
}
</script>
