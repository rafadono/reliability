<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-slate-900 dark:to-slate-950 transition-colors duration-300">
    <nav class="bg-white dark:bg-slate-800 border-b dark:border-slate-700 shadow-sm transition-colors duration-300">
      <div class="w-full flex items-center justify-between">
        <div class="flex items-center">
          <button v-if="dataLoaded" @click="isSidebarOpen = !isSidebarOpen" class="p-2 ml-4 mr-2 text-gray-500 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-md transition-colors" title="Toggle Sidebar">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <div class="py-4 flex items-center gap-3 shrink-0" :class="{'pl-2': dataLoaded, 'pl-6': !dataLoaded}">
            <div class="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center shrink-0">
              <span class="text-white font-bold text-lg">RA</span>
            </div>
            <h1 class="text-xl font-bold text-gray-900 dark:text-white whitespace-nowrap">{{ $t('navbar.title') }}</h1>
          </div>
        </div>
        <div class="flex items-center gap-4 text-sm px-6 py-4">
          <button 
            @click="toggleLanguage" 
            class="p-2 text-gray-500 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors text-xs font-bold uppercase tracking-wider"
            :title="locale === 'es' ? 'Switch to English' : 'Cambiar a Español'"
          >
            {{ locale === 'es' ? 'ES' : 'EN' }}
          </button>
          <button 
            @click="toggleDarkMode" 
            class="p-2 text-gray-500 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors text-xs font-semibold uppercase tracking-wider"
            :title="isDarkMode ? 'Light Mode' : 'Dark Mode'"
          >
            {{ isDarkMode ? 'Light' : 'Dark' }}
          </button>
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium flex items-center gap-1 transition-colors">
            API Swagger
          </a>
          <div class="text-gray-600 dark:text-slate-400 border-l border-gray-300 dark:border-slate-600 pl-4">v1.0</div>
        </div>
      </div>
    </nav>

    <div class="flex h-[calc(100vh-80px)]">
      <Sidebar 
        v-if="dataLoaded"
        v-show="isSidebarOpen"
        @upload-file="handleSidebarUpload"
        @notify="showNotification"
        @reset="handleReset"
        @export-pdf="handleExportPDF"
        :isLoading="isLoading"
      />
      <main class="flex-1 overflow-auto">
        <FileUpload 
          v-if="!dataLoaded"
          @file-uploaded="handleFileUploaded"
          :isLoading="isLoading"
        />
        <div id="dashboard-content" class="min-h-full" v-else>
          <Dashboard 
            :key="dashboardKey"
            :isLoading="isLoading"
            @filters-changed="handleFiltersChanged"
          />
        </div>
      </main>
    </div>

    <div v-if="notification" class="fixed bottom-4 right-4 p-4 bg-green-500 text-white rounded-lg shadow-lg">
      {{ notification }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import Dashboard from './components/Dashboard.vue'
import FileUpload from './components/FileUpload.vue'
import Sidebar from './components/Sidebar.vue'
import { apiService } from './api'
import html2pdf from 'html2pdf.js'

const { locale } = useI18n()
const dataLoaded = ref(false)
const isLoading = ref(false)
const notification = ref('')
const dashboardKey = ref(0)
const isSidebarOpen = ref(true)
const isDarkMode = ref(false)

const toggleLanguage = () => {
  locale.value = locale.value === 'es' ? 'en' : 'es'
  localStorage.setItem('app-lang', locale.value)
}

const toggleDarkMode = () => {
  isDarkMode.value = !isDarkMode.value
  if (isDarkMode.value) {
    document.documentElement.classList.add('dark')
    localStorage.setItem('theme', 'dark')
  } else {
    document.documentElement.classList.remove('dark')
    localStorage.setItem('theme', 'light')
  }
  // Dispatch a custom event to notify chart components to update their styles
  window.dispatchEvent(new Event('theme-changed'))
}

const handleFileUploaded = () => {
  dataLoaded.value = true
  showNotification('File uploaded successfully')
}

const handleFiltersChanged = () => {
  // Handle filter changes from Dashboard
}

const handleSidebarUpload = async (file) => {
  isLoading.value = true
  try {
    const response = await apiService.upload(file)
    if (response.data.status === 'success') {
      showNotification('New file uploaded successfully')
      dashboardKey.value += 1
    }
  } catch (error) {
    console.error('Error uploading file:', error)
    showNotification('Error uploading file')
  } finally {
    isLoading.value = false
  }
}

const handleReset = async () => {
  isLoading.value = true
  try {
    await apiService.resetFilters()
    showNotification('Filters reset')
    dashboardKey.value += 1
  } catch (error) {
    console.error('Error resetting filters:', error)
  } finally {
    isLoading.value = false
  }
}

const handleExportPDF = () => {
  showNotification('Generating PDF, please wait...')
  const element = document.getElementById('dashboard-content')
  
  const opt = {
    margin:       [0.3, 0.3],
    filename:     'Reliability_Report.pdf',
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, logging: false },
    jsPDF:        { unit: 'in', format: 'a3', orientation: 'landscape' },
    pagebreak:    { mode: ['css', 'legacy'], avoid: '.card' }
  }
  
  html2pdf().set(opt).from(element).save().then(() => {
    showNotification('PDF downloaded successfully')
  }).catch(err => {
    console.error('Error exporting PDF:', err)
    showNotification('Error generating PDF')
  })
}

const showNotification = (message) => {
  notification.value = message
  setTimeout(() => {
    notification.value = ''
  }, 3000)
}

onMounted(async () => {
  // Initialize dark mode theme
  const savedTheme = localStorage.getItem('theme')
  if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    isDarkMode.value = true
    document.documentElement.classList.add('dark')
  } else {
    isDarkMode.value = false
    document.documentElement.classList.remove('dark')
  }

  // Check if data is already loaded
  try {
    await apiService.getSummaryStats()
    dataLoaded.value = true
  } catch {
    dataLoaded.value = false
  }
})
</script>