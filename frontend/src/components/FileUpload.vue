<template>
  <div class="flex items-center justify-center h-full">
    <div class="card max-w-md w-full">
      <div class="text-center mb-6">
        <div class="w-16 h-16 bg-blue-100 dark:bg-slate-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white">Upload Data</h2>
        <p class="text-gray-600 dark:text-slate-400 mt-2">Drag CSV file or click to select</p>
      </div>

      <div 
        @drop="handleDrop"
        @dragover.prevent="isDragging = true"
        @dragleave="isDragging = false"
        class="border-2 border-dashed border-gray-300 dark:border-slate-700 rounded-lg p-8 text-center cursor-pointer transition-colors"
        :class="isDragging ? 'border-blue-500 bg-blue-50 dark:bg-slate-900/30' : ''"
      >
        <input
          type="file"
          accept=".csv"
          @change="handleFileSelect"
          class="hidden"
          ref="fileInput"
          :disabled="isLoading"
        />
        <button
          @click="$refs.fileInput.click()"
          class="text-blue-600 dark:text-blue-400 hover:text-blue-700 font-medium"
          :disabled="isLoading"
        >
          Select File
        </button>
      </div>

      <div v-if="selectedFile" class="mt-4 p-3 bg-gray-50 dark:bg-slate-900/50 rounded-lg">
        <p class="text-sm text-gray-700 dark:text-slate-300">
          <strong>File:</strong> {{ selectedFile.name }}
        </p>
        <div v-if="isLoading" class="mt-2 text-blue-600 dark:text-blue-400 text-sm font-medium text-center flex items-center justify-center gap-2">
          <span class="inline-block animate-pulse">⏳</span> Uploading...
        </div>
      </div>

      <div v-if="error" class="mt-4 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 rounded-lg">
        <p class="text-red-700 dark:text-red-400 text-sm">{{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { apiService } from '../api'

const emit = defineEmits(['file-uploaded'])

const selectedFile = ref(null)
const isDragging = ref(false)
const isLoading = ref(false)
const error = ref('')
const fileInput = ref(null)

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file && file.type === 'text/csv') {
    selectedFile.value = file
    error.value = ''
    uploadFile()
  } else {
    error.value = 'Please select a valid CSV file'
  }
}

const handleDrop = (event) => {
  event.preventDefault()
  isDragging.value = false
  
  const file = event.dataTransfer.files[0]
  if (file && file.type === 'text/csv') {
    selectedFile.value = file
    error.value = ''
    uploadFile()
  } else {
    error.value = 'Please drop a valid CSV file'
  }
}

const uploadFile = async () => {
  if (!selectedFile.value) return

  isLoading.value = true
  error.value = ''

  try {
    await apiService.upload(selectedFile.value)
    emit('file-uploaded')
  } catch (err) {
    error.value = err.response?.data?.detail || 'Upload failed. Please try again.'
    console.error('Upload error:', err)
  } finally {
    isLoading.value = false
  }
}
</script>
