<template>
  <div class="flex items-center justify-center h-full">
    <div class="card max-w-md w-full">
      <div class="text-center mb-6">
        <div class="w-16 h-16 bg-blue-100 dark:bg-slate-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white">{{ $t('upload.title') }}</h2>
        <p class="text-gray-600 dark:text-slate-400 mt-2">{{ $t('upload.desc') }}</p>
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
          {{ $t('upload.button') }}
        </button>
      </div>

      <div v-if="pendingFile" class="mt-4 p-3 bg-gray-50 dark:bg-slate-900/50 rounded-lg">
        <p class="text-sm text-gray-700 dark:text-slate-300">
          {{ pendingFile.name }} &mdash; {{ formatFileSize(pendingFile.size) }}
        </p>
        <div class="mt-3 flex items-center justify-center gap-3">
          <button
            @click="confirmUpload"
            :disabled="isLoading"
            class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition-colors disabled:opacity-50"
          >
            {{ $t('upload.confirm') }}
          </button>
          <button
            @click="cancelSelection"
            :disabled="isLoading"
            class="px-4 py-1.5 text-sm font-medium rounded-lg bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-200 hover:bg-gray-300 dark:hover:bg-slate-600 transition-colors disabled:opacity-50"
          >
            {{ $t('upload.cancel') }}
          </button>
        </div>
        <div v-if="isLoading" class="mt-2 text-blue-600 dark:text-blue-400 text-sm font-medium text-center flex items-center justify-center gap-2">
          {{ $t('upload.processing') }}
        </div>
      </div>

      <div v-else-if="selectedFile" class="mt-4 p-3 bg-gray-50 dark:bg-slate-900/50 rounded-lg">
        <p class="text-sm text-gray-700 dark:text-slate-300">
          <strong>{{ $t('upload.file') }}</strong> {{ selectedFile.name }}
        </p>
        <p v-if="qualityMessage" class="mt-1 text-xs text-amber-700 dark:text-amber-400">
          {{ qualityMessage }}
        </p>
      </div>

      <div v-if="error" class="mt-4 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 rounded-lg">
        <p class="text-red-700 dark:text-red-400 text-sm">{{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../api'

const { t } = useI18n()
const emit = defineEmits(['file-uploaded'])

const selectedFile = ref(null)
const pendingFile = ref(null)
const isDragging = ref(false)
const isLoading = ref(false)
const error = ref('')
const qualityMessage = ref('')
const fileInput = ref(null)

const isCsvFile = (file) => {
  if (!file) return false
  const name = file.name ? file.name.toLowerCase() : ''
  const type = file.type ? file.type.toLowerCase() : ''
  return name.endsWith('.csv') || type === 'text/csv' || type.includes('csv') || type === 'application/vnd.ms-excel' || type === ''
}

// First line of defensive feedback, before even offering the confirm step.
const hasCsvExtension = (file) => {
  if (!file || !file.name) return false
  return file.name.toLowerCase().endsWith('.csv')
}

const formatFileSize = (bytes) => {
  if (!bytes && bytes !== 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

const selectCandidateFile = (file) => {
  if (!file) return
  if (!hasCsvExtension(file)) {
    error.value = t('upload.err_invalid_extension')
    pendingFile.value = null
    return
  }
  error.value = ''
  qualityMessage.value = ''
  pendingFile.value = file
}

const handleFileSelect = (event) => {
  selectCandidateFile(event.target.files[0])
}

const handleDrop = (event) => {
  event.preventDefault()
  isDragging.value = false
  selectCandidateFile(event.dataTransfer.files[0])
}

const cancelSelection = () => {
  pendingFile.value = null
  error.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const buildQualityMessage = (data) => {
  const invalidDates = data?.invalid_dates_count || 0
  const duplicates = data?.duplicates_removed_count || 0
  if (invalidDates > 0 && duplicates > 0) {
    return t('upload.quality_report', { invalidDates, duplicates })
  } else if (invalidDates > 0) {
    return t('upload.quality_report_dates_only', { invalidDates })
  } else if (duplicates > 0) {
    return t('upload.quality_report_duplicates_only', { duplicates })
  }
  return ''
}

const confirmUpload = async () => {
  const file = pendingFile.value
  if (!file) return

  // Keep the existing permissive MIME/extension acceptance for the actual upload call.
  if (!isCsvFile(file)) {
    error.value = t('upload.err_invalid')
    return
  }

  isLoading.value = true
  error.value = ''
  qualityMessage.value = ''

  try {
    const response = await apiService.upload(file)
    selectedFile.value = file
    pendingFile.value = null
    qualityMessage.value = buildQualityMessage(response.data)
    emit('file-uploaded')
  } catch (err) {
    error.value = err.response?.data?.detail || t('upload.err_failed')
    console.error('Upload error:', err)
  } finally {
    isLoading.value = false
  }
}
</script>
