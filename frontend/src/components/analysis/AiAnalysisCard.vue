<template>
  <div class="card">
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ $t('charts.nlp.title') }}</h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
          </button>
        </div>
        <p class="text-sm text-gray-550 dark:text-slate-400">{{ $t('charts.nlp.desc') }}</p>
      </div>

      <div class="flex flex-wrap items-center gap-3 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700">
        <!-- New Model Selector -->
        <div v-if="modelStatusList.length > 0" class="flex items-center gap-2 mr-2 border-r border-gray-300 dark:border-slate-600 pr-4">
          <label class="text-xs font-semibold text-gray-600 dark:text-slate-400">Model:</label>
          <select 
            v-model="selectedModel"
            class="w-full sm:w-80 px-2 py-1 bg-slate-800 border border-slate-600 rounded text-sm text-white focus:outline-none focus:border-indigo-500"
          >
            <option v-for="model in modelStatusList" :key="model.name" :value="model.name">
              {{ model.name }} {{ model.downloaded ? '✅' : '⏳' }} {{ model.downloaded ? (model.size_mb > 0 ? `(${model.size_mb}MB)` : '') : '(No descargado)' }}
            </option>
          </select>
        </div>

        <label class="flex items-center gap-2 cursor-pointer">
          <input 
            type="checkbox" 
            v-model="isEnabled" 
            class="rounded text-indigo-600 focus:ring-indigo-500" 
          />
          <span class="text-sm font-semibold text-gray-700 dark:text-slate-350">{{ $t('charts.nlp.enable') }}</span>
        </label>
        
        <button 
          @click="loadMiningData" 
          :disabled="!isEnabled || loadingState !== 'idle'"
          class="bg-indigo-600 text-white px-3 py-1 rounded text-sm hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <template v-if="loadingState !== 'idle'">
            {{ $t('charts.nlp.mining') }}
          </template>
          <template v-else-if="!isSelectedModelDownloaded">
            Descargar y Analizar
          </template>
          <template v-else>
            {{ $t('charts.nlp.analyze') }}
          </template>
        </button>
      </div>

      <!-- Error Banner -->
      <div v-if="errorMessage" class="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded shadow-sm">
        <div class="flex">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">Error durante la ejecución</h3>
            <div class="mt-2 text-sm text-red-700">
              <p>{{ errorMessage }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-show="!isCollapsed">
      <div v-if="!isEnabled" class="text-center py-8 bg-gray-50/50 dark:bg-slate-900/30 rounded-lg border border-dashed border-gray-200 dark:border-slate-800">
        <p class="text-sm text-gray-500 dark:text-slate-400 mb-2">{{ $t('charts.nlp.mining_disabled') }}</p>
        <button 
          @click="isEnabled = true; loadMiningData()" 
          class="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline"
        >
          {{ $t('charts.nlp.click_enable') }}
        </button>
      </div>

      <div v-else-if="!isSelectedModelDownloaded && loadingState === 'idle'" class="bg-amber-50 dark:bg-amber-900/30 border-l-4 border-amber-500 p-4 mb-6 rounded shadow-sm">
        <div class="flex">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-amber-500" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
          </div>
          <div class="ml-3">
            <p class="text-sm text-amber-800 dark:text-amber-200">
              <strong>Atención:</strong> El modelo de IA seleccionado no se encuentra en el disco duro. 
              Si presionas "Descargar y Analizar", el sistema descargará aproximadamente ~350MB. Esto puede tardar varios minutos dependiendo de tu conexión. 
              Una vez completado, el modelo quedará guardado localmente de forma permanente.
            </p>
          </div>
        </div>
      </div>

      <div v-if="loadingState === 'downloading'" class="flex flex-col justify-center items-center py-12 px-4">
        <svg class="w-12 h-12 text-indigo-500 animate-bounce mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        <span class="text-indigo-700 dark:text-indigo-400 font-semibold mb-2">Fase 1/2: Descargando Modelos...</span>
        <p class="text-gray-500 dark:text-slate-400 text-sm max-w-lg text-center leading-relaxed">
          Descargando la Inteligencia Artificial de Hugging Face al disco duro local. Este paso solo ocurre una vez. Por favor, <strong>no recargues la página</strong>.
        </p>
      </div>

      <div v-else-if="loadingState === 'analyzing' && !hasAnyResults" class="flex flex-col justify-center items-center py-12 px-4">
        <div class="w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4"></div>
        <span class="text-indigo-700 dark:text-indigo-400 font-semibold mb-2">Fase 2/2: Analizando Comentarios</span>
        <p class="text-gray-500 dark:text-slate-400 text-sm max-w-lg text-center leading-relaxed">
          Dependiendo de tu cantidad de registros (miles), el análisis profundo mediante IA puede tardar <strong>entre 5 a 20 minutos</strong> sin tarjeta gráfica. El sistema está trabajando, <strong>por favor no recargues la página</strong>.
        </p>      <div v-else-if="miningData && miningData.total_comments > 0" class="space-y-6">
        
        <!-- Background Processing Banner -->
        <div v-if="loadingState === 'analyzing' && hasAnyResults" class="bg-indigo-50 dark:bg-indigo-900/30 border-l-4 border-indigo-500 p-4 rounded shadow-sm mb-2">
          <div class="flex items-center">
            <svg class="animate-spin h-5 w-5 text-indigo-500 mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span class="text-indigo-800 dark:text-indigo-200 text-sm">
              Trabajando en segundo plano... Analizando modelo {{ analyzingIndex }} de {{ analyzingTotal }}: <strong>{{ analyzingCurrentModel }}</strong>
            </span>
          </div>
        </div>

        <!-- Pestañas (Tabs) Navigation -->
        <div class="border-b border-gray-200 dark:border-slate-700/80 mb-6">
          <nav class="-mb-px flex flex-wrap gap-4 sm:gap-6" aria-label="Tabs">
            <button
              v-for="model in actualModels"
              :key="model.name"
              @click="activeTab = model.name"
              class="whitespace-nowrap pb-3 px-1 border-b-2 font-semibold text-sm flex items-center gap-2 transition-all duration-200"
              :class="[
                activeTab === model.name
                  ? 'border-indigo-600 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200'
              ]"
            >
              <span>{{ model.name }}</span>
              <span 
                class="w-2 h-2 rounded-full transition-colors duration-200"
                :class="hasResultsForModel(model.name) ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50' : 'bg-gray-300 dark:bg-slate-700'"
              ></span>
            </button>

            <button
              @click="activeTab = 'Comparativa'"
              :disabled="!canShowComparison"
              class="whitespace-nowrap pb-3 px-1 border-b-2 font-semibold text-sm flex items-center gap-2 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
              :class="[
                activeTab === 'Comparativa'
                  ? 'border-indigo-600 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200'
              ]"
            >
              <span>Comparativa 📊</span>
              <span 
                v-if="canShowComparison"
                class="text-xs bg-indigo-100 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 px-2 py-0.5 rounded-full font-bold"
              >
                {{ analyzedModelsCount }}
              </span>
            </button>
          </nav>
        </div>

        <!-- TAB CONTENT: COMPARATIVA -->
        <div v-if="activeTab === 'Comparativa' && canShowComparison" class="space-y-6">
          <div class="bg-indigo-50/50 dark:bg-indigo-950/10 p-5 rounded-lg border border-indigo-100/20 shadow-sm leading-relaxed">
            <h4 class="font-bold text-indigo-900 dark:text-indigo-300 mb-1">Métricas de Comparación Cruzada</h4>
            <p class="text-xs text-gray-600 dark:text-slate-400">
              Esta sección compara los resultados obtenidos por los modelos analizados. La tasa de coincidencia indica el porcentaje de comentarios clasificados bajo la misma categoría semántica.
            </p>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Tasa de Coincidencia (Agreement Rate) -->
            <div class="bg-white dark:bg-slate-800 p-5 rounded-lg border border-gray-200 dark:border-slate-750 shadow-sm">
              <h4 class="font-bold text-gray-900 dark:text-white mb-4">Tasa de Coincidencia</h4>
              <div class="space-y-4">
                <div v-for="metric in comparisonMetrics" :key="`${metric.modelA}-${metric.modelB}`" class="p-3.5 bg-gray-50 dark:bg-slate-900/40 rounded-lg border border-gray-150/40 dark:border-slate-700/50 space-y-2">
                  <div class="flex justify-between items-center text-xs text-gray-500 dark:text-slate-400 font-semibold">
                    <span class="truncate max-w-[40%]">{{ metric.modelA }}</span>
                    <span class="text-gray-400">vs</span>
                    <span class="truncate max-w-[40%]">{{ metric.modelB }}</span>
                  </div>
                  <div class="flex justify-between items-end">
                    <span class="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{{ metric.agreementRate.toFixed(1) }}%</span>
                    <span class="text-xs text-gray-555 dark:text-slate-400 font-semibold">{{ metric.matches }} / {{ metric.total }} coincidencias</span>
                  </div>
                  <div class="w-full bg-gray-100 dark:bg-slate-900 rounded-full h-2">
                    <div 
                      class="bg-indigo-600 h-2 rounded-full transition-all duration-500" 
                      :style="{ width: `${metric.agreementRate}%` }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Tiempos de Procesamiento -->
            <div class="bg-white dark:bg-slate-800 p-5 rounded-lg border border-gray-200 dark:border-slate-750 shadow-sm flex flex-col">
              <h4 class="font-bold text-gray-900 dark:text-white mb-4">Tiempos de Procesamiento</h4>
              <div class="space-y-4 my-auto">
                <div v-for="modelName in analyzedModelNames" :key="modelName" class="space-y-1">
                  <div class="flex justify-between text-xs font-semibold text-gray-700 dark:text-slate-350">
                    <span class="truncate max-w-[70%]">{{ modelName }}</span>
                    <span class="font-bold text-amber-600 dark:text-amber-450">{{ getExecutionTimeForModel(modelName) }}s</span>
                  </div>
                  <div class="w-full bg-gray-150 dark:bg-slate-900 rounded-full h-2">
                    <div 
                      class="bg-amber-500 h-2 rounded-full transition-all duration-500" 
                      :style="{ width: `${(getExecutionTimeForModel(modelName) / maxExecutionTime) * 100}%` }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Distribución de Categorías Comparada -->
          <div class="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-750 shadow-sm overflow-hidden">
            <div class="p-5 border-b border-gray-200 dark:border-slate-700/80">
              <h4 class="font-bold text-gray-900 dark:text-white">Distribución de Categorías</h4>
            </div>
            <div class="overflow-x-auto">
              <table class="min-w-full divide-y divide-gray-200 dark:divide-slate-700">
                <thead class="bg-gray-50 dark:bg-slate-900/50">
                  <tr>
                    <th scope="col" class="px-5 py-3.5 text-left text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Categoría</th>
                    <th 
                      v-for="modelName in analyzedModelNames" 
                      :key="modelName" 
                      scope="col" 
                      class="px-5 py-3.5 text-left text-xs font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider truncate max-w-[200px]"
                    >
                      {{ modelName }}
                    </th>
                  </tr>
                </thead>
                <tbody class="bg-white dark:bg-slate-800 divide-y divide-gray-250 dark:divide-slate-700/50">
                  <tr v-for="category in categoryList" :key="category" class="hover:bg-gray-50 dark:hover:bg-slate-700/20 transition-colors">
                    <td class="px-5 py-3.5 text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                      <span class="w-2.5 h-2.5 rounded-full" :class="getCategoryColorClass(category)"></span>
                      {{ translateCategory(category) }}
                    </td>
                    <td 
                      v-for="modelName in analyzedModelNames" 
                      :key="modelName" 
                      class="px-5 py-3.5 text-sm text-gray-700 dark:text-slate-350"
                    >
                      <span class="font-bold text-gray-900 dark:text-white">{{ getCategoryCountForModel(modelName, category) }}</span>
                      <span class="text-xs text-gray-500 dark:text-slate-405 ml-1.5">
                        ({{ ((getCategoryCountForModel(modelName, category) / miningData.total_comments) * 100).toFixed(1) }}%)
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- TAB CONTENT: INDIVIDUAL MODEL RESULTS -->
        <div v-else-if="selectedModelData" class="space-y-6">
          <!-- Stats summary -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-indigo-50 dark:bg-indigo-950/20 p-4 rounded-lg border border-indigo-100/10">
              <p class="text-xs text-gray-600 dark:text-slate-400">{{ $t('charts.nlp.found') }}</p>
              <p class="text-2xl font-bold text-indigo-600 dark:text-indigo-450">{{ miningData.total_comments }}</p>
            </div>
            <div class="bg-emerald-50 dark:bg-emerald-950/20 p-4 rounded-lg border border-emerald-100/10">
              <p class="text-xs text-gray-600 dark:text-slate-400">{{ $t('charts.nlp.coverage') }}</p>
              <p class="text-2xl font-bold text-emerald-600 dark:text-emerald-450">{{ miningData.coverage.toFixed(1) }}%</p>
            </div>
            <div class="col-span-2 bg-amber-50 dark:bg-amber-950/20 p-4 rounded-lg border border-amber-100/10">
              <p class="text-xs text-gray-600 dark:text-slate-400">Tiempo de Ejecución ({{ activeTab }})</p>
              <p class="text-2xl font-bold text-amber-600 dark:text-amber-450">{{ selectedModelData.execution_time_seconds.toFixed(2) }}s</p>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Keyword extraction results -->
            <div class="bg-white dark:bg-slate-800 p-4 rounded-lg border border-gray-200 dark:border-slate-700 shadow-sm">
              <h3 class="font-bold text-gray-800 dark:text-white mb-4">{{ $t('charts.nlp.top_keywords') }}</h3>
              <div class="space-y-3">
                <div v-for="item in selectedModelData.keywords" :key="item.word" class="space-y-1">
                  <div class="flex justify-between text-xs font-semibold text-gray-700 dark:text-slate-350">
                    <span class="capitalize">{{ item.word }}</span>
                    <span>{{ item.count }} {{ $t('charts.nlp.occurrences') }}</span>
                  </div>
                  <div class="w-full bg-gray-100 dark:bg-slate-900 rounded-full h-2">
                    <div 
                      class="bg-indigo-600 h-2 rounded-full" 
                      :style="{ width: `${(item.count / maxKeywordCount) * 100}%` }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Semantic categories breakdown -->
            <div class="bg-white dark:bg-slate-800 p-4 rounded-lg border border-gray-200 dark:border-slate-700 shadow-sm">
              <h3 class="font-bold text-gray-800 dark:text-white mb-4">{{ $t('charts.nlp.categories') }}</h3>
              <div class="space-y-3">
                <div v-for="cat in selectedModelData.categories" :key="cat.category" class="p-3 bg-gray-50/50 dark:bg-slate-900/30 rounded border border-gray-150/40 dark:border-slate-700/50">
                  <div class="flex items-center justify-between mb-1.5">
                    <div class="flex items-center gap-2">
                      <div class="w-2.5 h-2.5 rounded-full" :class="getCategoryColorClass(cat.category)"></div>
                      <span class="text-sm font-semibold text-gray-850 dark:text-slate-200">{{ translateCategory(cat.category) }}</span>
                    </div>
                    <span class="text-xs font-bold text-gray-900 dark:text-white bg-gray-100 dark:bg-slate-900 px-2 py-0.5 rounded">
                      {{ cat.count }} {{ $t('charts.nlp.logs') }}
                    </span>
                  </div>
                  <div v-if="cat.count > 0" class="text-xs text-gray-500 dark:text-slate-405 space-y-1.5 pl-4.5 border-l border-gray-200 dark:border-slate-700 ml-1.5 mt-1.5">
                    <div>
                      <span class="text-gray-650 dark:text-slate-450 font-medium">{{ $t('charts.nlp.top_types') }}</span> 
                      <span class="ml-1 text-gray-800 dark:text-slate-300 font-semibold">{{ cat.top_types.filter(t => t && t !== 'nan' && t !== 'Unknown').join(', ') || 'N/A' }}</span>
                    </div>
                    <div>
                      <span class="text-gray-650 dark:text-slate-450 font-medium">{{ $t('charts.nlp.top_modes') }}</span> 
                      <span class="ml-1 text-gray-800 dark:text-slate-300 font-semibold">{{ cat.top_modes.filter(m => m && m !== 'nan' && m !== 'Unknown').join(', ') || 'N/A' }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- TAB CONTENT: MODEL NOT ANALYZED YET -->
        <div v-else class="text-center py-12 bg-gray-50/50 dark:bg-slate-900/20 rounded-lg border border-dashed border-gray-200 dark:border-slate-800">
          <p class="text-sm text-gray-500 dark:text-slate-400 mb-2">Este modelo aún no ha sido analizado.</p>
          <button 
            @click="selectedModel = activeTab; loadMiningData()" 
            class="text-xs font-bold text-indigo-600 dark:text-indigo-450 hover:underline"
          >
            Presiona aquí para descargar y analizar {{ activeTab }}
          </button>
        </div>
      </div>

      <div v-else class="text-center py-12 text-gray-500 dark:text-slate-400 text-sm">
        {{ $t('charts.nlp.no_comments') }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'

const { t } = useI18n()
const isCollapsed = ref(false)
const isEnabled = ref(false)
const loadingState = ref('idle')
const errorMessage = ref('')
const miningData = ref(null)
const selectedModel = ref('')
const modelStatusList = ref([])

const activeTab = ref('')
const analyzingCurrentModel = ref('')
const analyzingIndex = ref(0)
const analyzingTotal = ref(0)

const categoryList = [
  'Operational',
  'Cleaning/Blockage',
  'Mechanical',
  'Electrical',
  'Instrumentation/Failure',
  'Others'
]

onMounted(async () => {
  try {
    const res = await apiService.getModelsStatus()
    if (res.data && res.data.models) {
      modelStatusList.value = res.data.models
      // Agregar opción de Todos los modelos
      modelStatusList.value.unshift({
        name: 'Todos los modelos',
        downloaded: res.data.models.every(m => m.downloaded),
        size_mb: Math.round(res.data.models.reduce((sum, m) => sum + (m.downloaded ? m.size_mb : 0), 0))
      })

      if (modelStatusList.value.length > 0) {
        selectedModel.value = modelStatusList.value[0].name
      }
      
      const firstModel = modelStatusList.value.find(m => m.name !== 'Todos los modelos')
      if (firstModel) {
        activeTab.value = firstModel.name
      }
    }
  } catch (e) {
    console.error('Failed to load model status', e)
  }
})

const actualModels = computed(() => {
  return modelStatusList.value.filter(m => m.name !== 'Todos los modelos')
})

const isSelectedModelDownloaded = computed(() => {
  const model = modelStatusList.value.find(m => m.name === selectedModel.value)
  return model ? model.downloaded : false
})

const hasAnyResults = computed(() => {
  return miningData.value?.results && Object.keys(miningData.value.results).length > 0
})

const hasResultsForModel = (modelName) => {
  return !!(miningData.value?.results && miningData.value.results[modelName])
}

const analyzedModelsCount = computed(() => {
  if (!miningData.value?.results) return 0
  return Object.keys(miningData.value.results).length
})

const canShowComparison = computed(() => {
  return analyzedModelsCount.value >= 2
})

const analyzedModelNames = computed(() => {
  if (!miningData.value || !miningData.value.results) return []
  return Object.keys(miningData.value.results)
})

const selectedModelData = computed(() => {
  if (!miningData.value || !miningData.value.results || !activeTab.value) return null
  return miningData.value.results[activeTab.value]
})

const maxKeywordCount = computed(() => {
  const data = selectedModelData.value
  if (!data || !data.keywords || data.keywords.length === 0) return 1
  return Math.max(...data.keywords.map(k => k.count))
})

const getCategoryCountForModel = (modelName, category) => {
  const modelData = miningData.value?.results?.[modelName]
  if (!modelData || !modelData.categories) return 0
  const catObj = modelData.categories.find(c => c.category === category)
  return catObj ? catObj.count : 0
}

const getExecutionTimeForModel = (modelName) => {
  return miningData.value?.results?.[modelName]?.execution_time_seconds || 0
}

const maxExecutionTime = computed(() => {
  const times = analyzedModelNames.value.map(name => getExecutionTimeForModel(name))
  return Math.max(...times, 1)
})

const comparisonMetrics = computed(() => {
  if (!miningData.value || !miningData.value.results) return []
  
  const results = miningData.value.results
  const models = Object.keys(results).filter(m => results[m].predictions && results[m].predictions.length > 0)
  
  if (models.length < 2) return []
  
  const metrics = []
  for (let i = 0; i < models.length; i++) {
    for (let j = i + 1; j < models.length; j++) {
      const m1 = models[i]
      const m2 = models[j]
      
      const preds1 = results[m1].predictions
      const preds2 = results[m2].predictions
      
      const minLen = Math.min(preds1.length, preds2.length)
      if (minLen === 0) continue
      
      let matches = 0
      for (let k = 0; k < minLen; k++) {
        if (preds1[k] === preds2[k]) {
          matches++
        }
      }
      
      const agreementRate = (matches / minLen) * 100
      metrics.push({
        modelA: m1,
        modelB: m2,
        agreementRate: agreementRate,
        matches: matches,
        total: minLen
      })
    }
  }
  return metrics
})

const getCategoryColorClass = (category) => {
  switch (category) {
    case 'Operational': return 'bg-blue-500'
    case 'Cleaning/Blockage': return 'bg-amber-500'
    case 'Mechanical': return 'bg-red-500'
    case 'Electrical': return 'bg-purple-500'
    case 'Instrumentation/Failure': return 'bg-emerald-500'
    default: return 'bg-gray-500'
  }
}

const translateCategory = (cat) => {
  const map = {
    'Operational': 'charts.nlp.cat_op',
    'Cleaning/Blockage': 'charts.nlp.cat_clean',
    'Mechanical': 'charts.nlp.cat_mech',
    'Electrical': 'charts.nlp.cat_elec',
    'Instrumentation/Failure': 'charts.nlp.cat_inst'
  }
  return map[cat] ? t(map[cat]) : cat
}

const loadMiningData = async () => {
  if (!isEnabled.value) return
  
  errorMessage.value = ''
  
  try {
    if (!isSelectedModelDownloaded.value) {
      loadingState.value = 'downloading'
      await apiService.downloadModels(selectedModel.value)
    }

    loadingState.value = 'analyzing'
    
    // Preparar el contenedor de datos limpio
    miningData.value = { results: {}, coverage: 0, total_comments: 0 }
    
    let modelsToRun = []
    if (selectedModel.value === 'Todos los modelos') {
      modelsToRun = modelStatusList.value.filter(m => m.name !== 'Todos los modelos').map(m => m.name)
    } else {
      modelsToRun = [selectedModel.value]
    }
    
    analyzingTotal.value = modelsToRun.length
    
    for (let i = 0; i < modelsToRun.length; i++) {
      const modelName = modelsToRun[i]
      analyzingCurrentModel.value = modelName
      analyzingIndex.value = i + 1
      
      const res = await apiService.getCommentMining(null, null, modelName)
      
      if (res.data) {
        // Fusionar resultados incrementalmente
        if (res.data.results && res.data.results[modelName]) {
          miningData.value.results[modelName] = res.data.results[modelName]
        }
        miningData.value.coverage = res.data.coverage || miningData.value.coverage
        miningData.value.total_comments = res.data.total_comments || miningData.value.total_comments
        
        // Enfocar la pestaña en el primer modelo que termine de procesar
        if (i === 0) {
          activeTab.value = modelName
        }
      }
    }
  } catch (err) {
    console.error('Error running text analysis:', err)
    errorMessage.value = err.response?.data?.detail || err.message || 'Ocurrió un error desconocido al contactar al servidor.'
  } finally {
    // Refresh model status ALWAYs, because we might have just downloaded models
    // even if the subsequent analysis timed out or failed
    try {
      const statusRes = await apiService.getModelsStatus()
      if (statusRes.data && statusRes.data.models) {
        modelStatusList.value = statusRes.data.models
        modelStatusList.value.unshift({
          name: 'Todos los modelos',
          downloaded: statusRes.data.models.every(m => m.downloaded),
          size_mb: Math.round(statusRes.data.models.reduce((sum, m) => sum + (m.downloaded ? m.size_mb : 0), 0))
        })
      }
    } catch (e) {
      console.warn('Failed to refresh model status after download', e)
    }
    loadingState.value = 'idle'
  }
}
</script>
