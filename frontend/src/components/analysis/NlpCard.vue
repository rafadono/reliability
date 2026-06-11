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
          :disabled="!isEnabled || loading"
          class="bg-indigo-600 text-white px-3 py-1 rounded text-sm hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ loading ? $t('charts.nlp.mining') : $t('charts.nlp.analyze') }}
        </button>
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

      <div v-else-if="loading" class="flex justify-center items-center py-12">
        <span class="text-gray-500 dark:text-slate-400 animate-pulse text-sm">{{ $t('charts.nlp.mining') }}</span>
      </div>

      <div v-else-if="miningData && miningData.total_comments > 0" class="space-y-6">
        <!-- Stats summary -->
        <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div class="bg-indigo-50 dark:bg-indigo-950/20 p-4 rounded-lg border border-indigo-100/10">
            <p class="text-xs text-gray-600 dark:text-slate-400">{{ $t('charts.nlp.found') }}</p>
            <p class="text-2xl font-bold text-indigo-600 dark:text-indigo-450">{{ miningData.total_comments }}</p>
          </div>
          <div class="bg-emerald-50 dark:bg-emerald-950/20 p-4 rounded-lg border border-emerald-100/10">
            <p class="text-xs text-gray-600 dark:text-slate-400">{{ $t('charts.nlp.coverage') }}</p>
            <p class="text-2xl font-bold text-emerald-600 dark:text-emerald-450">{{ miningData.coverage.toFixed(1) }}%</p>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Keyword extraction results -->
          <div class="bg-white dark:bg-slate-800 p-4 rounded-lg border border-gray-200 dark:border-slate-700 shadow-sm">
            <h3 class="font-bold text-gray-800 dark:text-white mb-4">{{ $t('charts.nlp.top_keywords') }}</h3>
            <div class="space-y-3">
              <div v-for="item in miningData.keywords" :key="item.word" class="space-y-1">
                <div class="flex justify-between text-xs font-semibold text-gray-700 dark:text-slate-300">
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
              <div v-for="cat in miningData.categories" :key="cat.category" class="p-3 bg-gray-50/50 dark:bg-slate-900/30 rounded border border-gray-150/40 dark:border-slate-700/50">
                <div class="flex items-center justify-between mb-1.5">
                  <div class="flex items-center gap-2">
                    <div class="w-2.5 h-2.5 rounded-full" :class="getCategoryColorClass(cat.category)"></div>
                    <span class="text-sm font-semibold text-gray-805 dark:text-slate-200">{{ translateCategory(cat.category) }}</span>
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

      <div v-else class="text-center py-12 text-gray-500 dark:text-slate-400 text-sm">
        {{ $t('charts.nlp.no_comments') }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'

const { t } = useI18n()
const isCollapsed = ref(false)
const isEnabled = ref(false)
const loading = ref(false)
const miningData = ref(null)
const selectedModel = ref('')
const availableModels = ref([])

const selectedModelData = computed(() => {
  if (!miningData.value || !miningData.value.models) return null
  return miningData.value.models[selectedModel.value]
})

const maxKeywordCount = computed(() => {
  if (!miningData.value?.keywords || miningData.value.keywords.length === 0) return 1
  return Math.max(...miningData.value.keywords.map(k => k.count))
})

const getCategoryColorClass = (category) => {
  switch (category) {
    case 'Operational':
      return 'bg-blue-500'
    case 'Cleaning/Blockage':
      return 'bg-amber-500'
    case 'Mechanical':
      return 'bg-red-500'
    case 'Electrical':
      return 'bg-purple-500'
    case 'Instrumentation/Failure':
      return 'bg-emerald-500'
    default:
      return 'bg-gray-500'
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
  loading.value = true
  try {
    const res = await apiService.getCommentMining()
    miningData.value = res.data
  } catch (err) {
    console.error('Error running text analysis:', err)
  } finally {
    loading.value = false
  }
}
</script>
