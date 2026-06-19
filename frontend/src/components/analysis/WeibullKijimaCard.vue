<template>
  <div class="card">
    <!-- Header -->
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
      <div class="flex flex-col md:flex-row items-start md:items-center gap-4">
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-gray-900 dark:text-white">
            {{ activeTab === 'TBX' ? $t('charts.kijima.title') : $t('charts.weibull.title') }}
          </h2>
          <button 
            @click="isCollapsed = !isCollapsed"
            class="text-xs font-semibold px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-gray-600 dark:text-slate-300 transition-colors"
          >
            {{ isCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
          </button>
        </div>
        <!-- Tabs -->
        <div class="flex bg-gray-100 dark:bg-slate-900 p-1 rounded-lg">
          <button 
            @click="activeTab = 'TBX'"
            :class="activeTab === 'TBX' ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white'"
            class="px-4 py-1.5 text-sm font-medium rounded-md transition-colors"
          >
            {{ $t('charts.weibull.reliability') }} (TBX)
          </button>
          <button 
            @click="activeTab = 'TTX'"
            :class="activeTab === 'TTX' ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white'"
            class="px-4 py-1.5 text-sm font-medium rounded-md transition-colors"
          >
            {{ $t('charts.weibull.maintainability') }} (TTX)
          </button>
        </div>
      </div>
      <!-- Equipment, Min TBX and Refit -->
      <div class="flex flex-wrap gap-3 bg-gray-50 dark:bg-slate-900/50 p-2 rounded-lg border border-gray-200 dark:border-slate-700 items-center">
        <div class="flex items-center gap-1.5">
          <label class="text-xs font-semibold text-gray-600 dark:text-slate-300">{{ $t('charts.kpi.asset_label') || 'Activo:' }}</label>
          <select v-model="localFilters.equipment" class="text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500 py-1">
            <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
          </select>
        </div>
        <div class="flex items-center gap-1.5">
          <label class="text-xs font-semibold text-gray-600 dark:text-slate-300">{{ $t('charts.kijima.min_tbx_label') }}:</label>
          <input type="number" v-model.number="minTbx" min="0" step="0.1" class="w-16 text-sm border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded focus:ring-blue-500 py-1 px-2" />
        </div>
        <button @click="loadAnalysis" :disabled="loading" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 disabled:opacity-50 font-semibold">
          {{ loading ? $t('sidebar.loading') : $t('charts.weibull.refit') }}
        </button>
      </div>
    </div>

    <!-- Main Content -->
    <div v-show="!isCollapsed">
      <!-- Advanced filters for Types to Fit and Censored Types -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">{{ $t('charts.weibull.types_to_fit') }}</label>
          <div class="space-y-2 max-h-40 overflow-y-auto bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700">
            <label v-for="t in availableTypes" :key="'fit-'+t" class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-700/50 p-1 rounded">
              <input type="checkbox" :value="t" v-model="typesToFit" :disabled="censoredTypes.includes(t)" class="rounded text-blue-600 focus:ring-blue-500 disabled:opacity-50" />
              <span class="text-sm text-gray-700 dark:text-slate-300" :class="{ 'opacity-50': censoredTypes.includes(t) }">{{ t }}</span>
            </label>
          </div>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">{{ $t('charts.weibull.censored_types') }}</label>
          <div class="space-y-2 max-h-40 overflow-y-auto bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700">
            <label v-for="t in availableTypes" :key="'cen-'+t" class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-700/50 p-1 rounded">
              <input type="checkbox" :value="t" v-model="censoredTypes" :disabled="typesToFit.includes(t)" class="rounded text-orange-600 focus:ring-orange-500 disabled:opacity-50" />
              <span class="text-sm text-gray-700 dark:text-slate-300" :class="{ 'opacity-50': typesToFit.includes(t) }">{{ t }}</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Curve selections (TBX Mode Only) -->
      <div v-if="activeTab === 'TBX'" class="mb-6 p-4 bg-white dark:bg-slate-800/40 rounded-lg border border-gray-200 dark:border-slate-700">
        <label class="block text-sm font-bold text-gray-800 dark:text-slate-200 mb-2">{{ $t('charts.kijima.models_to_plot') }}</label>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <label v-for="opt in modelOptions" :key="opt.id"
            class="flex items-center gap-2 p-2 rounded-md cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors"
          >
            <input type="checkbox" v-model="selectedCurves" :value="opt.id" class="rounded focus:ring-blue-500" :style="{ color: opt.color }" />
            <span class="text-sm font-semibold" :style="{ color: opt.color }">{{ opt.label }}</span>
          </label>
        </div>
      </div>

      <!-- Parameter Quick Details -->
      <div v-if="hasFitData" class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <template v-if="activeTab === 'TBX'">
          <div v-for="mc in activeFitSummary" :key="mc.name" class="p-4 rounded-lg border shadow-sm" :class="mc.bgClass">
            <p class="text-xs font-bold uppercase tracking-wider mb-1" :style="{ color: mc.color }">{{ mc.name }}</p>
            <div class="space-y-1 text-sm text-gray-800 dark:text-slate-200">
              <div><span class="text-xs opacity-75">β:</span> <strong>{{ mc.beta }}</strong></div>
              <div><span class="text-xs opacity-75">η:</span> <strong>{{ mc.eta }}</strong></div>
              <div v-if="mc.ar !== undefined && mc.name !== $t('charts.kijima.weibull')">
                <span class="text-xs opacity-75">ar / ap:</span> <strong>{{ mc.ar }} / {{ mc.ap }}</strong>
              </div>
              <div v-if="mc.br !== undefined && mc.name !== $t('charts.kijima.weibull') && (mc.br !== '0.0000' || mc.bp !== '0.0000')">
                <span class="text-xs opacity-75">br / bp:</span> <strong>{{ mc.br }} / {{ mc.bp }}</strong>
              </div>
            </div>
          </div>
        </template>
        <template v-else>
          <!-- TTX Mode (Weibull Only) -->
          <div class="p-4 rounded-lg border shadow-sm bg-indigo-50/30 border-indigo-100 dark:bg-indigo-950/10 dark:border-indigo-900/30 col-span-2">
            <p class="text-xs font-bold uppercase tracking-wider mb-1 text-indigo-600 dark:text-indigo-400">
              {{ $t('charts.kijima.weibull') }} (TTX)
            </p>
            <div class="grid grid-cols-2 gap-4 text-sm text-gray-800 dark:text-slate-200 mt-2">
              <div><span class="text-xs opacity-75">{{ $t('charts.weibull.beta') }}:</span> <strong class="text-lg block text-indigo-600 dark:text-indigo-400">{{ weibullResult?.parameters?.beta?.toFixed(3) || '-' }}</strong></div>
              <div><span class="text-xs opacity-75">{{ $t('charts.weibull.eta') }}:</span> <strong class="text-lg block text-green-600 dark:text-green-400">{{ weibullResult?.parameters?.eta?.toFixed(1) || '-' }}</strong></div>
            </div>
          </div>
        </template>
      </div>

      <!-- Charts Grid -->
      <div v-if="hasFitData" class="space-y-6">
        <div class="grid grid-cols-1 gap-6" :class="{ 'md:grid-cols-2': !expandedChart }">
          <!-- Chart 1: R(t) or F(t) -->
          <div 
            v-show="!expandedChart || expandedChart === 'rel'"
            class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm"
            :class="{ 'col-span-2': expandedChart === 'rel' }"
          >
            <div class="flex justify-between items-center mb-2">
              <h3 class="font-bold text-gray-800 dark:text-white">
                {{ activeTab === 'TBX' ? $t('charts.kijima.legends_display') + ' R(t)' : $t('charts.weibull.cum_prob') }}
              </h3>
              <button 
                @click="toggleExpand('rel')" 
                class="p-1 rounded text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-700 transition-colors"
              >
                <svg v-if="expandedChart === 'rel'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4h4v4m12-4h-4v4M4 20h4v-4m12 4h-4v-4" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4" />
                </svg>
              </button>
            </div>
            <div class="relative" :class="expandedChart === 'rel' ? 'h-[500px]' : 'h-80'">
              <canvas ref="relChartRef"></canvas>
            </div>
          </div>

          <!-- Chart 2: h(t) or f(t) -->
          <div 
            v-show="!expandedChart || expandedChart === 'hazard'"
            class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm"
            :class="{ 'col-span-2': expandedChart === 'hazard' }"
          >
            <div class="flex justify-between items-center mb-2">
              <h3 class="font-bold text-gray-800 dark:text-white">
                {{ activeTab === 'TBX' ? $t('charts.kijima.legends_display') + ' h(t)' : $t('charts.weibull.prob_density') }}
              </h3>
              <button 
                @click="toggleExpand('hazard')" 
                class="p-1 rounded text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-700 transition-colors"
              >
                <svg v-if="expandedChart === 'hazard'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4h4v4m12-4h-4v4M4 20h4v-4m12 4h-4v-4" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4" />
                </svg>
              </button>
            </div>
            <div class="relative" :class="expandedChart === 'hazard' ? 'h-[500px]' : 'h-80'">
              <canvas ref="hazardChartRef"></canvas>
            </div>
          </div>
        </div>

        <!-- Chart 3: Virtual Age V(t) (TBX Mode and Collapsible) -->
        <div 
          v-if="activeTab === 'TBX'"
          class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm"
        >
          <div class="flex justify-between items-center mb-2">
            <div class="flex items-center gap-2">
              <h3 class="font-bold text-gray-800 dark:text-white">{{ $t('charts.kijima.virtual_age') }} V(t)</h3>
              <button 
                @click="isVirtualAgeCollapsed = !isVirtualAgeCollapsed"
                class="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-gray-600 dark:text-slate-300 transition-colors"
              >
                {{ isVirtualAgeCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
              </button>
            </div>
            <div class="flex items-center gap-4">
              <label v-show="!isVirtualAgeCollapsed" class="flex items-center gap-1.5 text-xs font-semibold text-gray-600 dark:text-slate-300 cursor-pointer">
                <input type="checkbox" v-model="showNoRepairBaseline" class="rounded text-blue-600 focus:ring-blue-500 w-3.5 h-3.5" />
                <span>{{ $t('charts.kijima.show_baseline') }}</span>
              </label>
              <button 
                v-show="!isVirtualAgeCollapsed"
                @click="toggleExpand('vage')" 
                class="p-1 rounded text-gray-400 hover:text-gray-600 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-700 transition-colors"
              >
                <svg v-if="expandedChart === 'vage'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4h4v4m12-4h-4v4M4 20h4v-4m12 4h-4v-4" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4" />
                </svg>
              </button>
            </div>
          </div>
          <div v-show="!isVirtualAgeCollapsed" class="relative" :class="expandedChart === 'vage' ? 'h-[500px]' : 'h-80'">
            <canvas ref="vAgeChartRef"></canvas>
          </div>
        </div>

        <!-- Collapsible Intervals Details Table (TBF/TTX) -->
        <div v-if="hasFitData && fittedIntervals.length > 0" class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm mb-6">
          <div class="flex justify-between items-center mb-3">
            <div class="flex items-center gap-2">
              <h3 class="font-bold text-gray-800 dark:text-white">{{ $t('charts.kijima.intervals_details_title') }}</h3>
              <button 
                @click="isIntervalsCollapsed = !isIntervalsCollapsed"
                class="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-gray-600 dark:text-slate-300 transition-colors"
              >
                {{ isIntervalsCollapsed ? $t('charts.expand') + ' ⌄' : $t('charts.collapse') + ' ⌃' }}
              </button>
            </div>
          </div>
          
          <div v-show="!isIntervalsCollapsed" class="max-h-60 overflow-y-auto border border-gray-100 dark:border-slate-700 rounded">
            <table class="min-w-full divide-y divide-gray-200 dark:divide-slate-700 text-sm">
              <thead class="bg-gray-50 dark:bg-slate-900/60 sticky top-0">
                <tr>
                  <th class="px-4 py-2 text-left font-semibold text-gray-600 dark:text-slate-300"># Evento</th>
                  <th class="px-4 py-2 text-left font-semibold text-gray-600 dark:text-slate-300">Fecha</th>
                  <th class="px-4 py-2 text-right font-semibold text-gray-600 dark:text-slate-300">
                    {{ activeTab === 'TBX' ? 'TBF (hrs)' : 'TTX (hrs)' }}
                  </th>
                  <th class="px-4 py-2 text-left font-semibold text-gray-600 dark:text-slate-300">Tipo</th>
                  <th class="px-4 py-2 text-left font-semibold text-gray-600 dark:text-slate-300">Modo de Falla</th>
                  <th class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">Estado</th>
                  <th class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">Excluir</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-gray-200 dark:divide-slate-700 text-gray-900 dark:text-white">
                <tr v-for="item in fittedIntervals" :key="item.index + '-' + item.date" :class="item.is_baseline ? 'bg-gray-50/50 dark:bg-slate-800/50' : (item.included ? 'hover:bg-gray-50 dark:hover:bg-slate-700/30' : 'bg-red-50/20 dark:bg-red-950/10 opacity-70')">
                  <td class="px-4 py-2 font-medium">{{ item.is_baseline ? 'Base' : item.index }}</td>
                  <td class="px-4 py-2">{{ item.date }}</td>
                  <td class="px-4 py-2 text-right font-mono">{{ item.tbx.toFixed(1) }}</td>
                  <td class="px-4 py-2">{{ item.type }}</td>
                  <td class="px-4 py-2">{{ item.mode }}</td>
                  <td class="px-4 py-2 text-center">
                    <span v-if="item.is_baseline" class="px-2 py-0.5 text-xs font-semibold rounded bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300">
                      Inicio Historial
                    </span>
                    <span v-else-if="item.manually_excluded" class="px-2 py-0.5 text-xs font-semibold rounded bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400">
                      Excluido Manual
                    </span>
                    <span v-else-if="item.included" class="px-2 py-0.5 text-xs font-semibold rounded bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400">
                      Ajustado
                    </span>
                    <span v-else class="px-2 py-0.5 text-xs font-semibold rounded bg-orange-100 text-orange-700 dark:bg-orange-950/30 dark:text-orange-400">
                      Filtrado (&lt; {{ minTbx }}h)
                    </span>
                  </td>
                  <td class="px-4 py-2 text-center">
                    <input 
                      v-if="!item.is_baseline" 
                      type="checkbox" 
                      :checked="excludedIndices.includes(item.index)" 
                      @change="toggleIndexExclusion(item.index)" 
                      class="rounded text-red-600 focus:ring-red-500 cursor-pointer" 
                    />
                    <span v-else class="text-xs text-gray-400">-</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Comprehensive comparison table (TBX Mode Only) -->
        <div v-if="activeTab === 'TBX'" class="bg-white dark:bg-slate-800 p-4 rounded border border-gray-200 dark:border-slate-700 shadow-sm overflow-x-auto">
          <h3 class="font-bold text-gray-800 dark:text-white mb-3">{{ $t('charts.kijima.comparison_table') }}</h3>
          <table class="min-w-full divide-y divide-gray-200 dark:divide-slate-700 text-sm">
            <thead class="bg-gray-50 dark:bg-slate-900/60">
              <tr>
                <th scope="col" class="px-4 py-2 text-left font-semibold text-gray-600 dark:text-slate-300">{{ $t('charts.kijima.model_type') }}</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">β (Forma)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">η (Escala)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">a_r (Correctivo)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">a_p (Preventivo)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">b_r (Pendiente)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">b_p (Pendiente)</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">AIC</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">BIC</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">KS p-valor</th>
                <th scope="col" class="px-4 py-2 text-center font-semibold text-gray-600 dark:text-slate-300">MTBF Estimado</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-slate-700 text-gray-900 dark:text-white">
              <tr 
                v-for="row in comparisonTableData" 
                :key="row.name" 
                class="hover:bg-gray-50 dark:hover:bg-slate-700/30 transition-colors"
                :class="{ 'bg-blue-50/20 dark:bg-blue-900/10': row.active }"
              >
                <td class="px-4 py-2 font-medium" :style="{ color: row.color }">{{ row.name }}</td>
                <td class="px-4 py-2 text-center">{{ row.beta }}</td>
                <td class="px-4 py-2 text-center">{{ row.eta }}</td>
                <td class="px-4 py-2 text-center">{{ row.ar }}</td>
                <td class="px-4 py-2 text-center">{{ row.ap }}</td>
                <td class="px-4 py-2 text-center">{{ row.br }}</td>
                <td class="px-4 py-2 text-center">{{ row.bp }}</td>
                <td class="px-4 py-2 text-center" :class="{ 'font-bold': row.isBestAic }">{{ row.aic }}</td>
                <td class="px-4 py-2 text-center" :class="{ 'font-bold': row.isBestBic }">{{ row.bic }}</td>
                <td class="px-4 py-2 text-center">
                  <span 
                    :class="[
                      row.p_value >= 0.05 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400',
                      row.isBestKs ? 'font-bold underline underline-offset-2' : ''
                    ]"
                  >
                    {{ row.p_value }}
                  </span>
                </td>
                <td class="px-4 py-2 text-center font-semibold">{{ row.mtbf }}</td>
              </tr>
            </tbody>
          </table>
          <p class="text-xs text-gray-500 dark:text-slate-400 mt-2 font-medium">
            * {{ $t('charts.kijima.mtbf_note') }}
          </p>
        </div>

        <!-- Proactive Analysis Section (TBX Mode Only) -->
        <div v-if="activeTab === 'TBX'" id="proactive-pm-section" class="mt-8 border-t border-gray-200 dark:border-slate-700 pt-6">
          <h3 class="text-lg font-bold text-gray-800 dark:text-white mb-4">{{ $t('charts.weibull.proactive_calc') }}</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Optimal PM -->
            <div class="bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
              <h4 class="font-semibold text-gray-900 dark:text-white mb-3">{{ $t('charts.weibull.opt_pm') }}</h4>
              <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('charts.weibull.pm_cost') }}</label>
                  <input type="number" v-model.number="pmCost" class="input-field mt-1 w-full"/>
                </div>
                <div>
                  <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('charts.weibull.fail_cost') }}</label>
                  <input type="number" v-model.number="failureCost" class="input-field mt-1 w-full"/>
                </div>
              </div>
              <button @click="calculateOptimalPm" class="btn-secondary w-full">{{ $t('charts.weibull.calc_opt') }}</button>
              <div v-if="optimalPmInterval !== ''" class="mt-4 text-center bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 p-3 rounded">
                <p class="text-sm text-gray-600 dark:text-slate-400">{{ $t('charts.weibull.opt_interval') }}</p>
                <p class="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {{ typeof optimalPmInterval === 'number' ? optimalPmInterval.toFixed(1) + ' hrs' : (optimalPmInterval === 'Infinity' ? $t('charts.weibull.na_beta') : $t('charts.weibull.error')) }}
                </p>
              </div>
            </div>
            <!-- Conditional Reliability -->
            <div class="bg-gray-50 dark:bg-slate-900/50 p-4 rounded-lg">
              <h4 class="font-semibold text-gray-900 dark:text-white mb-3">{{ $t('charts.weibull.cond_rel') }}</h4>
              <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('charts.weibull.curr_age') }}</label>
                  <input type="number" v-model.number="currentAge" class="input-field mt-1 w-full"/>
                </div>
                <div>
                  <label class="block text-xs font-medium text-gray-600 dark:text-slate-400">{{ $t('charts.weibull.target_age') }}</label>
                  <input type="number" v-model.number="missionTime" class="input-field mt-1 w-full"/>
                </div>
              </div>
              <button @click="calculateConditionalReliability" class="btn-secondary w-full">{{ $t('charts.weibull.calc_mission') }}</button>
              <div v-if="conditionalReliability" class="mt-4 text-center bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 p-3 rounded">
                <template v-if="conditionalReliability === 'Error'">
                  <p class="text-red-600 font-bold">{{ $t('charts.weibull.calc_error') }}</p>
                </template>
                <template v-else>
                  <p class="text-sm text-gray-600 dark:text-slate-400">{{ $t('charts.weibull.success_prob') }}</p>
                  <p class="text-2xl font-bold" :class="conditionalReliability.success_probability > 0.9 ? 'text-green-600' : 'text-red-600'">
                    {{ (conditionalReliability.success_probability * 100).toFixed(1) }}%
                  </p>
                </template>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div v-else-if="!loading" class="text-center py-8 text-gray-500 dark:text-slate-400 border border-dashed rounded-lg border-gray-300 dark:border-slate-700">
        No hay datos suficientes para graficar. Haz clic en "Ajustar Curva".
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiService } from '../../api'
import { Chart } from 'chart.js/auto'

const { t, locale } = useI18n()

const props = defineProps({
  availableEquipment: Array,
  availableTypes: Array
})

const isCollapsed = ref(false)
const loading = ref(false)
const localFilters = ref({ equipment: '' })
const activeTab = ref('TBX') // 'TBX' or 'TTX'
const expandedChart = ref(null)
const isVirtualAgeCollapsed = ref(false)
const showNoRepairBaseline = ref(false)
const minTbx = ref(0.0)
const fittedIntervals = ref([])
const isIntervalsCollapsed = ref(true)
const excludedIndices = ref([])

const typesToFit = ref([])
const censoredTypes = ref([])

// Active curves to draw (for TBX mode)
const selectedCurves = ref(['weibull', 'k1_c', 'k2_c', 'k1_td', 'k2_td', 'k1_td2', 'k2_td2'])

// Fit results
const weibullResult = ref(null) // holds traditional Weibull results
const kijimaResult = ref([])   // holds Kijima models results

// Proactive analysis state
const pmCost = ref(100)
const failureCost = ref(1000)
const optimalPmInterval = ref('')
const currentAge = ref(0)
const missionTime = ref(100)
const conditionalReliability = ref(null)

const hasFitData = computed(() => {
  if (activeTab.value === 'TTX') {
    return weibullResult.value !== null
  }
  return weibullResult.value !== null || kijimaResult.value.length > 0
})

const modelOptions = computed(() => {
  return [
    { id: 'weibull', label: t('charts.kijima.weibull'), color: '#6366f1' },
    { id: 'k1_c',   label: t('charts.kijima.k1_c'),   color: '#3b82f6' },
    { id: 'k2_c',   label: t('charts.kijima.k2_c'),   color: '#10b981' },
    { id: 'k1_td',  label: t('charts.kijima.k1_td'),  color: '#f59e0b' },
    { id: 'k2_td',  label: t('charts.kijima.k2_td'),  color: '#ef4444' },
    { id: 'k1_td2', label: t('charts.kijima.k1_td2') || 'Kijima I TD2 (Logistic)', color: '#d946ef' },
    { id: 'k2_td2', label: t('charts.kijima.k2_td2') || 'Kijima II TD2 (Logistic)', color: '#f97316' },
  ]
})

const activeFitSummary = computed(() => {
  const list = []
  
  if (selectedCurves.value.includes('weibull') && weibullResult.value) {
    // weibullResult is either the direct TTX fit or parsed from Kijima list
    const params = weibullResult.value.parameters || weibullResult.value
    list.push({
      name: t('charts.kijima.weibull'),
      beta: params.beta?.toFixed(3) || '-',
      eta: params.eta?.toFixed(1) || '-',
      color: '#6366f1',
      bgClass: 'bg-indigo-50/30 border-indigo-100 dark:bg-indigo-950/10 dark:border-indigo-900/30'
    })
  }

  const namesMap = {
    'Kijima I': { id: 'k1_c', color: '#3b82f6', bg: 'bg-blue-50/30 border-blue-100 dark:bg-blue-950/10 dark:border-blue-900/30' },
    'Kijima II': { id: 'k2_c', color: '#10b981', bg: 'bg-emerald-50/30 border-emerald-100 dark:bg-emerald-950/10 dark:border-emerald-900/30' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b', bg: 'bg-amber-50/30 border-amber-100 dark:bg-amber-950/10 dark:border-amber-900/30' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444', bg: 'bg-rose-50/30 border-rose-100 dark:bg-rose-950/10 dark:border-rose-900/30' },
    'Kijima I TD2 (Logistic)': { id: 'k1_td2', color: '#d946ef', bg: 'bg-fuchsia-50/30 border-fuchsia-100 dark:bg-fuchsia-950/10 dark:border-fuchsia-900/30' },
    'Kijima II TD2 (Logistic)': { id: 'k2_td2', color: '#f97316', bg: 'bg-orange-50/30 border-orange-100 dark:bg-orange-950/10 dark:border-orange-900/30' },
  }

  kijimaResult.value.forEach(m => {
    const cfg = namesMap[m.model_name]
    if (cfg && selectedCurves.value.includes(cfg.id)) {
      list.push({
        name: m.model_name,
        beta: m.beta?.toFixed(3),
        eta: m.eta?.toFixed(1),
        ar: m.ar?.toFixed(3),
        ap: m.ap?.toFixed(3),
        br: m.br?.toFixed(4),
        bp: m.bp?.toFixed(4),
        color: cfg.color,
        bgClass: cfg.bg
      })
    }
  })

  return list
})

const comparisonTableData = computed(() => {
  const rows = []
  
  // Weibull
  if (weibullResult.value) {
    const wRes = weibullResult.value
    const params = wRes.parameters || wRes
    const gof = wRes.goodness_of_fit || wRes
    const wAic = gof.aic || gof.AIC
    const wBic = gof.bic || gof.BIC
    const wPv = gof.p_value
    const wMtbf = wRes.mtbf || wRes.mean

    rows.push({
      rawAic: (wAic != null && isFinite(wAic)) ? wAic : null,
      rawBic: (wBic != null && isFinite(wBic)) ? wBic : null,
      rawPv: (wPv != null && isFinite(wPv)) ? wPv : null,
      name: t('charts.kijima.weibull'),
      beta: params?.beta?.toFixed(3) || '-',
      eta: params?.eta?.toFixed(1) || '-',
      ar: '-',
      ap: '-',
      br: '-',
      bp: '-',
      aic: wAic != null && isFinite(wAic) ? wAic.toFixed(1) : '-',
      bic: wBic != null && isFinite(wBic) ? wBic.toFixed(1) : '-',
      p_value: wPv != null && isFinite(wPv) ? wPv.toFixed(4) : '-',
      mtbf: (wMtbf != null && isFinite(wMtbf)) ? wMtbf.toFixed(1) + ' hrs' : '-',
      color: '#6366f1',
      active: selectedCurves.value.includes('weibull')
    })
  }

  const namesMap = {
    'Kijima I': { id: 'k1_c', color: '#3b82f6' },
    'Kijima II': { id: 'k2_c', color: '#10b981' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444' },
    'Kijima I TD2 (Logistic)': { id: 'k1_td2', color: '#d946ef' },
    'Kijima II TD2 (Logistic)': { id: 'k2_td2', color: '#f97316' },
  }

  kijimaResult.value.forEach(m => {
    if (m.model_name === 'Weibull') return
    const cfg = namesMap[m.model_name]
    const mAic = m.AIC
    const mBic = m.BIC
    const mPv = m.p_value

    rows.push({
      rawAic: (mAic != null && isFinite(mAic)) ? mAic : null,
      rawBic: (mBic != null && isFinite(mBic)) ? mBic : null,
      rawPv: (mPv != null && isFinite(mPv)) ? mPv : null,
      name: m.model_name,
      beta: m.beta?.toFixed(3) || '-',
      eta: m.eta?.toFixed(1) || '-',
      ar: m.ar?.toFixed(3) || '-',
      ap: m.ap?.toFixed(3) || '-',
      br: m.br?.toFixed(4) || '-',
      bp: m.bp?.toFixed(4) || '-',
      aic: mAic != null && isFinite(mAic) ? mAic.toFixed(1) : '-',
      bic: mBic != null && isFinite(mBic) ? mBic.toFixed(1) : '-',
      p_value: mPv != null && isFinite(mPv) ? mPv.toFixed(4) : '-',
      mtbf: m.mean != null && isFinite(m.mean) ? m.mean.toFixed(1) + ' hrs' : '-',
      color: cfg?.color || '#94a3b8',
      active: cfg ? selectedCurves.value.includes(cfg.id) : false
    })
  })

  // Find min/max values to check for distinct winner
  let minAic = Infinity, maxAic = -Infinity
  let minBic = Infinity, maxBic = -Infinity
  let minPv = Infinity, maxPv = -Infinity

  rows.forEach(r => {
    if (r.rawAic !== null) {
      if (r.rawAic < minAic) minAic = r.rawAic
      if (r.rawAic > maxAic) maxAic = r.rawAic
    }
    if (r.rawBic !== null) {
      if (r.rawBic < minBic) minBic = r.rawBic
      if (r.rawBic > maxBic) maxBic = r.rawBic
    }
    if (r.rawPv !== null) {
      if (r.rawPv < minPv) minPv = r.rawPv
      if (r.rawPv > maxPv) maxPv = r.rawPv
    }
  })

  // Set isBest flags
  rows.forEach(r => {
    r.isBestAic = r.rawAic !== null && Math.abs(r.rawAic - minAic) < 1e-9 && minAic !== maxAic
    r.isBestBic = r.rawBic !== null && Math.abs(r.rawBic - minBic) < 1e-9 && minBic !== maxBic
    r.isBestKs = r.rawPv !== null && Math.abs(r.rawPv - maxPv) < 1e-9 && minPv !== maxPv
  })

  return rows
})

// Chart elements
const relChartRef = ref(null)
const hazardChartRef = ref(null)
const vAgeChartRef = ref(null)

let relChartInstance = null
let hazardChartInstance = null
let vAgeChartInstance = null

const toggleExpand = (chartKey) => {
  if (expandedChart.value === chartKey) {
    expandedChart.value = null
  } else {
    expandedChart.value = chartKey
  }
  nextTick(() => {
    if (relChartInstance) relChartInstance.resize()
    if (hazardChartInstance) hazardChartInstance.resize()
    if (vAgeChartInstance) vAgeChartInstance.resize()
  })
}

watch(selectedCurves, () => {
  renderCharts()
}, { deep: true })

watch(isVirtualAgeCollapsed, () => {
  nextTick(() => {
    if (!isVirtualAgeCollapsed.value) {
      renderVAgeChart()
    }
  })
})

watch(showNoRepairBaseline, () => {
  nextTick(() => {
    if (!isVirtualAgeCollapsed.value) {
      renderVAgeChart()
    }
  })
})

watch(() => props.availableEquipment, (newVal) => {
  if (newVal && newVal.length > 0 && (!localFilters.value.equipment || !newVal.includes(localFilters.value.equipment))) {
    localFilters.value.equipment = newVal[0]
  }
}, { immediate: true })

watch(activeTab, () => {
  excludedIndices.value = []
  loadAnalysis()
})

watch(() => localFilters.value.equipment, () => {
  excludedIndices.value = []
})

const toggleIndexExclusion = (index) => {
  const idx = excludedIndices.value.indexOf(index)
  if (idx > -1) {
    excludedIndices.value.splice(idx, 1)
  } else {
    excludedIndices.value.push(index)
  }
  loadAnalysis()
}

const loadAnalysis = async () => {
  loading.value = true
  try {
    await apiService.setFilters(localFilters.value.equipment)
    const toFit = typesToFit.value.length ? typesToFit.value : null
    const toCens = censoredTypes.value.length ? censoredTypes.value : null
    const exclList = excludedIndices.value.length ? excludedIndices.value : null

    if (activeTab.value === 'TTX') {
      // TTX Mode: Only Weibull fitting
      const res = await apiService.fitData(undefined, undefined, toFit, toCens, 'TTX', minTbx.value, exclList)
      weibullResult.value = res.data?.status === 'success' ? res.data : null
      fittedIntervals.value = res.data?.intervals || []
      kijimaResult.value = []
      selectedCurves.value = ['weibull']
    } else {
      // TBX Mode: Weibull + all Kijima models in parallel
      const kRes = await apiService.fitKijima(undefined, undefined, toFit, toCens, minTbx.value, exclList)
      if (kRes.data?.status === 'success') {
        kijimaResult.value = kRes.data.models || []
        fittedIntervals.value = kRes.data.intervals || []
        // Extract Weibull baseline if present
        const wModel = kijimaResult.value.find(m => m.model_name === 'Weibull')
        weibullResult.value = wModel || null
      } else {
        kijimaResult.value = []
        weibullResult.value = null
        fittedIntervals.value = []
      }
      selectedCurves.value = ['weibull', 'k1_c', 'k2_c', 'k1_td', 'k2_td', 'k1_td2', 'k2_td2']
    }

    await nextTick()
    renderCharts()
  } catch (err) {
    console.error('Error loading analysis curves:', err)
  } finally {
    loading.value = false
  }
}

const renderCharts = () => {
  if (!hasFitData.value) return

  renderRelAndHazardCharts()
  if (activeTab.value === 'TBX') {
    renderVAgeChart()
  }
}

const renderRelAndHazardCharts = () => {
  if (!relChartRef.value || !hazardChartRef.value) return
  if (relChartInstance) relChartInstance.destroy()
  if (hazardChartInstance) hazardChartInstance.destroy()

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const relDatasets = []
  const hazardDatasets = []

  const isTTX = activeTab.value === 'TTX'

  if (isTTX) {
    // TTX: Weibull CDF and PDF
    if (weibullResult.value && weibullResult.value.reliability_curve) {
      const curve = weibullResult.value.reliability_curve
      const times = curve.time || []
      
      const cdfPoints = times.map((t, idx) => ({ x: t, y: curve.cdf[idx] }))
      const pdfPoints = times.map((t, idx) => ({ x: t, y: curve.pdf[idx] }))

      relDatasets.push({
        label: t('charts.weibull.cum_prob') + ' F(t)',
        data: cdfPoints,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.1
      })

      hazardDatasets.push({
        label: t('charts.weibull.prob_density') + ' f(t)',
        data: pdfPoints,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        fill: true,
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.1
      })
    }
  } else {
    // TBX: Kijima + Weibull curves
    const kNames = {
      'Weibull': { id: 'weibull', color: '#6366f1' },
      'Kijima I': { id: 'k1_c', color: '#3b82f6' },
      'Kijima II': { id: 'k2_c', color: '#10b981' },
      'Kijima I TD': { id: 'k1_td', color: '#f59e0b' },
      'Kijima II TD': { id: 'k2_td', color: '#ef4444' },
      'Kijima I TD2 (Logistic)': { id: 'k1_td2', color: '#d946ef' },
      'Kijima II TD2 (Logistic)': { id: 'k2_td2', color: '#f97316' },
    }

    kijimaResult.value.forEach(m => {
      const cfg = kNames[m.model_name]
      if (cfg && selectedCurves.value.includes(cfg.id)) {
        const betaStr = m.beta?.toFixed(2)
        const etaStr = m.eta?.toFixed(1)
        const arStr = m.ar?.toFixed(2)
        const apStr = m.ap?.toFixed(2)
        const brStr = m.br !== 0 ? `, br=${m.br?.toFixed(4)}` : ''
        const bpStr = m.bp !== 0 ? `, bp=${m.bp?.toFixed(4)}` : ''
        
        let legendLabel = `${m.model_name} (β=${betaStr}, η=${etaStr})`
        if (m.model_name !== 'Weibull') {
          legendLabel = `${m.model_name} (β=${betaStr}, η=${etaStr}, ar=${arStr}, ap=${apStr}${brStr}${bpStr})`
        }

        const times = m.t || []
        const relPoints = times.map((t, idx) => ({ x: t, y: m.R[idx] })).filter(p => p.x != null && p.y != null && isFinite(p.y))
        const hazPoints = times.map((t, idx) => ({ x: t, y: m.failure_rate[idx] })).filter(p => p.x != null && p.y != null && isFinite(p.y))

        relDatasets.push({
          label: legendLabel,
          data: relPoints,
          borderColor: cfg.color,
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0
        })

        hazardDatasets.push({
          label: legendLabel,
          data: hazPoints,
          borderColor: cfg.color,
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0
        })
      }
    })
  }

  // Setup options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: textColor,
          font: { size: 10 }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: t('charts.weibull.time') || 'Tiempo',
          color: textColor
        },
        ticks: { color: textColor },
        grid: { color: gridColor }
      },
      y: {
        ticks: { color: textColor },
        grid: { color: gridColor }
      }
    }
  }

  relChartInstance = new Chart(relChartRef.value, {
    type: 'line',
    data: { datasets: relDatasets },
    options: {
      ...chartOptions,
      scales: {
        ...chartOptions.scales,
        y: {
          ...chartOptions.scales.y,
          min: isTTX ? 0 : undefined,
          max: isTTX ? 1.05 : undefined,
          title: {
            display: true,
            text: isTTX ? 'F(t)' : 'R(t)',
            color: textColor
          }
        }
      }
    }
  })

  hazardChartInstance = new Chart(hazardChartRef.value, {
    type: 'line',
    data: { datasets: hazardDatasets },
    options: {
      ...chartOptions,
      scales: {
        ...chartOptions.scales,
        y: {
          ...chartOptions.scales.y,
          title: {
            display: true,
            text: isTTX ? 'f(t)' : 'h(t)',
            color: textColor
          }
        }
      }
    }
  })
}

const renderVAgeChart = () => {
  if (!vAgeChartRef.value || isVirtualAgeCollapsed.value) return
  if (vAgeChartInstance) vAgeChartInstance.destroy()

  const isDark = document.documentElement.classList.contains('dark')
  const textColor = isDark ? '#cbd5e1' : '#475569'
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'

  const vAgeDatasets = []

  const kNames = {
    'Weibull': { id: 'weibull', color: '#6366f1' },
    'Kijima I': { id: 'k1_c', color: '#3b82f6' },
    'Kijima II': { id: 'k2_c', color: '#10b981' },
    'Kijima I TD': { id: 'k1_td', color: '#f59e0b' },
    'Kijima II TD': { id: 'k2_td', color: '#ef4444' },
    'Kijima I TD2 (Logistic)': { id: 'k1_td2', color: '#d946ef' },
    'Kijima II TD2 (Logistic)': { id: 'k2_td2', color: '#f97316' },
  }

  // Sawtooth virtual age curves
  kijimaResult.value.forEach(m => {
    const cfg = kNames[m.model_name]
    // Weibull has no virtual age curve, but Kijima models do
    if (m.model_name !== 'Weibull' && cfg && selectedCurves.value.includes(cfg.id)) {
      const times = m.t || []
      const points = times.map((t, idx) => ({ x: t, y: m.V_curve[idx] })).filter(p => p.x != null && p.y != null && isFinite(p.y))
      
      vAgeDatasets.push({
        label: `${m.model_name} V(t)`,
        data: points,
        borderColor: cfg.color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 1,
        pointBackgroundColor: cfg.color,
        showLine: true,
        tension: 0
      })
    }
  })

  // Ideal baseline (V(t) = t)
  if (vAgeDatasets.length > 0 && showNoRepairBaseline.value) {
    const maxT = Math.max(...kijimaResult.value.map(m => m.T ? m.T[m.T.length - 1] : 0))
    vAgeDatasets.push({
      label: t('charts.kijima.no_repair_baseline'),
      data: [{ x: 0, y: 0 }, { x: maxT, y: maxT }],
      borderColor: '#94a3b8',
      borderDash: [5, 5],
      borderWidth: 1.5,
      pointRadius: 0,
      backgroundColor: 'transparent'
    })
  }

  vAgeChartInstance = new Chart(vAgeChartRef.value, {
    type: 'line',
    data: { datasets: vAgeDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: textColor }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: t('charts.kijima.cum_calendar_time'),
            color: textColor
          },
          ticks: { color: textColor },
          grid: { color: gridColor }
        },
        y: {
          title: {
            display: true,
            text: t('charts.kijima.virtual_age_y_axis'),
            color: textColor
          },
          ticks: { color: textColor },
          grid: { color: gridColor }
        }
      }
    }
  })
}

// Proactive calculations
const getApiFilters = () => ({
  equipment: localFilters.value.equipment || undefined,
  failure_type: undefined,
  types_to_fit: typesToFit.value.length ? typesToFit.value : null,
  censored_failure_types: censoredTypes.value.length ? censoredTypes.value : null,
  target_column: 'TBX'
})

const calculateOptimalPm = async () => {
  try {
    const filters = getApiFilters()
    const res = await apiService.getOptimalPm(filters, pmCost.value, failureCost.value)
    optimalPmInterval.value = res.data.optimal_pm_interval === null ? 'Infinity' : res.data.optimal_pm_interval
  } catch (err) {
    console.error(err)
    optimalPmInterval.value = 'Error'
  }
}

const calculateConditionalReliability = async () => {
  try {
    const filters = getApiFilters()
    const res = await apiService.getConditionalReliability(filters, currentAge.value, missionTime.value)
    conditionalReliability.value = res.data
  } catch (err) {
    console.error(err)
    conditionalReliability.value = 'Error'
  }
}

const handleThemeChange = () => {
  renderCharts()
}

watch(locale, () => {
  renderCharts()
})

onMounted(() => {
  loadAnalysis()
  window.addEventListener('theme-changed', handleThemeChange)
})

onUnmounted(() => {
  window.removeEventListener('theme-changed', handleThemeChange)
})
</script>
