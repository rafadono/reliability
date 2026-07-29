<template>
  <div class="space-y-4">
    <!-- Weibull -->
    <div v-if="node.type === 'weibull'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Calcula la confiabilidad estructural ajustando una distribución biparamétrica de Weibull (beta y eta) al conjunto de datos filtrado.
      </p>
      <div class="bg-gray-50 dark:bg-slate-950/50 rounded-lg p-3 border border-gray-100 dark:border-slate-800 text-xs">
        <span class="font-bold text-gray-700 dark:text-slate-300 block mb-1">Método de Ajuste:</span>
        <span class="text-gray-600 dark:text-slate-400">Regresión lineal de rangos (Median Ranks) de mínimos cuadrados sobre curva de distribución acumulada CDF.</span>
      </div>
    </div>

    <!-- Pareto -->
    <div v-if="node.type === 'pareto'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Prioriza las causas raíz mediante la regla del 80/20 acumulando horas de parada o frecuencia por equipo y modo.
      </p>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Agrupar por</label>
        <select
          v-model="node.data.group_by"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="Equipment">Equipo</option>
          <option value="Type">Tipo de Falla</option>
          <option value="mdf">Modo de Falla</option>
        </select>
      </div>
    </div>

    <!-- Jackknife -->
    <div v-if="node.type === 'jackknife'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Evalúa la variabilidad relativa graficando Frecuencia vs Tiempo Fuera de Servicio.
      </p>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Comparar por</label>
        <select
          v-model="node.data.compare_by"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="Equipment">Equipo</option>
          <option value="Type">Tipo de Falla</option>
          <option value="mdf">Modo de Falla</option>
        </select>
      </div>
    </div>

    <!-- Criticality Matrix -->
    <div v-if="node.type === 'criticality'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Matriz de Criticidad Frecuencia vs Consecuencia (Horas).
      </p>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Comparar por</label>
        <select
          v-model="node.data.compare_by"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="equipment">Equipo</option>
          <option value="type">Tipo de Falla</option>
          <option value="mode">Modo de Falla</option>
        </select>
      </div>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Métrica del eje X</label>
        <select
          v-model="node.data.metric_x"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="count">Frecuencia (conteo)</option>
          <option value="probability">Probabilidad (%)</option>
        </select>
      </div>
    </div>

    <!-- APM / Bad Actors -->
    <div v-if="node.type === 'apm'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Ranking de equipos "Bad Actors" por horas de detención acumuladas (Análisis de Modo Predominante).
      </p>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Comparar por</label>
        <select
          v-model="node.data.compare_by"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="equipment">Equipo</option>
          <option value="type">Tipo de Falla</option>
        </select>
      </div>
    </div>

    <!-- Comment Mining -->
    <div v-if="node.type === 'comment_mining'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Extracción NLP de patrones recurrentes en comentarios técnicos (clasificación por palabras clave y cobertura de categorías).
      </p>
      <div class="bg-gray-50 dark:bg-slate-950/50 rounded-lg p-3 border border-gray-100 dark:border-slate-800 text-xs text-gray-600 dark:text-slate-400">
        Este bloque ejecuta el modelo <strong>Legacy Keyword NLP</strong> como vista previa rápida dentro del pipeline. Para comparar contra modelos semánticos de Hugging Face, usa la pestaña <strong>Copiloto IA</strong>.
      </div>
    </div>

    <!-- KPI Trend -->
    <div v-if="node.type === 'trend'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Muestra el perfil histórico mensual y las tendencias de fallas, MTBF, MTTR y disponibilidad.
      </p>
    </div>

    <!-- RAM Simulator -->
    <div v-if="node.type === 'ram' || node.type === 'ramSimulator'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Simulación de Disponibilidad y Aseguramiento de Producción según ISO 20815.
      </p>
      <div class="space-y-2">
        <div class="flex justify-between items-center text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
          <span>Eficiencia Preventiva</span>
          <span class="text-indigo-600 dark:text-indigo-400">{{ Math.round((node.data.preventive_efficiency ?? 0.8) * 100) }}%</span>
        </div>
        <input
          type="range" min="0" max="1" step="0.05"
          v-model.number="node.data.preventive_efficiency"
          class="w-full h-1.5 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
        />
      </div>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider block">Demora Logística Promedio (hrs)</label>
        <input
          type="number" v-model.number="node.data.logistics_delay" min="0" step="0.5"
          class="w-full text-xs border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none"
        />
      </div>
    </div>

    <!-- RCM Assistant -->
    <div v-if="node.type === 'rcm'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Genera fichas RCM (7 preguntas, SAE JA1011) para el equipo seleccionado, usando IA sobre el historial de comentarios.
      </p>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Equipo</label>
        <select
          v-model="node.data.equipment"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="">(usar equipo del filtro aguas arriba)</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
      </div>
    </div>

    <!-- RCA -->
    <div v-if="node.type === 'rca'" class="space-y-4">
      <p class="text-xs text-gray-500 dark:text-slate-400 leading-relaxed">
        Genera 5 Porqués y diagrama de Ishikawa (IEC 62740) para el equipo seleccionado, usando IA sobre el historial de comentarios.
      </p>
      <div class="space-y-1">
        <label class="text-[10px] font-bold text-gray-500 dark:text-slate-400 uppercase tracking-wider">Equipo</label>
        <select
          v-model="node.data.equipment"
          class="w-full text-xs bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
        >
          <option value="">(usar equipo del filtro aguas arriba)</option>
          <option v-for="eq in availableEquipment" :key="eq" :value="eq">{{ eq }}</option>
        </select>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  node: { type: Object, required: true },
  availableEquipment: { type: Array, default: () => [] }
})
</script>
