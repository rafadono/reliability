<template>
  <div 
    v-if="isOpen" 
    class="absolute bottom-0 left-0 right-0 h-64 bg-slate-900/95 text-slate-200 border-t border-slate-700/80 backdrop-blur-md z-40 flex flex-col shadow-2xl animate-slide-up font-mono text-xs"
  >
    <!-- Cabecera de la Consola de Logs -->
    <div class="px-4 py-2 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span class="font-bold text-slate-100 uppercase tracking-wider text-[11px]">Consola de Diagnóstico & Registros</span>
        </div>

        <!-- Filtros por Nivel de Log -->
        <div class="flex items-center gap-1 bg-slate-900 p-0.5 rounded border border-slate-800 text-[10px]">
          <button 
            @click="activeFilter = 'ALL'" 
            class="px-2 py-0.5 rounded font-bold transition-colors cursor-pointer"
            :class="activeFilter === 'ALL' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'"
          >
            Todos ({{ logs.length }})
          </button>
          <button 
            @click="activeFilter = 'INFO'" 
            class="px-2 py-0.5 rounded font-bold transition-colors cursor-pointer"
            :class="activeFilter === 'INFO' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-blue-400'"
          >
            Info ({{ infoCount }})
          </button>
          <button 
            @click="activeFilter = 'WARNING'" 
            class="px-2 py-0.5 rounded font-bold transition-colors cursor-pointer"
            :class="activeFilter === 'WARNING' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-amber-400'"
          >
            Warnings ({{ warningCount }})
          </button>
          <button 
            @click="activeFilter = 'ERROR'" 
            class="px-2 py-0.5 rounded font-bold transition-colors cursor-pointer"
            :class="activeFilter === 'ERROR' ? 'bg-red-600 text-white' : 'text-slate-400 hover:text-red-400'"
          >
            Errores ({{ errorCount }})
          </button>
        </div>
      </div>

      <!-- Acciones de la Consola -->
      <div class="flex items-center gap-2">
        <button 
          @click="copyLogs" 
          title="Copiar logs al portapapeles"
          class="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] flex items-center gap-1 transition-all cursor-pointer"
        >
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          {{ copied ? '¡Copiado!' : 'Copiar' }}
        </button>
        <button 
          @click="$emit('clear')" 
          title="Limpiar historial de la consola"
          class="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] transition-all cursor-pointer"
        >
          Limpiar
        </button>
        <button 
          @click="$emit('close')" 
          title="Cerrar consola"
          class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition-colors cursor-pointer"
        >
          ✕
        </button>
      </div>
    </div>

    <!-- Cuerpo de Logs / Lista de Registros -->
    <div class="flex-1 overflow-y-auto p-3 space-y-1.5 scrollbar-thin">
      <div v-if="filteredLogs.length === 0" class="text-slate-500 italic text-center py-6">
        No hay registros de diagnóstico para mostrar.
      </div>
      <div 
        v-for="(log, idx) in filteredLogs" 
        :key="log.id || idx" 
        class="flex items-start gap-2 py-1 px-2 rounded hover:bg-slate-800/60 transition-colors border-l-2"
        :class="getBorderColor(log.level)"
      >
        <span class="text-slate-500 shrink-0">[{{ log.timestamp }}]</span>
        <span class="px-1.5 py-0.2 text-[9px] font-bold rounded uppercase shrink-0" :class="getBadgeClass(log.level)">
          {{ log.level }}
        </span>
        <span class="text-indigo-400 font-semibold shrink-0">[{{ log.node_type || 'dag' }}: {{ log.node_id }}]</span>
        <span class="text-slate-300 break-all leading-snug">{{ log.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  isOpen: { type: Boolean, default: false },
  logs: { type: Array, default: () => [] }
})

defineEmits(['close', 'clear'])

const activeFilter = ref('ALL')
const copied = ref(false)

const filteredLogs = computed(() => {
  if (activeFilter.value === 'ALL') return props.logs
  return props.logs.filter(l => l.level === activeFilter.value)
})

const infoCount = computed(() => props.logs.filter(l => l.level === 'INFO').length)
const warningCount = computed(() => props.logs.filter(l => l.level === 'WARNING').length)
const errorCount = computed(() => props.logs.filter(l => l.level === 'ERROR').length)

const getBadgeClass = (level) => {
  if (level === 'ERROR') return 'bg-red-950 text-red-400 border border-red-800'
  if (level === 'WARNING') return 'bg-amber-950 text-amber-400 border border-amber-800'
  return 'bg-blue-950 text-blue-400 border border-blue-800'
}

const getBorderColor = (level) => {
  if (level === 'ERROR') return 'border-red-500'
  if (level === 'WARNING') return 'border-amber-500'
  return 'border-blue-500'
}

const copyLogs = () => {
  const text = filteredLogs.value.map(l => `[${l.timestamp}] [${l.level}] [${l.node_type}:${l.node_id}] ${l.message}`).join('\n')
  navigator.clipboard.writeText(text)
  copied.value = true
  setTimeout(() => { copied.value = false }, 2000)
}
</script>

<style scoped>
.animate-slide-up {
  animation: slideUp 0.25s ease-out;
}
@keyframes slideUp {
  from { transform: translateY(100%); }
  to { transform: translateY(0); }
}
</style>
