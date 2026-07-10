<template>
  <div class="relative overflow-hidden w-full h-full">
    <!-- Contenido Original (Borrosidad si no está activo) -->
    <div :class="[
      'transition-all duration-500 w-full h-full',
      !active ? 'filter blur-sm pointer-events-none select-none opacity-50' : ''
    ]">
      <slot />
    </div>

    <!-- Capa de Bloqueo Glassmorphic -->
    <div 
      v-if="!active" 
      class="absolute inset-0 bg-slate-950/20 dark:bg-slate-950/40 backdrop-blur-[4px] flex items-center justify-center p-6 z-20 transition-all duration-500 animate-fade-in"
    >
      <div class="bg-white/90 dark:bg-slate-900/90 border border-gray-200/80 dark:border-slate-700/80 rounded-2xl p-6 shadow-2xl max-w-sm text-center transform scale-100 transition-all duration-300 backdrop-blur-md">
        <!-- Icono de Candado con Brillo Pulsante -->
        <div class="w-12 h-12 bg-indigo-500/10 dark:bg-indigo-500/20 border border-indigo-500/20 text-indigo-600 dark:text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-4 shadow-[0_0_15px_rgba(99,102,241,0.15)]">
          <svg class="w-6 h-6 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>

        <!-- Titulo y Mensaje Localizado -->
        <h4 class="text-base font-extrabold text-gray-900 dark:text-white mb-2">
          {{ $t('workbench.locked_title') }}
        </h4>
        <p class="text-xs text-gray-600 dark:text-slate-300 leading-relaxed mb-5 font-medium">
          {{ $t(`workbench.locked_${nodeType}`) }}
        </p>

        <!-- Botón de Acción para volver al Workbench -->
        <button 
          @click="$emit('navigate', 'workbench')"
          class="w-full py-2 px-4 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold transition-all shadow-lg shadow-indigo-500/20 active:scale-95 flex items-center justify-center gap-1.5"
        >
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
          {{ $t('workbench.go_to_workbench') }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  active: {
    type: Boolean,
    required: true
  },
  nodeType: {
    type: String,
    required: true
  }
})

defineEmits(['navigate'])
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>
