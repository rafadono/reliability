<template>
  <div class="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 p-6 transition-all duration-300 flex flex-col h-[550px]">
    <div class="border-b border-gray-100 dark:border-slate-700 pb-4 mb-4 shrink-0">
      <h4 class="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <svg class="w-5 h-5 text-indigo-600 dark:text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
        {{ $t('charts.iso_analysis.copilot_title') }}
      </h4>
      <p class="text-xs text-gray-500 dark:text-slate-400">{{ $t('charts.iso_analysis.copilot_desc') }}</p>
    </div>

    <!-- Historial de Chat -->
    <div class="flex-1 overflow-y-auto space-y-4 pr-2 mb-4 scrollbar-thin" ref="chatHistoryRef">
      <div 
        v-for="(msg, idx) in messages" 
        :key="idx" 
        class="flex gap-3"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <!-- Avatar Bot -->
        <div 
          v-if="msg.role === 'assistant'" 
          class="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-950 flex items-center justify-center shrink-0 border border-indigo-200/50"
        >
          <span class="text-indigo-600 dark:text-indigo-400 font-bold text-xs">AI</span>
        </div>

        <!-- Mensaje -->
        <div 
          class="max-w-[75%] rounded-2xl px-4 py-2.5 text-xs shadow-sm leading-relaxed"
          :class="[
            msg.role === 'user' 
              ? 'bg-indigo-600 text-white rounded-tr-none' 
              : 'bg-gray-50 dark:bg-slate-900/60 text-gray-800 dark:text-slate-200 border border-gray-100 dark:border-slate-800 rounded-tl-none'
          ]"
        >
          <p class="whitespace-pre-line">{{ msg.content }}</p>
          <span class="text-[9px] opacity-60 block mt-1.5 text-right">{{ msg.time }}</span>
        </div>

        <!-- Avatar Usuario -->
        <div 
          v-if="msg.role === 'user'" 
          class="w-8 h-8 rounded-lg bg-slate-200 dark:bg-slate-700 flex items-center justify-center shrink-0"
        >
          <span class="text-slate-700 dark:text-slate-300 font-bold text-xs">US</span>
        </div>
      </div>

      <!-- Indicador de Carga / IA Escribiendo -->
      <div v-if="typing" class="flex gap-3 justify-start items-center">
        <div class="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-950 flex items-center justify-center shrink-0 border border-indigo-200/50">
          <span class="text-indigo-600 dark:text-indigo-400 font-bold text-xs animate-pulse">AI</span>
        </div>
        <div class="bg-gray-50 dark:bg-slate-900/60 border border-gray-100 dark:border-slate-800 rounded-2xl rounded-tl-none px-4 py-3 flex gap-1 items-center">
          <div class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></div>
          <div class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
          <div class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
        </div>
      </div>
    </div>

    <!-- Sugerencias de Preguntas Rápidas -->
    <div class="flex flex-wrap gap-2 mb-3 shrink-0">
      <button 
        v-for="(q, i) in quickQuestions" 
        :key="i"
        @click="sendQuickQuestion(q)"
        class="text-[10px] font-semibold bg-gray-100 hover:bg-gray-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-gray-700 dark:text-slate-300 px-2.5 py-1 rounded-full transition-colors"
      >
        {{ q.label }}
      </button>
    </div>

    <!-- Entrada del Mensaje -->
    <div class="flex gap-2 shrink-0">
      <input 
        v-model="userInput" 
        @keyup.enter="sendMessage"
        type="text" 
        :placeholder="$t('charts.iso_analysis.chat_placeholder')" 
        class="flex-1 text-xs bg-gray-50 dark:bg-slate-900/60 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-white rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
        :disabled="typing"
      />
      <button 
        @click="sendMessage"
        :disabled="!userInput.trim() || typing"
        class="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 dark:disabled:bg-slate-700 disabled:text-gray-500 text-white font-bold px-4 rounded-xl transition-colors flex items-center justify-center shadow-md"
      >
        <svg class="w-4 h-4 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const chatHistoryRef = ref(null)
const userInput = ref('')
const typing = ref(false)

const messages = ref([
  {
    role: 'assistant',
    content: t('charts.iso_analysis.welcome_msg'),
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
])

const quickQuestions = computed(() => [
  { label: t('charts.iso_analysis.q1'), text: t('charts.iso_analysis.q1') },
  { label: t('charts.iso_analysis.q2'), text: t('charts.iso_analysis.q2') },
  { label: t('charts.iso_analysis.q3'), text: t('charts.iso_analysis.q3') },
  { label: t('charts.iso_analysis.q4'), text: t('charts.iso_analysis.q4') }
])

const scrollToBottom = async () => {
  await nextTick()
  if (chatHistoryRef.value) {
    chatHistoryRef.value.scrollTop = chatHistoryRef.value.scrollHeight
  }
}

const sendQuickQuestion = (q) => {
  userInput.value = q.text
  sendMessage()
}

const sendMessage = async () => {
  if (!userInput.value.trim() || typing.value) return

  const userText = userInput.value.trim()
  userInput.value = ''

  messages.value.push({
    role: 'user',
    content: userText,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  })
  
  await scrollToBottom()
  typing.value = true

  // Simulamos una respuesta sofisticada del Agente Coordinador bilingüe
  setTimeout(async () => {
    let aiResponse = ''
    const lowerText = userText.toLowerCase()

    if (lowerText.includes('rcm') || lowerText.includes('ja1011') || lowerText.includes('ja1012')) {
      aiResponse = t('charts.iso_analysis.res_rcm')
    } else if (lowerText.includes('rpn') || lowerText.includes('fmeca') || lowerText.includes('60812') || lowerText.includes('risk') || lowerText.includes('riesgo')) {
      aiResponse = t('charts.iso_analysis.res_rpn')
    } else if (lowerText.includes('iso 20815') || lowerText.includes('ram')) {
      aiResponse = t('charts.iso_analysis.res_ram')
    } else if (lowerText.includes('rodamiento') || lowerText.includes('falla') || lowerText.includes('bearing') || lowerText.includes('failure')) {
      aiResponse = t('charts.iso_analysis.res_rodamiento')
    } else {
      aiResponse = t('charts.iso_analysis.res_default')
    }

    messages.value.push({
      role: 'assistant',
      content: aiResponse,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    })

    typing.value = false
    await scrollToBottom()
  }, 1200)
}

onMounted(() => {
  scrollToBottom()
})
</script>

<style scoped>
.scrollbar-thin::-webkit-scrollbar {
  width: 5px;
}
.scrollbar-thin::-webkit-scrollbar-track {
  background: transparent;
}
.scrollbar-thin::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}
.dark .scrollbar-thin::-webkit-scrollbar-thumb {
  background: #475569;
}
</style>
