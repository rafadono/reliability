import { createI18n } from 'vue-i18n'
import en from './locales/en.json'
import es from './locales/es.json'

// Check browser language or use local storage
const getBrowserLang = () => {
  const storedLang = localStorage.getItem('app-lang')
  if (storedLang) return storedLang
  
  const browserLang = navigator.language.split('-')[0]
  return browserLang === 'es' ? 'es' : 'en'
}

const i18n = createI18n({
  legacy: false, // use Composition API
  globalInjection: true, // IMPORTANT: Allows $t in all templates
  locale: getBrowserLang(),
  fallbackLocale: 'en',
  messages: {
    en,
    es
  }
})

export default i18n
