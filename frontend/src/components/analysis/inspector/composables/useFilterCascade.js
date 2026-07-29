import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { apiService } from '../../../../api'

export function useFilterCascade(props, emit) {
  const showTypeDropdown = ref(false)
  const showMdfDropdown = ref(false)
  const showCensoredTypesDropdown = ref(false)
  const showCensoredMdfsDropdown = ref(false)
  const censoredMdfOptions = ref([])

  const selectedTypesText = computed(() => {
    const selected = props.node?.data?.type || []
    if (selected.length === 0) return 'Todos los Tipos'
    return selected.join(', ')
  })

  const selectedMdfsText = computed(() => {
    const selected = props.node?.data?.mdf || []
    if (selected.length === 0) return 'Todos los Modos'
    return selected.join(', ')
  })

  const selectedCensoredTypesText = computed(() => {
    const selected = props.node?.data?.censored_types || []
    if (selected.length === 0) return 'Ningún Tipo Censurado'
    return selected.join(', ')
  })

  const selectedCensoredMdfsText = computed(() => {
    const selected = props.node?.data?.censored_mdfs || []
    const censoredTypes = props.node?.data?.censored_types || []
    if (selected.length === 0) {
      if (censoredTypes.length > 0) {
        return `Todos los Modos (${censoredTypes.join(', ')})`
      }
      return 'Ningún Modo Censurado'
    }
    return selected.join(', ')
  })

  const isAllCensoredMdfsSelected = computed(() => {
    const selected = props.node?.data?.censored_mdfs || []
    return selected.length === 0
  })

  // Exclusión Mutua entre Selección de Fallas y Censura
  const availableTypesForFailure = computed(() => {
    const allTypes = props.filterOptions?.types || []
    const censoredTypes = props.node?.data?.censored_types || []
    return allTypes.filter(t => !censoredTypes.includes(t))
  })

  const availableTypesForCensorship = computed(() => {
    const allTypes = props.filterOptions?.types || []
    const failureTypes = props.node?.data?.type || []
    return allTypes.filter(t => !failureTypes.includes(t))
  })

  const availableMdfsForFailure = computed(() => {
    const allMdfs = props.filterOptions?.mdfs || []
    const censoredMdfs = props.node?.data?.censored_mdfs || []
    return allMdfs.filter(m => !censoredMdfs.includes(m))
  })

  const availableMdfsForCensorship = computed(() => {
    const allCensoredMdfs = censoredMdfOptions.value || []
    const failureMdfs = props.node?.data?.mdf || []
    return allCensoredMdfs.filter(m => !failureMdfs.includes(m))
  })

  const updateCensoredMdfOptions = async () => {
    if (!props.node || props.node.type !== 'filter') return
    const plant = props.node?.data?.plant || undefined
    const eq = props.node?.data?.equipment || undefined
    const cTypes = props.node?.data?.censored_types
    if (cTypes && Array.isArray(cTypes) && cTypes.length > 0) {
      try {
        const res = await apiService.getFilters(plant, eq, cTypes)
        censoredMdfOptions.value = res.data?.failure_modes || []
      } catch (e) {
        censoredMdfOptions.value = props.filterOptions?.mdfs || []
      }
    } else {
      censoredMdfOptions.value = props.filterOptions?.mdfs || []
    }
  }

  const onFilterChange = async () => {
    emit('filter-changed', {
      plant: props.node?.data?.plant,
      equipment: props.node?.data?.equipment,
      type: props.node?.data?.type,
      censored_types: props.node?.data?.censored_types
    })
    await updateCensoredMdfOptions()
  }

  const toggleAllTypes = (event) => {
    if (event.target.checked && props.node?.data) {
      props.node.data.type = []
    }
    onFilterChange()
  }

  const toggleAllMdfs = (event) => {
    if (event.target.checked && props.node?.data) {
      props.node.data.mdf = []
    }
  }

  const toggleAllCensoredTypes = (event) => {
    if (event.target.checked && props.node?.data) {
      props.node.data.censored_types = []
    }
    onFilterChange()
  }

  const toggleAllCensoredMdfs = (event) => {
    if (event.target.checked && props.node?.data) {
      props.node.data.censored_mdfs = []
    }
  }

  const handleClickOutside = (e) => {
    if (!e.target.closest('.type-dropdown-container')) showTypeDropdown.value = false
    if (!e.target.closest('.mdf-dropdown-container')) showMdfDropdown.value = false
    if (!e.target.closest('.censored-types-container')) showCensoredTypesDropdown.value = false
    if (!e.target.closest('.censored-mdfs-container')) showCensoredMdfsDropdown.value = false
  }

  onMounted(() => {
    document.addEventListener('click', handleClickOutside)
  })

  onUnmounted(() => {
    document.removeEventListener('click', handleClickOutside)
  })

  watch(
    () => [
      props.node?.data?.equipment,
      props.node?.data?.censored_types ? (Array.isArray(props.node.data.censored_types) ? props.node.data.censored_types.join(',') : props.node.data.censored_types) : '',
      props.filterOptions?.mdfs ? props.filterOptions.mdfs.join(',') : ''
    ],
    updateCensoredMdfOptions,
    { immediate: true }
  )

  return {
    showTypeDropdown,
    showMdfDropdown,
    showCensoredTypesDropdown,
    showCensoredMdfsDropdown,
    selectedTypesText,
    selectedMdfsText,
    selectedCensoredTypesText,
    selectedCensoredMdfsText,
    isAllCensoredMdfsSelected,
    availableTypesForFailure,
    availableTypesForCensorship,
    availableMdfsForFailure,
    availableMdfsForCensorship,
    toggleAllTypes,
    toggleAllMdfs,
    toggleAllCensoredTypes,
    toggleAllCensoredMdfs,
    onFilterChange
  }
}
