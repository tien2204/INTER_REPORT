import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'
import { useNotificationStore } from './notifications'

export interface Banner {
  id: string
  name: string
  description?: string
  status: 'draft' | 'processing' | 'completed' | 'failed'
  progress_percentage: number
  current_step: string
  current_agent: string
  created_at: string
  updated_at: string
  completed_at?: string
  campaign_id?: string
  design_data?: any
  generated_code?: any
  preview_urls?: {
    svg?: string
    html?: string
    png?: string
  }
  errors?: string[]
}

export interface CreateBannerRequest {
  name: string
  description?: string
  campaign_brief: string
  brand_guidelines?: any
  target_audience?: any
  design_preferences?: any
  assets?: File[]
}

export const useBannerStore = defineStore('banners', () => {
  const banners = ref<Banner[]>([])
  const currentBanner = ref<Banner | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const notificationStore = useNotificationStore()

  // Computed
  const recentBanners = computed(() => 
    banners.value
      .slice()
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(0, 10)
  )

  const activeBanners = computed(() => 
    banners.value.filter(banner => banner.status === 'processing')
  )

  const completedBanners = computed(() => 
    banners.value.filter(banner => banner.status === 'completed')
  )

  const failedBanners = computed(() => 
    banners.value.filter(banner => banner.status === 'failed')
  )

  const bannerStats = computed(() => ({
    total: banners.value.length,
    processing: activeBanners.value.length,
    completed: completedBanners.value.length,
    failed: failedBanners.value.length,
    draft: banners.value.filter(b => b.status === 'draft').length
  }))

  // Actions
  const fetchBanners = async () => {
    try {
      loading.value = true
      error.value = null
      
      const response = await apiService.get('/designs')
      banners.value = response.data.designs || []
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch banners'
      notificationStore.error('Error', 'Failed to load banners')
    } finally {
      loading.value = false
    }
  }

  const fetchBanner = async (id: string) => {
    try {
      loading.value = true
      error.value = null
      
      const response = await apiService.get(`/designs/${id}`)
      const banner = response.data
      
      // Update in list if exists
      const index = banners.value.findIndex(b => b.id === id)
      if (index !== -1) {
        banners.value[index] = banner
      } else {
        banners.value.push(banner)
      }
      
      currentBanner.value = banner
      return banner
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch banner'
      notificationStore.error('Error', 'Failed to load banner details')
      return null
    } finally {
      loading.value = false
    }
  }

  const createBanner = async (request: CreateBannerRequest) => {
    try {
      loading.value = true
      error.value = null

      // Create form data for file upload
      const formData = new FormData()
      formData.append('name', request.name)
      if (request.description) {
        formData.append('description', request.description)
      }
      formData.append('campaign_brief', request.campaign_brief)
      
      if (request.brand_guidelines) {
        formData.append('brand_guidelines', JSON.stringify(request.brand_guidelines))
      }
      if (request.target_audience) {
        formData.append('target_audience', JSON.stringify(request.target_audience))
      }
      if (request.design_preferences) {
        formData.append('design_preferences', JSON.stringify(request.design_preferences))
      }

      // Add asset files
      if (request.assets?.length) {
        request.assets.forEach((file, index) => {
          formData.append(`assets[${index}]`, file)
        })
      }

      const response = await apiService.post('/designs/generate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      const newBanner = response.data
      banners.value.unshift(newBanner)
      currentBanner.value = newBanner

      notificationStore.success(
        'Banner Created', 
        `"${request.name}" is being generated`
      )

      return newBanner
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to create banner'
      notificationStore.error('Error', 'Failed to create banner')
      throw err
    } finally {
      loading.value = false
    }
  }

  const updateBanner = async (id: string, updates: Partial<Banner>) => {
    try {
      const response = await apiService.put(`/designs/${id}`, updates)
      const updatedBanner = response.data

      // Update in list
      const index = banners.value.findIndex(b => b.id === id)
      if (index !== -1) {
        banners.value[index] = updatedBanner
      }

      // Update current if it's the same banner
      if (currentBanner.value?.id === id) {
        currentBanner.value = updatedBanner
      }

      return updatedBanner
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to update banner'
      notificationStore.error('Error', 'Failed to update banner')
      throw err
    }
  }

  const deleteBanner = async (id: string) => {
    try {
      await apiService.delete(`/designs/${id}`)
      
      // Remove from list
      const index = banners.value.findIndex(b => b.id === id)
      if (index !== -1) {
        banners.value.splice(index, 1)
      }

      // Clear current if it's the same banner
      if (currentBanner.value?.id === id) {
        currentBanner.value = null
      }

      notificationStore.success('Deleted', 'Banner has been deleted')
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to delete banner'
      notificationStore.error('Error', 'Failed to delete banner')
      throw err
    }
  }

  const retryBanner = async (id: string) => {
    try {
      const response = await apiService.post(`/designs/${id}/retry`)
      const updatedBanner = response.data

      // Update in list
      const index = banners.value.findIndex(b => b.id === id)
      if (index !== -1) {
        banners.value[index] = updatedBanner
      }

      notificationStore.success('Retrying', 'Banner generation restarted')
      return updatedBanner
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to retry banner'
      notificationStore.error('Error', 'Failed to retry banner generation')
      throw err
    }
  }

  const downloadBanner = async (id: string, format: 'svg' | 'html' | 'png' | 'figma') => {
    try {
      const response = await apiService.get(`/designs/${id}/download/${format}`, {
        responseType: 'blob'
      })

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      
      const banner = banners.value.find(b => b.id === id)
      const filename = `${banner?.name || 'banner'}.${format}`
      link.setAttribute('download', filename)
      
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)

      notificationStore.success('Downloaded', `${format.toUpperCase()} file downloaded`)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to download banner'
      notificationStore.error('Error', 'Failed to download banner')
      throw err
    }
  }

  const updateBannerProgress = (id: string, progress: Partial<Banner>) => {
    const index = banners.value.findIndex(b => b.id === id)
    if (index !== -1) {
      banners.value[index] = { ...banners.value[index], ...progress }
    }

    if (currentBanner.value?.id === id) {
      currentBanner.value = { ...currentBanner.value, ...progress }
    }
  }

  const clearError = () => {
    error.value = null
  }

  const setBanner = (banner: Banner) => {
    currentBanner.value = banner
  }

  return {
    // State
    banners,
    currentBanner,
    loading,
    error,
    
    // Computed
    recentBanners,
    activeBanners,
    completedBanners,
    failedBanners,
    bannerStats,
    
    // Actions
    fetchBanners,
    fetchBanner,
    createBanner,
    updateBanner,
    deleteBanner,
    retryBanner,
    downloadBanner,
    updateBannerProgress,
    clearError,
    setBanner
  }
})
