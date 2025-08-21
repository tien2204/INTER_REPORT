<template>
  <div v-if="banner" class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <button
          @click="$router.go(-1)"
          class="text-gray-400 hover:text-gray-500"
        >
          <ArrowLeftIcon class="h-6 w-6" />
        </button>
        <div>
          <h1 class="text-2xl font-bold text-gray-900">{{ banner.name }}</h1>
          <p v-if="banner.description" class="mt-1 text-sm text-gray-500">
            {{ banner.description }}
          </p>
        </div>
      </div>
      
      <div class="flex items-center space-x-3">
        <span 
          class="status-badge"
          :class="{
            'status-success': banner.status === 'completed',
            'status-warning': banner.status === 'processing',
            'status-error': banner.status === 'failed',
            'status-info': banner.status === 'draft'
          }"
        >
          {{ banner.status }}
        </span>
        
        <div v-if="banner.status === 'completed'" class="flex items-center space-x-2">
          <button
            @click="downloadBanner('svg')"
            class="btn btn-secondary btn-sm"
          >
            <ArrowDownTrayIcon class="h-4 w-4 mr-2" />
            SVG
          </button>
          <button
            @click="downloadBanner('html')"
            class="btn btn-secondary btn-sm"
          >
            <ArrowDownTrayIcon class="h-4 w-4 mr-2" />
            HTML
          </button>
          <button
            @click="downloadBanner('png')"
            class="btn btn-secondary btn-sm"
          >
            <ArrowDownTrayIcon class="h-4 w-4 mr-2" />
            PNG
          </button>
        </div>
        
        <button
          v-if="banner.status === 'failed'"
          @click="retryBanner"
          class="btn btn-primary btn-sm"
        >
          <ArrowPathIcon class="h-4 w-4 mr-2" />
          Retry
        </button>
      </div>
    </div>

    <!-- Progress Section (for processing banners) -->
    <div v-if="banner.status === 'processing'" class="card">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-medium text-gray-900">Generation Progress</h2>
        <span class="text-sm text-gray-500">{{ banner.progress_percentage }}%</span>
      </div>
      
      <div class="progress-bar mb-4">
        <div 
          class="progress-fill"
          :style="{ width: `${banner.progress_percentage}%` }"
        ></div>
      </div>
      
      <div class="flex justify-between items-center text-sm text-gray-500">
        <span>Current step: {{ banner.current_step }}</span>
        <span>Agent: {{ banner.current_agent }}</span>
      </div>
    </div>

    <!-- Error Section (for failed banners) -->
    <div v-if="banner.status === 'failed' && banner.errors?.length" class="card border-red-200">
      <div class="flex items-center mb-4">
        <ExclamationTriangleIcon class="h-5 w-5 text-red-400 mr-2" />
        <h2 class="text-lg font-medium text-red-900">Generation Failed</h2>
      </div>
      
      <div class="space-y-2">
        <div 
          v-for="(error, index) in banner.errors" 
          :key="index"
          class="p-3 bg-red-50 border border-red-200 rounded-md"
        >
          <p class="text-sm text-red-800">{{ error }}</p>
        </div>
      </div>
    </div>

    <!-- Preview Section -->
    <div v-if="banner.preview_urls || banner.status === 'completed'" class="card">
      <h2 class="text-lg font-medium text-gray-900 mb-4">Preview</h2>
      
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Visual Preview -->
        <div>
          <h3 class="text-sm font-medium text-gray-700 mb-2">Visual Preview</h3>
          <div class="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
            <div v-if="banner.preview_urls?.png || banner.preview_urls?.svg" class="w-full h-full">
              <img 
                :src="banner.preview_urls.png || banner.preview_urls.svg" 
                :alt="banner.name"
                class="w-full h-full object-contain rounded-lg"
              />
            </div>
            <div v-else class="text-gray-400">
              <PhotoIcon class="h-16 w-16" />
            </div>
          </div>
        </div>
        
        <!-- Code Preview -->
        <div v-if="banner.generated_code">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Generated Code</h3>
          <div class="bg-gray-900 rounded-lg p-4 overflow-auto max-h-96">
            <pre class="text-sm text-green-400"><code>{{ formatCode(banner.generated_code) }}</code></pre>
          </div>
        </div>
      </div>
    </div>

    <!-- Design Data -->
    <div v-if="banner.design_data" class="card">
      <h2 class="text-lg font-medium text-gray-900 mb-4">Design Information</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Campaign Brief -->
        <div v-if="banner.design_data.campaign_brief">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Campaign Brief</h3>
          <p class="text-sm text-gray-600 p-3 bg-gray-50 rounded-lg">
            {{ banner.design_data.campaign_brief }}
          </p>
        </div>
        
        <!-- Brand Guidelines -->
        <div v-if="banner.design_data.brand_guidelines">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Brand Guidelines</h3>
          <div class="space-y-2">
            <div v-if="banner.design_data.brand_guidelines.colors" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Colors</h4>
              <div class="flex items-center space-x-2 mt-1">
                <div 
                  v-for="color in banner.design_data.brand_guidelines.colors.split(',')" 
                  :key="color"
                  class="w-6 h-6 rounded border border-gray-300"
                  :style="{ backgroundColor: color.trim() }"
                ></div>
                <span class="text-sm text-gray-600">{{ banner.design_data.brand_guidelines.colors }}</span>
              </div>
            </div>
            
            <div v-if="banner.design_data.brand_guidelines.fonts" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Fonts</h4>
              <p class="text-sm text-gray-600 mt-1">{{ banner.design_data.brand_guidelines.fonts }}</p>
            </div>
            
            <div v-if="banner.design_data.brand_guidelines.voice_tone" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Voice & Tone</h4>
              <p class="text-sm text-gray-600 mt-1 capitalize">{{ banner.design_data.brand_guidelines.voice_tone }}</p>
            </div>
          </div>
        </div>
        
        <!-- Target Audience -->
        <div v-if="banner.design_data.target_audience">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Target Audience</h3>
          <div class="space-y-2">
            <div v-if="banner.design_data.target_audience.age_group" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Age Group</h4>
              <p class="text-sm text-gray-600 mt-1">{{ banner.design_data.target_audience.age_group }}</p>
            </div>
            
            <div v-if="banner.design_data.target_audience.interests" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Interests</h4>
              <p class="text-sm text-gray-600 mt-1">{{ banner.design_data.target_audience.interests }}</p>
            </div>
          </div>
        </div>
        
        <!-- Design Preferences -->
        <div v-if="banner.design_data.design_preferences">
          <h3 class="text-sm font-medium text-gray-700 mb-2">Design Preferences</h3>
          <div class="space-y-2">
            <div v-if="banner.design_data.design_preferences.style" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Style</h4>
              <p class="text-sm text-gray-600 mt-1 capitalize">{{ banner.design_data.design_preferences.style }}</p>
            </div>
            
            <div v-if="banner.design_data.design_preferences.color_scheme" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Color Scheme</h4>
              <p class="text-sm text-gray-600 mt-1 capitalize">{{ banner.design_data.design_preferences.color_scheme }}</p>
            </div>
            
            <div v-if="banner.design_data.design_preferences.layout" class="p-3 bg-gray-50 rounded-lg">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">Layout</h4>
              <p class="text-sm text-gray-600 mt-1 capitalize">{{ banner.design_data.design_preferences.layout }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Metadata -->
    <div class="card">
      <h2 class="text-lg font-medium text-gray-900 mb-4">Banner Details</h2>
      
      <dl class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <dt class="text-sm font-medium text-gray-500">Created</dt>
          <dd class="text-sm text-gray-900">{{ formatDate(banner.created_at) }}</dd>
        </div>
        
        <div>
          <dt class="text-sm font-medium text-gray-500">Last Updated</dt>
          <dd class="text-sm text-gray-900">{{ formatDate(banner.updated_at) }}</dd>
        </div>
        
        <div v-if="banner.completed_at">
          <dt class="text-sm font-medium text-gray-500">Completed</dt>
          <dd class="text-sm text-gray-900">{{ formatDate(banner.completed_at) }}</dd>
        </div>
        
        <div v-if="banner.campaign_id">
          <dt class="text-sm font-medium text-gray-500">Campaign ID</dt>
          <dd class="text-sm text-gray-900">{{ banner.campaign_id }}</dd>
        </div>
      </dl>
    </div>
  </div>
  
  <!-- Loading State -->
  <div v-else-if="bannerStore.loading" class="flex justify-center py-12">
    <div class="spinner w-8 h-8"></div>
  </div>
  
  <!-- Not Found -->
  <div v-else class="text-center py-12">
    <PhotoIcon class="mx-auto h-12 w-12 text-gray-400" />
    <h3 class="mt-2 text-sm font-medium text-gray-900">Banner not found</h3>
    <p class="mt-1 text-sm text-gray-500">The banner you're looking for doesn't exist.</p>
    <div class="mt-6">
      <router-link to="/campaigns" class="btn btn-primary">
        Back to Banners
      </router-link>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { 
  ArrowLeftIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon,
  PhotoIcon,
  ExclamationTriangleIcon
} from '@heroicons/vue/24/outline'
import { useBannerStore } from '@/stores/banners'
import { useWebSocketStore } from '@/stores/websocket'

const route = useRoute()
const bannerStore = useBannerStore()
const websocketStore = useWebSocketStore()

const bannerId = computed(() => route.params.id as string)
const banner = computed(() => bannerStore.currentBanner)

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleString()
}

const formatCode = (code: any) => {
  if (typeof code === 'string') {
    return code
  }
  return JSON.stringify(code, null, 2)
}

const downloadBanner = async (format: 'svg' | 'html' | 'png' | 'figma') => {
  if (banner.value) {
    await bannerStore.downloadBanner(banner.value.id, format)
  }
}

const retryBanner = async () => {
  if (banner.value) {
    await bannerStore.retryBanner(banner.value.id)
  }
}

onMounted(async () => {
  // Fetch banner details
  await bannerStore.fetchBanner(bannerId.value)
  
  // Subscribe to real-time updates for this banner
  if (websocketStore.isConnected) {
    websocketStore.subscribeToDesign(bannerId.value)
  }
})

onUnmounted(() => {
  // Unsubscribe from real-time updates
  if (websocketStore.isConnected) {
    websocketStore.unsubscribeFromDesign(bannerId.value)
  }
})
</script>
