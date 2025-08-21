<template>
  <div class="max-w-4xl mx-auto space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Create New Banner</h1>
      <p class="mt-1 text-sm text-gray-500">
        Generate a professional banner using AI-powered multi-agent system
      </p>
    </div>

    <!-- Creation Form -->
    <form @submit.prevent="handleSubmit" class="space-y-6">
      <!-- Basic Information -->
      <div class="card">
        <h2 class="text-lg font-medium text-gray-900 mb-4">Basic Information</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label for="name" class="form-label">Banner Name *</label>
            <input
              id="name"
              v-model="form.name"
              type="text"
              required
              class="form-input"
              placeholder="e.g., Summer Sale Campaign"
            />
          </div>
          
          <div>
            <label for="description" class="form-label">Description</label>
            <input
              id="description"
              v-model="form.description"
              type="text"
              class="form-input"
              placeholder="Brief description of the banner"
            />
          </div>
        </div>
      </div>

      <!-- Campaign Brief -->
      <div class="card">
        <h2 class="text-lg font-medium text-gray-900 mb-4">Campaign Brief</h2>
        
        <div>
          <label for="campaign_brief" class="form-label">Campaign Brief *</label>
          <textarea
            id="campaign_brief"
            v-model="form.campaign_brief"
            required
            rows="4"
            class="form-input"
            placeholder="Describe your campaign goals, target audience, key messages, and any specific requirements..."
          ></textarea>
          <p class="mt-2 text-sm text-gray-500">
            Provide detailed information about your campaign to help our AI agents create the perfect banner.
          </p>
        </div>
      </div>

      <!-- Brand Guidelines -->
      <div class="card">
        <h2 class="text-lg font-medium text-gray-900 mb-4">Brand Guidelines</h2>
        
        <div class="space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label for="brand_colors" class="form-label">Brand Colors</label>
              <input
                id="brand_colors"
                v-model="form.brand_guidelines.colors"
                type="text"
                class="form-input"
                placeholder="e.g., #FF6B6B, #4ECDC4, #45B7D1"
              />
              <p class="mt-1 text-xs text-gray-500">
                Comma-separated hex color codes
              </p>
            </div>
            
            <div>
              <label for="brand_fonts" class="form-label">Brand Fonts</label>
              <input
                id="brand_fonts"
                v-model="form.brand_guidelines.fonts"
                type="text"
                class="form-input"
                placeholder="e.g., Inter, Roboto, Open Sans"
              />
              <p class="mt-1 text-xs text-gray-500">
                Comma-separated font names
              </p>
            </div>
          </div>
          
          <div>
            <label for="brand_voice" class="form-label">Brand Voice & Tone</label>
            <select
              id="brand_voice"
              v-model="form.brand_guidelines.voice_tone"
              class="form-input"
            >
              <option value="">Select tone...</option>
              <option value="professional">Professional</option>
              <option value="friendly">Friendly</option>
              <option value="playful">Playful</option>
              <option value="authoritative">Authoritative</option>
              <option value="casual">Casual</option>
              <option value="luxury">Luxury</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Target Audience -->
      <div class="card">
        <h2 class="text-lg font-medium text-gray-900 mb-4">Target Audience</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label for="age_group" class="form-label">Age Group</label>
            <select
              id="age_group"
              v-model="form.target_audience.age_group"
              class="form-input"
            >
              <option value="">Select age group...</option>
              <option value="18-24">18-24</option>
              <option value="25-34">25-34</option>
              <option value="35-44">35-44</option>
              <option value="45-54">45-54</option>
              <option value="55+">55+</option>
            </select>
          </div>
          
          <div>
            <label for="interests" class="form-label">Interests</label>
            <input
              id="interests"
              v-model="form.target_audience.interests"
              type="text"
              class="form-input"
              placeholder="e.g., technology, fashion, fitness"
            />
          </div>
        </div>
      </div>

      <!-- Design Preferences -->
      <div class="card">
        <h2 class="text-lg font-medium text-gray-900 mb-4">Design Preferences</h2>
        
        <div class="space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label for="style" class="form-label">Style</label>
              <select
                id="style"
                v-model="form.design_preferences.style"
                class="form-input"
              >
                <option value="">Select style...</option>
                <option value="modern">Modern</option>
                <option value="classic">Classic</option>
                <option value="minimalist">Minimalist</option>
                <option value="bold">Bold</option>
                <option value="elegant">Elegant</option>
                <option value="playful">Playful</option>
              </select>
            </div>
            
            <div>
              <label for="color_scheme" class="form-label">Color Scheme</label>
              <select
                id="color_scheme"
                v-model="form.design_preferences.color_scheme"
                class="form-input"
              >
                <option value="">Select scheme...</option>
                <option value="vibrant">Vibrant</option>
                <option value="muted">Muted</option>
                <option value="monochromatic">Monochromatic</option>
                <option value="complementary">Complementary</option>
                <option value="analogous">Analogous</option>
              </select>
            </div>
            
            <div>
              <label for="layout" class="form-label">Layout</label>
              <select
                id="layout"
                v-model="form.design_preferences.layout"
                class="form-input"
              >
                <option value="">Select layout...</option>
                <option value="centered">Centered</option>
                <option value="left_aligned">Left Aligned</option>
                <option value="right_aligned">Right Aligned</option>
                <option value="grid">Grid</option>
                <option value="asymmetric">Asymmetric</option>
              </select>
            </div>
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label for="banner_size" class="form-label">Banner Size</label>
              <select
                id="banner_size"
                v-model="form.design_preferences.size"
                class="form-input"
              >
                <option value="">Select size...</option>
                <option value="leaderboard">Leaderboard (728x90)</option>
                <option value="rectangle">Rectangle (300x250)</option>
                <option value="skyscraper">Skyscraper (160x600)</option>
                <option value="square">Square (250x250)</option>
                <option value="banner">Banner (468x60)</option>
                <option value="custom">Custom Size</option>
              </select>
            </div>
            
            <div v-if="form.design_preferences.size === 'custom'" class="flex space-x-2">
              <div class="flex-1">
                <label for="custom_width" class="form-label">Width (px)</label>
                <input
                  id="custom_width"
                  v-model.number="form.design_preferences.custom_width"
                  type="number"
                  class="form-input"
                  placeholder="800"
                />
              </div>
              <div class="flex-1">
                <label for="custom_height" class="form-label">Height (px)</label>
                <input
                  id="custom_height"
                  v-model.number="form.design_preferences.custom_height"
                  type="number"
                  class="form-input"
                  placeholder="600"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Asset Upload -->
      <div class="card">
        <h2 class="text-lg font-medium text-gray-900 mb-4">Assets (Optional)</h2>
        
        <div>
          <label class="form-label">Upload Images/Logos</label>
          <div 
            class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-gray-400 transition-colors"
            @drop="handleDrop"
            @dragover.prevent
            @dragenter.prevent
          >
            <div class="space-y-1 text-center">
              <PhotoIcon class="mx-auto h-12 w-12 text-gray-400" />
              <div class="flex text-sm text-gray-600">
                <label 
                  for="file-upload" 
                  class="relative cursor-pointer bg-white rounded-md font-medium text-primary-600 hover:text-primary-500"
                >
                  <span>Upload files</span>
                  <input 
                    id="file-upload" 
                    type="file" 
                    class="sr-only" 
                    multiple 
                    accept="image/*"
                    @change="handleFileSelect"
                  />
                </label>
                <p class="pl-1">or drag and drop</p>
              </div>
              <p class="text-xs text-gray-500">
                PNG, JPG, SVG up to 10MB each
              </p>
            </div>
          </div>
          
          <!-- Selected Files -->
          <div v-if="selectedFiles.length > 0" class="mt-4 space-y-2">
            <h4 class="text-sm font-medium text-gray-900">Selected Files:</h4>
            <div class="space-y-2">
              <div 
                v-for="(file, index) in selectedFiles" 
                :key="index"
                class="flex items-center justify-between p-2 bg-gray-50 rounded-md"
              >
                <div class="flex items-center space-x-2">
                  <PhotoIcon class="h-5 w-5 text-gray-400" />
                  <span class="text-sm text-gray-900">{{ file.name }}</span>
                  <span class="text-xs text-gray-500">({{ formatFileSize(file.size) }})</span>
                </div>
                <button
                  type="button"
                  @click="removeFile(index)"
                  class="text-red-500 hover:text-red-700"
                >
                  <XMarkIcon class="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="flex justify-end space-x-4">
        <button
          type="button"
          @click="$router.go(-1)"
          class="btn btn-secondary"
        >
          Cancel
        </button>
        <button
          type="submit"
          :disabled="bannerStore.loading || !form.name || !form.campaign_brief"
          class="btn btn-primary"
        >
          <div v-if="bannerStore.loading" class="flex items-center">
            <div class="spinner w-4 h-4 mr-2"></div>
            Creating Banner...
          </div>
          <div v-else class="flex items-center">
            <SparklesIcon class="h-5 w-5 mr-2" />
            Generate Banner
          </div>
        </button>
      </div>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { 
  PhotoIcon, 
  XMarkIcon,
  SparklesIcon
} from '@heroicons/vue/24/outline'
import { useBannerStore } from '@/stores/banners'
import { useNotificationStore } from '@/stores/notifications'
import type { CreateBannerRequest } from '@/stores/banners'

const router = useRouter()
const bannerStore = useBannerStore()
const notificationStore = useNotificationStore()

const selectedFiles = ref<File[]>([])

const form = reactive<CreateBannerRequest>({
  name: '',
  description: '',
  campaign_brief: '',
  brand_guidelines: {
    colors: '',
    fonts: '',
    voice_tone: ''
  },
  target_audience: {
    age_group: '',
    interests: ''
  },
  design_preferences: {
    style: '',
    color_scheme: '',
    layout: '',
    size: '',
    custom_width: undefined,
    custom_height: undefined
  }
})

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files) {
    addFiles(Array.from(target.files))
  }
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  if (event.dataTransfer?.files) {
    addFiles(Array.from(event.dataTransfer.files))
  }
}

const addFiles = (files: File[]) => {
  for (const file of files) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      notificationStore.warning('Invalid File', `${file.name} is not an image file`)
      continue
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      notificationStore.warning('File Too Large', `${file.name} exceeds 10MB limit`)
      continue
    }
    
    selectedFiles.value.push(file)
  }
}

const removeFile = (index: number) => {
  selectedFiles.value.splice(index, 1)
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const handleSubmit = async () => {
  try {
    // Prepare the request
    const request: CreateBannerRequest = {
      ...form,
      assets: selectedFiles.value
    }
    
    // Create the banner
    const banner = await bannerStore.createBanner(request)
    
    if (banner) {
      // Navigate to the banner detail page
      router.push(`/banners/${banner.id}`)
    }
  } catch (error) {
    console.error('Failed to create banner:', error)
  }
}
</script>
