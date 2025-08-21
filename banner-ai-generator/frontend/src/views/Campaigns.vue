<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Banners</h1>
        <p class="mt-1 text-sm text-gray-500">
          Manage all your AI-generated banners
        </p>
      </div>
      <router-link 
        to="/create" 
        class="btn btn-primary"
      >
        <PlusIcon class="h-5 w-5 mr-2" />
        Create Banner
      </router-link>
    </div>

    <!-- Filters and Search -->
    <div class="card">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
        <div class="flex items-center space-x-4">
          <!-- Search -->
          <div class="relative">
            <MagnifyingGlassIcon class="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search banners..."
              class="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            />
          </div>
          
          <!-- Status Filter -->
          <select
            v-model="statusFilter"
            class="border border-gray-300 rounded-md px-3 py-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
          >
            <option value="">All Status</option>
            <option value="draft">Draft</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        
        <!-- View Toggle -->
        <div class="flex items-center space-x-2">
          <button
            @click="viewMode = 'grid'"
            :class="[
              'p-2 rounded-md',
              viewMode === 'grid' 
                ? 'bg-primary-100 text-primary-600' 
                : 'text-gray-400 hover:text-gray-500'
            ]"
          >
            <Squares2X2Icon class="h-5 w-5" />
          </button>
          <button
            @click="viewMode = 'list'"
            :class="[
              'p-2 rounded-md',
              viewMode === 'list' 
                ? 'bg-primary-100 text-primary-600' 
                : 'text-gray-400 hover:text-gray-500'
            ]"
          >
            <ListBulletIcon class="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="bannerStore.loading" class="flex justify-center py-12">
      <div class="spinner w-8 h-8"></div>
    </div>

    <!-- Empty State -->
    <div 
      v-else-if="filteredBanners.length === 0 && !searchQuery && !statusFilter"
      class="text-center py-12"
    >
      <PhotoIcon class="mx-auto h-12 w-12 text-gray-400" />
      <h3 class="mt-2 text-sm font-medium text-gray-900">No banners</h3>
      <p class="mt-1 text-sm text-gray-500">Get started by creating your first banner.</p>
      <div class="mt-6">
        <router-link to="/create" class="btn btn-primary">
          <PlusIcon class="h-5 w-5 mr-2" />
          Create Banner
        </router-link>
      </div>
    </div>

    <!-- No Results -->
    <div 
      v-else-if="filteredBanners.length === 0"
      class="text-center py-12"
    >
      <MagnifyingGlassIcon class="mx-auto h-12 w-12 text-gray-400" />
      <h3 class="mt-2 text-sm font-medium text-gray-900">No results found</h3>
      <p class="mt-1 text-sm text-gray-500">Try adjusting your search or filter criteria.</p>
    </div>

    <!-- Grid View -->
    <div 
      v-else-if="viewMode === 'grid'"
      class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
    >
      <div
        v-for="banner in filteredBanners"
        :key="banner.id"
        class="card hover:shadow-md transition-shadow cursor-pointer"
        @click="$router.push(`/banners/${banner.id}`)"
      >
        <!-- Preview -->
        <div class="aspect-video bg-gray-100 rounded-lg mb-4 flex items-center justify-center">
          <div v-if="banner.preview_urls?.png" class="w-full h-full">
            <img 
              :src="banner.preview_urls.png" 
              :alt="banner.name"
              class="w-full h-full object-cover rounded-lg"
            />
          </div>
          <div v-else class="text-gray-400">
            <PhotoIcon class="h-12 w-12" />
          </div>
        </div>
        
        <!-- Content -->
        <div>
          <div class="flex items-start justify-between">
            <div class="flex-1 min-w-0">
              <h3 class="text-sm font-medium text-gray-900 truncate">
                {{ banner.name }}
              </h3>
              <p v-if="banner.description" class="text-xs text-gray-500 mt-1 line-clamp-2">
                {{ banner.description }}
              </p>
            </div>
            <span 
              class="ml-2 status-badge"
              :class="{
                'status-success': banner.status === 'completed',
                'status-warning': banner.status === 'processing',
                'status-error': banner.status === 'failed',
                'status-info': banner.status === 'draft'
              }"
            >
              {{ banner.status }}
            </span>
          </div>
          
          <!-- Progress Bar for Processing -->
          <div v-if="banner.status === 'processing'" class="mt-3">
            <div class="progress-bar">
              <div 
                class="progress-fill"
                :style="{ width: `${banner.progress_percentage}%` }"
              ></div>
            </div>
            <div class="flex justify-between text-xs text-gray-500 mt-1">
              <span>{{ banner.current_step }}</span>
              <span>{{ banner.progress_percentage }}%</span>
            </div>
          </div>
          
          <!-- Meta -->
          <div class="flex items-center justify-between mt-3 text-xs text-gray-500">
            <span>{{ formatDate(banner.updated_at) }}</span>
            <div class="flex items-center space-x-2">
              <button
                v-if="banner.status === 'completed'"
                @click.stop="downloadBanner(banner.id, 'svg')"
                class="text-primary-600 hover:text-primary-500"
              >
                Download
              </button>
              <button
                v-if="banner.status === 'failed'"
                @click.stop="retryBanner(banner.id)"
                class="text-primary-600 hover:text-primary-500"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- List View -->
    <div v-else class="card overflow-hidden">
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Banner
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Progress
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Updated
              </th>
              <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr 
              v-for="banner in filteredBanners"
              :key="banner.id"
              class="hover:bg-gray-50 cursor-pointer"
              @click="$router.push(`/banners/${banner.id}`)"
            >
              <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                  <div class="flex-shrink-0 h-10 w-10">
                    <div v-if="banner.preview_urls?.png" class="h-10 w-10 rounded-lg overflow-hidden">
                      <img 
                        :src="banner.preview_urls.png" 
                        :alt="banner.name"
                        class="h-full w-full object-cover"
                      />
                    </div>
                    <div v-else class="h-10 w-10 bg-gray-100 rounded-lg flex items-center justify-center">
                      <PhotoIcon class="h-6 w-6 text-gray-400" />
                    </div>
                  </div>
                  <div class="ml-4">
                    <div class="text-sm font-medium text-gray-900">
                      {{ banner.name }}
                    </div>
                    <div v-if="banner.description" class="text-sm text-gray-500">
                      {{ banner.description }}
                    </div>
                  </div>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
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
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <div v-if="banner.status === 'processing'" class="w-full">
                  <div class="progress-bar">
                    <div 
                      class="progress-fill"
                      :style="{ width: `${banner.progress_percentage}%` }"
                    ></div>
                  </div>
                  <div class="text-xs text-gray-500 mt-1">
                    {{ banner.current_step }}
                  </div>
                </div>
                <span v-else class="text-sm text-gray-500">
                  {{ banner.status === 'completed' ? '100%' : '-' }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {{ formatDate(banner.updated_at) }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <div class="flex items-center justify-end space-x-2">
                  <button
                    v-if="banner.status === 'completed'"
                    @click.stop="downloadBanner(banner.id, 'svg')"
                    class="text-primary-600 hover:text-primary-900"
                  >
                    Download
                  </button>
                  <button
                    v-if="banner.status === 'failed'"
                    @click.stop="retryBanner(banner.id)"
                    class="text-primary-600 hover:text-primary-900"
                  >
                    Retry
                  </button>
                  <button
                    @click.stop="deleteBanner(banner.id)"
                    class="text-red-600 hover:text-red-900"
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { 
  PlusIcon,
  PhotoIcon,
  MagnifyingGlassIcon,
  Squares2X2Icon,
  ListBulletIcon
} from '@heroicons/vue/24/outline'
import { useBannerStore } from '@/stores/banners'
import { useNotificationStore } from '@/stores/notifications'

const bannerStore = useBannerStore()
const notificationStore = useNotificationStore()

const searchQuery = ref('')
const statusFilter = ref('')
const viewMode = ref<'grid' | 'list'>('grid')

const filteredBanners = computed(() => {
  let banners = bannerStore.banners
  
  // Apply search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    banners = banners.filter(banner => 
      banner.name.toLowerCase().includes(query) ||
      banner.description?.toLowerCase().includes(query)
    )
  }
  
  // Apply status filter
  if (statusFilter.value) {
    banners = banners.filter(banner => banner.status === statusFilter.value)
  }
  
  return banners
})

const formatDate = (dateString: string) => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMinutes = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMinutes < 1) return 'Just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  
  return date.toLocaleDateString()
}

const downloadBanner = async (id: string, format: 'svg' | 'html' | 'png' | 'figma') => {
  try {
    await bannerStore.downloadBanner(id, format)
  } catch (error) {
    console.error('Download failed:', error)
  }
}

const retryBanner = async (id: string) => {
  try {
    await bannerStore.retryBanner(id)
  } catch (error) {
    console.error('Retry failed:', error)
  }
}

const deleteBanner = async (id: string) => {
  if (confirm('Are you sure you want to delete this banner?')) {
    try {
      await bannerStore.deleteBanner(id)
    } catch (error) {
      console.error('Delete failed:', error)
    }
  }
}

onMounted(() => {
  bannerStore.fetchBanners()
})
</script>
