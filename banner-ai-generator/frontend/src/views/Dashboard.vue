<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p class="mt-1 text-sm text-gray-500">
          Overview of your AI banner generation system
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

    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <!-- Total Banners -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <PhotoIcon class="h-8 w-8 text-primary-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Total Banners
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ bannerStore.bannerStats.total }}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Processing -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <ClockIcon class="h-8 w-8 text-yellow-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Processing
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ bannerStore.bannerStats.processing }}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Completed -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <CheckCircleIcon class="h-8 w-8 text-green-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Completed
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ bannerStore.bannerStats.completed }}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Success Rate -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <ChartBarIcon class="h-8 w-8 text-blue-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Success Rate
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ successRate }}%
              </dd>
            </dl>
          </div>
        </div>
      </div>
    </div>

    <!-- System Health -->
    <div class="card">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-medium text-gray-900">System Health</h2>
        <div class="flex items-center space-x-2">
          <div 
            class="w-3 h-3 rounded-full"
            :class="{
              'bg-green-400': systemStore.systemHealth === 'healthy',
              'bg-yellow-400': systemStore.systemHealth === 'degraded',
              'bg-red-400': systemStore.systemHealth === 'critical',
              'bg-gray-400': systemStore.systemHealth === 'unknown'
            }"
          ></div>
          <span class="text-sm font-medium capitalize">
            {{ systemStore.systemHealth }}
          </span>
        </div>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- Agents -->
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-gray-900">
            {{ systemStore.agentStats.healthy }}/{{ systemStore.agentStats.total }}
          </div>
          <div class="text-sm text-gray-500">Agents Online</div>
        </div>
        
        <!-- Workflows -->
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-gray-900">
            {{ systemStore.workflowStats.active }}
          </div>
          <div class="text-sm text-gray-500">Active Workflows</div>
        </div>
        
        <!-- Throughput -->
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl font-bold text-gray-900">
            {{ systemStore.metrics?.system_throughput?.toFixed(1) || '0' }}
          </div>
          <div class="text-sm text-gray-500">Workflows/Hour</div>
        </div>
      </div>
    </div>

    <!-- Recent Banners and Active Workflows -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Recent Banners -->
      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-medium text-gray-900">Recent Banners</h2>
          <router-link 
            to="/campaigns" 
            class="text-sm text-primary-600 hover:text-primary-500"
          >
            View all →
          </router-link>
        </div>
        
        <div class="space-y-3">
          <div 
            v-if="bannerStore.loading"
            class="flex justify-center py-8"
          >
            <div class="spinner w-8 h-8"></div>
          </div>
          
          <div 
            v-else-if="bannerStore.recentBanners.length === 0"
            class="text-center py-8 text-gray-500"
          >
            No banners created yet
          </div>
          
          <div 
            v-else
            v-for="banner in bannerStore.recentBanners.slice(0, 5)" 
            :key="banner.id"
            class="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
            @click="$router.push(`/banners/${banner.id}`)"
          >
            <div class="flex items-center space-x-3">
              <div class="flex-shrink-0">
                <PhotoIcon class="h-8 w-8 text-gray-400" />
              </div>
              <div>
                <div class="text-sm font-medium text-gray-900">
                  {{ banner.name }}
                </div>
                <div class="text-xs text-gray-500">
                  {{ formatDate(banner.updated_at) }}
                </div>
              </div>
            </div>
            <div class="flex items-center space-x-2">
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
            </div>
          </div>
        </div>
      </div>

      <!-- Active Workflows -->
      <div class="card">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-medium text-gray-900">Active Workflows</h2>
          <router-link 
            to="/monitor" 
            class="text-sm text-primary-600 hover:text-primary-500"
          >
            View monitor →
          </router-link>
        </div>
        
        <div class="space-y-3">
          <div 
            v-if="systemStore.loading"
            class="flex justify-center py-8"
          >
            <div class="spinner w-8 h-8"></div>
          </div>
          
          <div 
            v-else-if="systemStore.activeWorkflows.length === 0"
            class="text-center py-8 text-gray-500"
          >
            No active workflows
          </div>
          
          <div 
            v-else
            v-for="workflow in systemStore.activeWorkflows.slice(0, 5)" 
            :key="workflow.workflow_id"
            class="p-3 bg-gray-50 rounded-lg"
          >
            <div class="flex items-center justify-between mb-2">
              <div class="text-sm font-medium text-gray-900">
                {{ workflow.workflow_type }}
              </div>
              <div class="text-xs text-gray-500">
                {{ workflow.progress_percentage }}%
              </div>
            </div>
            
            <div class="progress-bar">
              <div 
                class="progress-fill"
                :style="{ width: `${workflow.progress_percentage}%` }"
              ></div>
            </div>
            
            <div class="flex justify-between items-center mt-2">
              <div class="text-xs text-gray-500">
                {{ workflow.current_step?.step_id || 'Starting...' }}
              </div>
              <div class="text-xs text-gray-500">
                {{ workflow.completed_steps }}/{{ workflow.total_steps }} steps
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { 
  PlusIcon,
  PhotoIcon,
  ClockIcon, 
  CheckCircleIcon,
  ChartBarIcon
} from '@heroicons/vue/24/outline'
import { useBannerStore } from '@/stores/banners'
import { useSystemStore } from '@/stores/system'

const bannerStore = useBannerStore()
const systemStore = useSystemStore()

const successRate = computed(() => {
  const stats = bannerStore.bannerStats
  const total = stats.completed + stats.failed
  if (total === 0) return 0
  return Math.round((stats.completed / total) * 100)
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

onMounted(async () => {
  // Load initial data
  await Promise.all([
    bannerStore.fetchBanners(),
    systemStore.fetchAllSystemData()
  ])
  
  // Start periodic refresh for system data
  systemStore.startPeriodicRefresh(30000) // 30 seconds
})

onUnmounted(() => {
  systemStore.stopPeriodicRefresh()
})
</script>
