<template>
  <div id="app" class="min-h-screen bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <router-link 
              to="/" 
              class="flex items-center space-x-2 text-xl font-bold text-gray-900 hover:text-primary-600 transition-colors"
            >
              <svg class="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span>AI Banner Generator</span>
            </router-link>
          </div>
          
          <!-- Navigation Links -->
          <div class="flex items-center space-x-8">
            <router-link 
              to="/" 
              class="text-gray-700 hover:text-primary-600 px-3 py-2 text-sm font-medium transition-colors"
              :class="{ 'text-primary-600 border-b-2 border-primary-600': $route.path === '/' }"
            >
              Dashboard
            </router-link>
            <router-link 
              to="/create" 
              class="text-gray-700 hover:text-primary-600 px-3 py-2 text-sm font-medium transition-colors"
              :class="{ 'text-primary-600 border-b-2 border-primary-600': $route.path === '/create' }"
            >
              Create Banner
            </router-link>
            <router-link 
              to="/campaigns" 
              class="text-gray-700 hover:text-primary-600 px-3 py-2 text-sm font-medium transition-colors"
              :class="{ 'text-primary-600 border-b-2 border-primary-600': $route.path === '/campaigns' }"
            >
              Campaigns
            </router-link>
            <router-link 
              to="/monitor" 
              class="text-gray-700 hover:text-primary-600 px-3 py-2 text-sm font-medium transition-colors"
              :class="{ 'text-primary-600 border-b-2 border-primary-600': $route.path === '/monitor' }"
            >
              System Monitor
            </router-link>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <router-view />
    </main>

    <!-- Global Notifications -->
    <div 
      v-if="notificationStore.notifications.length > 0"
      class="fixed top-4 right-4 z-50 space-y-2"
    >
      <div
        v-for="notification in notificationStore.notifications"
        :key="notification.id"
        class="max-w-sm w-full bg-white shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden"
        :class="{
          'border-l-4 border-green-400': notification.type === 'success',
          'border-l-4 border-yellow-400': notification.type === 'warning',
          'border-l-4 border-red-400': notification.type === 'error',
          'border-l-4 border-blue-400': notification.type === 'info'
        }"
      >
        <div class="p-4">
          <div class="flex items-start">
            <div class="flex-shrink-0">
              <CheckCircleIcon v-if="notification.type === 'success'" class="h-6 w-6 text-green-400" />
              <ExclamationTriangleIcon v-else-if="notification.type === 'warning'" class="h-6 w-6 text-yellow-400" />
              <XCircleIcon v-else-if="notification.type === 'error'" class="h-6 w-6 text-red-400" />
              <InformationCircleIcon v-else class="h-6 w-6 text-blue-400" />
            </div>
            <div class="ml-3 w-0 flex-1 pt-0.5">
              <p class="text-sm font-medium text-gray-900">{{ notification.title }}</p>
              <p v-if="notification.message" class="mt-1 text-sm text-gray-500">{{ notification.message }}</p>
            </div>
            <div class="ml-4 flex-shrink-0 flex">
              <button 
                @click="notificationStore.removeNotification(notification.id)"
                class="bg-white rounded-md inline-flex text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                <XMarkIcon class="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  XCircleIcon, 
  InformationCircleIcon,
  XMarkIcon 
} from '@heroicons/vue/24/outline'
import { useNotificationStore } from '@/stores/notifications'
import { useWebSocketStore } from '@/stores/websocket'

const notificationStore = useNotificationStore()
const websocketStore = useWebSocketStore()

onMounted(() => {
  // Initialize WebSocket connection
  websocketStore.connect()
})
</script>
