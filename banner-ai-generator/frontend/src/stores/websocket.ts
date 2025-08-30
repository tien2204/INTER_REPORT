// frontend/src/stores/websocket.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { io, Socket } from 'socket.io-client'
import { useNotificationStore } from './notifications'

export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

export const useWebSocketStore = defineStore('websocket', () => {
  const socket = ref<Socket | null>(null)
  const connected = ref(false)
  const reconnecting = ref(false)
  const messages = ref<WebSocketMessage[]>([])
  const maxMessages = ref(1000)

  const notificationStore = useNotificationStore()

  const isConnected = computed(() => connected.value)
  const isReconnecting = computed(() => reconnecting.value)

  const connect = () => {
    if (socket.value?.connected) {
      return
    }

    try {
      // ✅ SỬA LẠI: Sử dụng proxy thay vì hardcode IP
      socket.value = io({
        path: '/socket.io/',
        transports: ['websocket', 'polling'], // Thêm polling làm fallback
        timeout: 10000,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        upgrade: true, // Cho phép upgrade từ polling lên websocket
        rememberUpgrade: true, // Nhớ upgrade cho lần kết nối tiếp theo
        forceNew: false // Tái sử dụng kết nối hiện có nếu có thể
      })

      socket.value.on('connect', () => {
        connected.value = true
        reconnecting.value = false
        console.log('✅ WebSocket connected:', socket.value?.id)
        notificationStore.success('Connected', 'Real-time updates enabled')
      })

      socket.value.on('disconnect', (reason) => {
        connected.value = false
        console.log('❌ WebSocket disconnected:', reason)

        // Tự động reconnect cho một số lý do disconnect
        if (reason === 'io server disconnect' || reason === 'transport close') {
          console.log('🔄 Attempting to reconnect...')
          setTimeout(() => {
            socket.value?.connect()
          }, 2000)
        }
      })

      socket.value.on('reconnect', (attemptNumber) => {
        connected.value = true
        reconnecting.value = false
        console.log('✅ WebSocket reconnected after', attemptNumber, 'attempts')
        notificationStore.success('Reconnected', 'Real-time updates restored')
      })

      socket.value.on('reconnect_attempt', (attemptNumber) => {
        reconnecting.value = true
        console.log('🔄 WebSocket attempting to reconnect (#' + attemptNumber + ')')
      })

      socket.value.on('reconnect_error', (error) => {
        console.error('❌ WebSocket reconnection error:', error)
      })

      socket.value.on('connect_error', (error) => {
        console.error('❌ WebSocket connection error:', error)
        connected.value = false
        
        // Chỉ hiển thị warning sau một vài lần thử
        if (error.message.includes('timeout') || error.message.includes('websocket')) {
          console.log('🔄 Will retry with polling transport...')
        } else {
          notificationStore.warning(
            'Connection Issue',
            'Unable to connect for real-time updates'
          )
        }
      })

      // Listen for specific events
      socket.value.on('workflow_progress', handleWorkflowProgress)
      socket.value.on('design_completed', handleDesignCompleted)
      socket.value.on('design_failed', handleDesignFailed)
      socket.value.on('system_alert', handleSystemAlert)

    } catch (error) {
      console.error('❌ Failed to initialize WebSocket:', error)
      notificationStore.error('Connection Error', 'Failed to initialize real-time connection')
    }
  }

  const disconnect = () => {
    if (socket.value) {
      socket.value.disconnect()
      socket.value = null
      connected.value = false
      reconnecting.value = false
    }
  }

  const emit = (event: string, data: any) => {
    if (socket.value?.connected) {
      socket.value.emit(event, data)
    } else {
      console.warn('⚠️ WebSocket not connected, cannot emit event:', event)
    }
  }

  const addMessage = (message: WebSocketMessage) => {
    messages.value.unshift(message)
    if (messages.value.length > maxMessages.value) {
      messages.value = messages.value.slice(0, maxMessages.value)
    }
  }

  // Event handlers
  const handleWorkflowProgress = (data: any) => {
    addMessage({
      type: 'workflow_progress',
      data,
      timestamp: new Date().toISOString()
    })

    if (data.progress_percentage >= 100) {
      notificationStore.success(
        'Workflow Complete',
        `${data.workflow_type} finished successfully`
      )
    }
  }

  const handleDesignCompleted = (data: any) => {
    addMessage({
      type: 'design_completed',
      data,
      timestamp: new Date().toISOString()
    })

    notificationStore.success(
      'Design Ready',
      `Banner design ${data.design_id} has been completed`
    )
  }

  const handleDesignFailed = (data: any) => {
    addMessage({
      type: 'design_failed',
      data,
      timestamp: new Date().toISOString()
    })

    notificationStore.error(
      'Design Failed',
      `Banner design ${data.design_id} failed: ${data.error}`
    )
  }

  const handleSystemAlert = (data: any) => {
    addMessage({
      type: 'system_alert',
      data,
      timestamp: new Date().toISOString()
    })

    const alertType = data.severity === 'critical' ? 'error' :
                     data.severity === 'warning' ? 'warning' : 'info'

    notificationStore[alertType]('System Alert', data.message)
  }

  return {
    socket,
    connected,
    reconnecting,
    messages,
    isConnected,
    isReconnecting,
    connect,
    disconnect,
    emit
  }
})
