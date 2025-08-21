<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">System Monitor</h1>
        <p class="mt-1 text-sm text-gray-500">
          Real-time monitoring of AI agents and system performance
        </p>
      </div>
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

    <!-- System Overview -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <!-- Agents -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <CpuChipIcon class="h-8 w-8 text-blue-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Active Agents
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ systemStore.agentStats.healthy }}/{{ systemStore.agentStats.total }}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Workflows -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <Cog6ToothIcon class="h-8 w-8 text-green-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Active Workflows
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ systemStore.workflowStats.active }}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Success Rate -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <ChartBarIcon class="h-8 w-8 text-purple-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Success Rate
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ systemStore.metrics?.workflow_success_rate?.toFixed(1) || '0' }}%
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Throughput -->
      <div class="card">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <BoltIcon class="h-8 w-8 text-yellow-600" />
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">
                Throughput
              </dt>
              <dd class="text-lg font-medium text-gray-900">
                {{ systemStore.metrics?.system_throughput?.toFixed(1) || '0' }}/h
              </dd>
            </dl>
          </div>
        </div>
      </div>
    </div>

    <!-- Agents Status -->
    <div class="card">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-lg font-medium text-gray-900">AI Agents</h2>
        <button
          @click="refreshAgents"
          :disabled="systemStore.loading"
          class="btn btn-secondary btn-sm"
        >
          <ArrowPathIcon class="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>
      
      <div v-if="systemStore.loading" class="flex justify-center py-8">
        <div class="spinner w-8 h-8"></div>
      </div>
      
      <div v-else-if="systemStore.agents.length === 0" class="text-center py-8 text-gray-500">
        No agents registered
      </div>
      
      <div v-else class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Agent
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Current Task
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Performance
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Last Heartbeat
              </th>
              <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="agent in systemStore.agents" :key="agent.agent_id">
              <td class="px-6 py-4 whitespace-nowrap">
                <div>
                  <div class="text-sm font-medium text-gray-900">
                    {{ agent.agent_name }}
                  </div>
                  <div class="text-sm text-gray-500">
                    {{ agent.agent_id }}
                  </div>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span 
                  class="status-badge"
                  :class="{
                    'status-success': agent.status === 'idle',
                    'status-warning': agent.status === 'busy',
                    'status-error': agent.status === 'error',
                    'bg-gray-100 text-gray-800': agent.status === 'offline'
                  }"
                >
                  {{ agent.status }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div v-if="agent.current_step">
                  <div class="font-medium">{{ agent.current_step }}</div>
                  <div class="text-xs">{{ agent.current_workflow }}</div>
                </div>
                <span v-else>-</span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div>
                  <div>{{ agent.performance.total_tasks }} tasks</div>
                  <div>{{ agent.performance.success_rate.toFixed(1) }}% success</div>
                  <div>{{ agent.performance.avg_execution_time.toFixed(1) }}s avg</div>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {{ formatDate(agent.last_heartbeat) }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <button
                  v-if="agent.status === 'error' || agent.status === 'offline'"
                  @click="restartAgent(agent.agent_id)"
                  class="text-primary-600 hover:text-primary-900"
                >
                  Restart
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Active Workflows -->
    <div class="card">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-lg font-medium text-gray-900">Active Workflows</h2>
        <button
          @click="refreshWorkflows"
          :disabled="systemStore.loading"
          class="btn btn-secondary btn-sm"
        >
          <ArrowPathIcon class="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>
      
      <div v-if="systemStore.loading" class="flex justify-center py-8">
        <div class="spinner w-8 h-8"></div>
      </div>
      
      <div v-else-if="systemStore.activeWorkflows.length === 0" class="text-center py-8 text-gray-500">
        No active workflows
      </div>
      
      <div v-else class="space-y-4">
        <div 
          v-for="workflow in systemStore.activeWorkflows" 
          :key="workflow.workflow_id"
          class="border border-gray-200 rounded-lg p-4"
        >
          <div class="flex items-center justify-between mb-3">
            <div>
              <h3 class="text-sm font-medium text-gray-900">
                {{ workflow.workflow_type }}
              </h3>
              <p class="text-xs text-gray-500">{{ workflow.workflow_id }}</p>
            </div>
            <div class="flex items-center space-x-4">
              <span class="text-sm text-gray-500">
                {{ workflow.progress_percentage }}%
              </span>
              <button
                @click="cancelWorkflow(workflow.workflow_id)"
                class="text-red-600 hover:text-red-900 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
          
          <div class="progress-bar mb-3">
            <div 
              class="progress-fill"
              :style="{ width: `${workflow.progress_percentage}%` }"
            ></div>
          </div>
          
          <div class="flex justify-between items-center text-xs text-gray-500">
            <div>
              Current: {{ workflow.current_step?.step_id || 'Starting...' }}
              <span v-if="workflow.current_step?.agent_id">
                ({{ workflow.current_step.agent_id }})
              </span>
            </div>
            <div>
              {{ workflow.completed_steps }}/{{ workflow.total_steps }} steps
            </div>
          </div>
          
          <div class="mt-2 text-xs text-gray-500">
            Started: {{ formatDate(workflow.started_at || workflow.created_at) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Real-time Messages -->
    <div class="card">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-lg font-medium text-gray-900">Real-time Events</h2>
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full"
            :class="websocketStore.isConnected ? 'bg-green-400' : 'bg-red-400'"
          ></div>
          <span class="text-sm text-gray-500">
            {{ websocketStore.isConnected ? 'Connected' : 'Disconnected' }}
          </span>
        </div>
      </div>
      
      <div class="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
        <div 
          v-if="websocketStore.messages.length === 0"
          class="text-center py-8 text-gray-500"
        >
          No recent events
        </div>
        
        <div 
          v-else
          v-for="message in websocketStore.messages.slice(0, 50)" 
          :key="message.timestamp"
          class="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg"
        >
          <div class="flex-shrink-0 mt-0.5">
            <div 
              class="w-2 h-2 rounded-full"
              :class="{
                'bg-green-400': message.type === 'workflow_progress' || message.type === 'design_completed',
                'bg-red-400': message.type === 'design_failed' || message.type === 'system_alert',
                'bg-blue-400': true
              }"
            ></div>
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <p class="text-sm font-medium text-gray-900">
                {{ formatEventType(message.type) }}
              </p>
              <p class="text-xs text-gray-500">
                {{ formatDate(message.timestamp) }}
              </p>
            </div>
            <p class="text-sm text-gray-600">
              {{ formatEventData(message.data) }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import { 
  CpuChipIcon,
  Cog6ToothIcon,
  ChartBarIcon,
  BoltIcon,
  ArrowPathIcon
} from '@heroicons/vue/24/outline'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'

const systemStore = useSystemStore()
const websocketStore = useWebSocketStore()

const formatDate = (dateString: string) => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMinutes = Math.floor(diffMs / 60000)

  if (diffMinutes < 1) return 'Just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  
  return date.toLocaleTimeString()
}

const formatEventType = (type: string) => {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

const formatEventData = (data: any) => {
  if (data.workflow_id) {
    return `Workflow: ${data.workflow_id}`
  }
  if (data.agent_id) {
    return `Agent: ${data.agent_id}`
  }
  if (data.message) {
    return data.message
  }
  return JSON.stringify(data).slice(0, 100) + '...'
}

const refreshAgents = () => {
  systemStore.fetchAgents()
}

const refreshWorkflows = () => {
  systemStore.fetchWorkflows()
}

const restartAgent = async (agentId: string) => {
  try {
    await systemStore.restartAgent(agentId)
  } catch (error) {
    console.error('Failed to restart agent:', error)
  }
}

const cancelWorkflow = async (workflowId: string) => {
  if (confirm('Are you sure you want to cancel this workflow?')) {
    try {
      await systemStore.cancelWorkflow(workflowId)
    } catch (error) {
      console.error('Failed to cancel workflow:', error)
    }
  }
}

onMounted(() => {
  // Load initial data
  systemStore.fetchAllSystemData()
  
  // Start periodic refresh
  systemStore.startPeriodicRefresh(10000) // 10 seconds for monitor page
})

onUnmounted(() => {
  systemStore.stopPeriodicRefresh()
})
</script>
