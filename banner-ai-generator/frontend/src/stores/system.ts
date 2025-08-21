import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'
import { useNotificationStore } from './notifications'

export interface SystemStatus {
  overall_health: 'healthy' | 'degraded' | 'critical'
  components: {
    message_queue: string
    event_dispatcher: string
    agent_coordinator: string
  }
  issues?: string[]
  timestamp: string
}

export interface AgentStatus {
  agent_id: string
  agent_name: string
  status: 'idle' | 'busy' | 'error' | 'offline'
  capabilities: string[]
  last_heartbeat: string
  current_workflow?: string
  current_step?: string
  performance: {
    total_tasks: number
    success_rate: number
    avg_execution_time: number
    last_task_time?: string
  }
}

export interface WorkflowStatus {
  workflow_id: string
  workflow_type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress_percentage: number
  total_steps: number
  completed_steps: number
  current_step?: {
    step_id: string
    agent_id: string
    action: string
    status: string
  }
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface SystemMetrics {
  active_agents: number
  active_workflows: number
  total_workflows: number
  workflow_success_rate: number
  avg_workflow_duration: number
  system_throughput: number
  message_throughput: number
  event_throughput: number
}

export const useSystemStore = defineStore('system', () => {
  const systemStatus = ref<SystemStatus | null>(null)
  const agents = ref<AgentStatus[]>([])
  const workflows = ref<WorkflowStatus[]>([])
  const metrics = ref<SystemMetrics | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const notificationStore = useNotificationStore()

  // Computed
  const healthyAgents = computed(() => 
    agents.value.filter(agent => agent.status !== 'offline' && agent.status !== 'error')
  )

  const busyAgents = computed(() => 
    agents.value.filter(agent => agent.status === 'busy')
  )

  const errorAgents = computed(() => 
    agents.value.filter(agent => agent.status === 'error')
  )

  const offlineAgents = computed(() => 
    agents.value.filter(agent => agent.status === 'offline')
  )

  const activeWorkflows = computed(() => 
    workflows.value.filter(workflow => 
      workflow.status === 'running' || workflow.status === 'pending'
    )
  )

  const completedWorkflows = computed(() => 
    workflows.value.filter(workflow => workflow.status === 'completed')
  )

  const failedWorkflows = computed(() => 
    workflows.value.filter(workflow => workflow.status === 'failed')
  )

  const systemHealth = computed(() => {
    if (!systemStatus.value) return 'unknown'
    return systemStatus.value.overall_health
  })

  const agentStats = computed(() => ({
    total: agents.value.length,
    healthy: healthyAgents.value.length,
    busy: busyAgents.value.length,
    error: errorAgents.value.length,
    offline: offlineAgents.value.length
  }))

  const workflowStats = computed(() => ({
    total: workflows.value.length,
    active: activeWorkflows.value.length,
    completed: completedWorkflows.value.length,
    failed: failedWorkflows.value.length,
    cancelled: workflows.value.filter(w => w.status === 'cancelled').length
  }))

  // Actions
  const fetchSystemStatus = async () => {
    try {
      loading.value = true
      error.value = null
      
      const response = await apiService.get('/system/status')
      systemStatus.value = response.data
      
      // Check for health issues
      if (systemStatus.value.overall_health === 'critical') {
        notificationStore.error(
          'System Critical',
          'System health is critical. Check system monitor for details.'
        )
      } else if (systemStatus.value.overall_health === 'degraded') {
        notificationStore.warning(
          'System Degraded',
          'System performance is degraded.'
        )
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch system status'
      notificationStore.error('Error', 'Failed to load system status')
    } finally {
      loading.value = false
    }
  }

  const fetchAgents = async () => {
    try {
      const response = await apiService.get('/agents')
      agents.value = response.data.agents || []
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch agents'
      notificationStore.error('Error', 'Failed to load agent status')
    }
  }

  const fetchWorkflows = async () => {
    try {
      const response = await apiService.get('/workflows')
      workflows.value = response.data.workflows || []
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch workflows'
      notificationStore.error('Error', 'Failed to load workflow status')
    }
  }

  const fetchMetrics = async () => {
    try {
      const response = await apiService.get('/system/metrics')
      metrics.value = response.data
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch metrics'
      notificationStore.error('Error', 'Failed to load system metrics')
    }
  }

  const fetchAllSystemData = async () => {
    await Promise.all([
      fetchSystemStatus(),
      fetchAgents(),
      fetchWorkflows(),
      fetchMetrics()
    ])
  }

  const restartAgent = async (agentId: string) => {
    try {
      await apiService.post(`/agents/${agentId}/restart`)
      
      // Update agent status
      const agent = agents.value.find(a => a.agent_id === agentId)
      if (agent) {
        agent.status = 'idle'
      }
      
      notificationStore.success('Agent Restarted', `Agent ${agentId} has been restarted`)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to restart agent'
      notificationStore.error('Error', `Failed to restart agent ${agentId}`)
      throw err
    }
  }

  const cancelWorkflow = async (workflowId: string) => {
    try {
      await apiService.post(`/workflows/${workflowId}/cancel`)
      
      // Update workflow status
      const workflow = workflows.value.find(w => w.workflow_id === workflowId)
      if (workflow) {
        workflow.status = 'cancelled'
      }
      
      notificationStore.success('Workflow Cancelled', `Workflow ${workflowId} has been cancelled`)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to cancel workflow'
      notificationStore.error('Error', `Failed to cancel workflow ${workflowId}`)
      throw err
    }
  }

  const updateAgentStatus = (agentId: string, status: Partial<AgentStatus>) => {
    const index = agents.value.findIndex(a => a.agent_id === agentId)
    if (index !== -1) {
      agents.value[index] = { ...agents.value[index], ...status }
    }
  }

  const updateWorkflowStatus = (workflowId: string, status: Partial<WorkflowStatus>) => {
    const index = workflows.value.findIndex(w => w.workflow_id === workflowId)
    if (index !== -1) {
      workflows.value[index] = { ...workflows.value[index], ...status }
    }
  }

  const clearError = () => {
    error.value = null
  }

  // Periodic data refresh
  let refreshInterval: number | null = null

  const startPeriodicRefresh = (intervalMs: number = 30000) => {
    if (refreshInterval) {
      clearInterval(refreshInterval)
    }
    
    refreshInterval = setInterval(() => {
      fetchAllSystemData()
    }, intervalMs)
  }

  const stopPeriodicRefresh = () => {
    if (refreshInterval) {
      clearInterval(refreshInterval)
      refreshInterval = null
    }
  }

  return {
    // State
    systemStatus,
    agents,
    workflows,
    metrics,
    loading,
    error,
    
    // Computed
    healthyAgents,
    busyAgents,
    errorAgents,
    offlineAgents,
    activeWorkflows,
    completedWorkflows,
    failedWorkflows,
    systemHealth,
    agentStats,
    workflowStats,
    
    // Actions
    fetchSystemStatus,
    fetchAgents,
    fetchWorkflows,
    fetchMetrics,
    fetchAllSystemData,
    restartAgent,
    cancelWorkflow,
    updateAgentStatus,
    updateWorkflowStatus,
    clearError,
    startPeriodicRefresh,
    stopPeriodicRefresh
  }
})
