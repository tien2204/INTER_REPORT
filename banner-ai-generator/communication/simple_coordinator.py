class SimpleWorkflowCoordinator:
    """Simple coordinator for linear agent workflow"""
    
    def __init__(self, memory_manager: BasicMemoryManager, session_manager: BasicSessionManager):
        self.memory_manager = memory_manager
        self.session_manager = session_manager
        self.messaging = BasicMessaging()
        self.workflow_executor = WorkflowExecutor(memory_manager, session_manager)
        
        # Register agents
        self._agents: Dict[str, Any] = {}
        self._agent_order = ['strategist', 'background_designer', 'foreground_designer', 'developer']
    
    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register agent with coordinator"""
        self._agents[agent_name] = agent_instance
        
        # Register message handlers
        if hasattr(agent_instance, 'handle_message'):
            self.messaging.register_handler(agent_name, 'execute', agent_instance.handle_message)
    
    def register_workflow_step(self, state: WorkflowState, executor: Callable) -> None:
        """Register executor for workflow step"""
        self.workflow_executor.register_step(state, executor)
    
    async def execute_campaign(self, session_id: str) -> ExecutionResult:
        """Execute complete campaign workflow"""
        print(f"Starting campaign execution for session: {session_id}")
        
        # Verify session is active
        if not self.session_manager.is_session_active(session_id):
            return ExecutionResult(False, error="Session is not active")
        
        # Execute workflow
        result = await self.workflow_executor.execute_workflow(session_id)
        
        if result.success:
            print(f"Campaign execution completed successfully for session: {session_id}")
            self.session_manager.close_session(session_id)
        else:
            print(f"Campaign execution failed for session: {session_id}. Error: {result.error}")
        
        return result
    
    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get current workflow status"""
        session_info = self.session_manager.get_session_info(session_id)
        campaign_data = self.memory_manager.retrieve(session_id, MemoryKey.CAMPAIGN_DATA)
        
        status = {
            'session_id': session_id,
            'session_active': self.session_manager.is_session_active(session_id),
            'workflow_state': session_info.get('workflow_state', WorkflowState.INITIALIZED).value if session_info else 'unknown',
            'current_iteration': campaign_data.current_iteration if campaign_data else 0,
            'max_iterations': campaign_data.max_iterations if campaign_data else 5,
            'agent_states': {}
        }
        
        if campaign_data:
            for agent_type, agent_state in campaign_data.agent_states.items():
                status['agent_states'][agent_type.value] = {
                    'status': agent_state.status,
                    'progress': agent_state.progress,
                    'current_task': agent_state.current_task,
                    'last_activity': agent_state.last_activity.isoformat() if agent_state.last_activity else None,
                    'error_message': agent_state.error_message
                }
        
        return status
    
    def emergency_stop(self, session_id: str) -> bool:
        """Emergency stop workflow execution"""
        try:
            # Update workflow state to failed
            self.session_manager.update_workflow_state(session_id, WorkflowState.FAILED)
            
            # Update all agent states to stopped
            campaign_data = self.memory_manager.retrieve(session_id, MemoryKey.CAMPAIGN_DATA)
            if campaign_data:
                for agent_state in campaign_data.agent_states.values():
                    agent_state.status = "stopped"
                    agent_state.error_message = "Emergency stop requested"
                
                self.memory_manager.store(session_id, MemoryKey.CAMPAIGN_DATA, campaign_data)
            
            return True
        except Exception as e:
            print(f"Error during emergency stop: {e}")
            return False
