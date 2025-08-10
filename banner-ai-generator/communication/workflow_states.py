from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import asyncio

class ExecutionResult:
    """Result of workflow step execution"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now()

class WorkflowExecutor:
    """Execute workflow steps in sequence"""
    
    def __init__(self, memory_manager: BasicMemoryManager, session_manager: BasicSessionManager):
        self.memory_manager = memory_manager
        self.session_manager = session_manager
        self._workflow_steps: Dict[WorkflowState, Callable] = {}
    
    def register_step(self, state: WorkflowState, executor: Callable) -> None:
        """Register executor function for workflow step"""
        self._workflow_steps[state] = executor
    
    async def execute_workflow(self, session_id: str) -> ExecutionResult:
        """Execute complete workflow for session"""
        try:
            # Get campaign data
            campaign_data = self.memory_manager.retrieve(session_id, MemoryKey.CAMPAIGN_DATA)
            if not campaign_data:
                return ExecutionResult(False, error="Campaign data not found")
            
            # Define workflow sequence
            workflow_sequence = [
                WorkflowState.STRATEGIST_WORKING,
                WorkflowState.BACKGROUND_WORKING,
                WorkflowState.FOREGROUND_WORKING,
                WorkflowState.DEVELOPER_WORKING
            ]
            
            results = {}
            
            for state in workflow_sequence:
                print(f"Executing workflow step: {state.value}")
                
                # Update workflow state
                self.session_manager.update_workflow_state(session_id, state)
                
                # Execute step
                if state in self._workflow_steps:
                    step_executor = self._workflow_steps[state]
                    result = await self._execute_step(step_executor, session_id)
                    
                    if not result.success:
                        # Mark workflow as failed
                        self.session_manager.update_workflow_state(session_id, WorkflowState.FAILED)
                        return ExecutionResult(False, error=f"Failed at step {state.value}: {result.error}")
                    
                    results[state.value] = result.data
                    
                    # Update completion state
                    completion_state = self._get_completion_state(state)
                    if completion_state:
                        self.session_manager.update_workflow_state(session_id, completion_state)
                else:
                    return ExecutionResult(False, error=f"No executor registered for state {state.value}")
            
            # Mark as completed
            self.session_manager.update_workflow_state(session_id, WorkflowState.COMPLETED)
            
            return ExecutionResult(True, data=results)
            
        except Exception as e:
            print(f"Workflow execution error: {e}")
            print(traceback.format_exc())
            self.session_manager.update_workflow_state(session_id, WorkflowState.FAILED)
            return ExecutionResult(False, error=str(e))
    
    async def _execute_step(self, executor: Callable, session_id: str) -> ExecutionResult:
        """Execute individual workflow step"""
        try:
            # Check if executor is async
            if asyncio.iscoroutinefunction(executor):
                result = await executor(session_id)
            else:
                result = executor(session_id)
            
            if isinstance(result, ExecutionResult):
                return result
            else:
                return ExecutionResult(True, data=result)
                
        except Exception as e:
            return ExecutionResult(False, error=str(e))
    
    def _get_completion_state(self, working_state: WorkflowState) -> Optional[WorkflowState]:
        """Get completion state for working state"""
        completion_map = {
            WorkflowState.STRATEGIST_WORKING: WorkflowState.STRATEGIST_COMPLETE,
            WorkflowState.BACKGROUND_WORKING: WorkflowState.BACKGROUND_COMPLETE,
            WorkflowState.FOREGROUND_WORKING: WorkflowState.FOREGROUND_COMPLETE,
            WorkflowState.DEVELOPER_WORKING: WorkflowState.DEVELOPER_COMPLETE
        }
        return completion_map.get(working_state)
