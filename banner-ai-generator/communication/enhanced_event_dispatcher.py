"""
Enhanced Event Dispatcher

Advanced event-driven communication system with filtering, routing,
metrics, and integration with Agent Coordinator.
"""

import asyncio
from typing import Dict, Any, List, Callable, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class EventSubscription:
    """Event subscription configuration"""
    handler: Callable
    event_filter: Optional[Callable] = None
    priority: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class EventMetrics:
    """Event processing metrics"""
    total_published: int = 0
    total_handled: int = 0
    total_errors: int = 0
    avg_processing_time: float = 0.0
    last_event_time: Optional[datetime] = None


class EnhancedEventDispatcher:
    """
    Enhanced event dispatcher for the banner generation system
    
    Capabilities:
    - Event subscription and publishing with priorities
    - Advanced event filtering and routing
    - Asynchronous event handling with retries
    - Event history, metrics, and monitoring
    - Event replay and debugging support
    - Integration with Agent Coordinator
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Event subscriptions with priorities
        self.subscribers: Dict[str, List[EventSubscription]] = {}
        
        # Event history and metrics
        self.event_history: List[Dict[str, Any]] = []
        self.event_metrics: Dict[str, EventMetrics] = {}
        self.max_history_size = config.get("max_history_size", 1000)
        
        # Event filtering and routing
        self.global_filters: List[Callable] = []
        self.event_routes: Dict[str, List[str]] = {}  # Event type -> target event types
        
        # Processing settings
        self.max_concurrent_events = config.get("max_concurrent_events", 100)
        self.event_timeout = config.get("event_timeout", 30)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Integration
        self.agent_coordinator = None
        
        # Processing queue and semaphore
        self._processing_queue = asyncio.Queue()
        self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_events)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced Event Dispatcher initialized")
    
    async def start(self):
        """Start the event dispatcher"""
        try:
            logger.info("Starting Enhanced Event Dispatcher")
            
            self._running = True
            self._processor_task = asyncio.create_task(self._process_events())
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            logger.info("Enhanced Event Dispatcher started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Enhanced Event Dispatcher: {e}")
            raise
    
    async def stop(self):
        """Stop the event dispatcher"""
        try:
            logger.info("Stopping Enhanced Event Dispatcher")
            
            self._running = False
            
            if self._processor_task:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Enhanced Event Dispatcher stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Enhanced Event Dispatcher: {e}")
    
    def set_agent_coordinator(self, coordinator):
        """Set reference to agent coordinator for integration"""
        self.agent_coordinator = coordinator
    
    async def subscribe(self, event_type: str, handler: Callable, 
                      event_filter: Optional[Callable] = None,
                      priority: int = 0, max_retries: int = 3,
                      timeout_seconds: int = 30):
        """Subscribe to an event type with advanced options"""
        try:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            subscription = EventSubscription(
                handler=handler,
                event_filter=event_filter,
                priority=priority,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds
            )
            
            self.subscribers[event_type].append(subscription)
            
            # Sort by priority (higher first)
            self.subscribers[event_type].sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Subscribed to event type: {event_type} (priority: {priority})")
            
        except Exception as e:
            logger.error(f"Error subscribing to event {event_type}: {e}")
    
    async def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        try:
            if event_type in self.subscribers:
                self.subscribers[event_type] = [
                    sub for sub in self.subscribers[event_type]
                    if sub.handler != handler
                ]
                logger.info(f"Unsubscribed from event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error unsubscribing from event {event_type}: {e}")
    
    async def publish(self, event_type: str, data: Dict[str, Any], 
                    source: str = "system", priority: int = 0):
        """Publish an event with enhanced features"""
        try:
            event = {
                "id": f"{event_type}_{datetime.utcnow().timestamp()}",
                "type": event_type,
                "data": data,
                "source": source,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            
            # Apply global filters
            if not await self._apply_global_filters(event):
                logger.debug(f"Event {event_type} filtered out by global filters")
                return
            
            # Add to processing queue
            await self._processing_queue.put(event)
            
            # Update metrics
            await self._update_publish_metrics(event_type)
            
            logger.debug(f"Published event: {event_type} from {source}")
            
        except Exception as e:
            logger.error(f"Error publishing event {event_type}: {e}")
    
    async def publish_batch(self, events: List[Dict[str, Any]]):
        """Publish multiple events as a batch"""
        try:
            for event_data in events:
                await self.publish(
                    event_data.get("type"),
                    event_data.get("data", {}),
                    event_data.get("source", "system"),
                    event_data.get("priority", 0)
                )
                
        except Exception as e:
            logger.error(f"Error publishing event batch: {e}")
    
    async def add_global_filter(self, filter_func: Callable):
        """Add a global event filter"""
        self.global_filters.append(filter_func)
        logger.info("Added global event filter")
    
    async def add_event_route(self, source_event: str, target_events: List[str]):
        """Add event routing rule"""
        self.event_routes[source_event] = target_events
        logger.info(f"Added event route: {source_event} -> {target_events}")
    
    async def replay_events(self, event_type: Optional[str] = None,
                          from_time: Optional[datetime] = None,
                          to_time: Optional[datetime] = None):
        """Replay events from history"""
        try:
            events_to_replay = []
            
            for event in self.event_history:
                # Filter by type
                if event_type and event["type"] != event_type:
                    continue
                
                # Filter by time range
                event_time = datetime.fromisoformat(event["timestamp"])
                if from_time and event_time < from_time:
                    continue
                if to_time and event_time > to_time:
                    continue
                
                events_to_replay.append(event)
            
            logger.info(f"Replaying {len(events_to_replay)} events")
            
            for event in events_to_replay:
                await self._processing_queue.put(event)
                
        except Exception as e:
            logger.error(f"Error replaying events: {e}")
    
    async def get_event_history(self, event_type: Optional[str] = None, 
                              limit: Optional[int] = None,
                              source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get event history with enhanced filtering"""
        try:
            history = self.event_history
            
            # Filter by type
            if event_type:
                history = [e for e in history if e["type"] == event_type]
            
            # Filter by source
            if source:
                history = [e for e in history if e.get("source") == source]
            
            # Apply limit
            if limit:
                history = history[-limit:]
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []
    
    async def get_event_metrics(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Get event processing metrics"""
        try:
            if event_type:
                metrics = self.event_metrics.get(event_type)
                if metrics:
                    return {
                        "event_type": event_type,
                        "total_published": metrics.total_published,
                        "total_handled": metrics.total_handled,
                        "total_errors": metrics.total_errors,
                        "avg_processing_time": metrics.avg_processing_time,
                        "success_rate": (metrics.total_handled / max(metrics.total_published, 1)) * 100,
                        "error_rate": (metrics.total_errors / max(metrics.total_published, 1)) * 100,
                        "last_event_time": metrics.last_event_time.isoformat() if metrics.last_event_time else None
                    }
                return {}
            else:
                # Return all metrics
                all_metrics = {}
                for event_type, metrics in self.event_metrics.items():
                    all_metrics[event_type] = {
                        "total_published": metrics.total_published,
                        "total_handled": metrics.total_handled,
                        "total_errors": metrics.total_errors,
                        "avg_processing_time": metrics.avg_processing_time,
                        "success_rate": (metrics.total_handled / max(metrics.total_published, 1)) * 100,
                        "error_rate": (metrics.total_errors / max(metrics.total_published, 1)) * 100
                    }
                
                return {
                    "by_type": all_metrics,
                    "global": {
                        "total_published": sum(m.total_published for m in self.event_metrics.values()),
                        "total_handled": sum(m.total_handled for m in self.event_metrics.values()),
                        "total_errors": sum(m.total_errors for m in self.event_metrics.values()),
                        "queue_size": self._processing_queue.qsize(),
                        "active_processors": self.max_concurrent_events - self._processing_semaphore._value
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting event metrics: {e}")
            return {}
    
    # Internal processing methods
    
    async def _process_events(self):
        """Main event processing loop"""
        while self._running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self._processing_queue.get(), timeout=1.0)
                
                # Process event with concurrency control
                asyncio.create_task(self._process_single_event(event))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_single_event(self, event: Dict[str, Any]):
        """Process a single event"""
        async with self._processing_semaphore:
            try:
                start_time = datetime.utcnow()
                event_type = event["type"]
                
                # Add to history
                self._add_to_history(event)
                
                # Process subscribers
                if event_type in self.subscribers:
                    await self._notify_subscribers(event_type, event)
                
                # Process routing
                await self._process_event_routing(event)
                
                # Update metrics
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                await self._update_processing_metrics(event_type, processing_time, success=True)
                
                # Notify coordinator if available
                if self.agent_coordinator:
                    await self._notify_coordinator(event)
                
            except Exception as e:
                logger.error(f"Error processing event {event.get('type')}: {e}")
                await self._update_processing_metrics(event.get("type"), 0.0, success=False)
                
                # Retry logic
                await self._handle_event_retry(event, str(e))
    
    async def _notify_subscribers(self, event_type: str, event: Dict[str, Any]):
        """Notify all subscribers of an event"""
        try:
            subscribers = self.subscribers.get(event_type, [])
            
            for subscription in subscribers:
                try:
                    # Apply event filter if provided
                    if subscription.event_filter:
                        if not await self._apply_event_filter(subscription.event_filter, event):
                            continue
                    
                    # Execute handler with timeout
                    if asyncio.iscoroutinefunction(subscription.handler):
                        await asyncio.wait_for(
                            subscription.handler(event),
                            timeout=subscription.timeout_seconds
                        )
                    else:
                        subscription.handler(event)
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Event handler timeout for {event_type}")
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
                    await self._handle_handler_error(subscription, event, str(e))
                    
        except Exception as e:
            logger.error(f"Error notifying subscribers for {event_type}: {e}")
    
    async def _process_event_routing(self, event: Dict[str, Any]):
        """Process event routing rules"""
        try:
            event_type = event["type"]
            
            if event_type in self.event_routes:
                target_events = self.event_routes[event_type]
                
                for target_event_type in target_events:
                    # Create routed event
                    routed_event = event.copy()
                    routed_event["type"] = target_event_type
                    routed_event["source"] = f"routed_from_{event_type}"
                    routed_event["original_type"] = event_type
                    
                    # Add to processing queue
                    await self._processing_queue.put(routed_event)
                    
        except Exception as e:
            logger.error(f"Error processing event routing: {e}")
    
    async def _apply_global_filters(self, event: Dict[str, Any]) -> bool:
        """Apply global event filters"""
        try:
            for filter_func in self.global_filters:
                if not await self._apply_event_filter(filter_func, event):
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Error applying global filters: {e}")
            return True  # Allow event on error
    
    async def _apply_event_filter(self, filter_func: Callable, event: Dict[str, Any]) -> bool:
        """Apply a single event filter"""
        try:
            if asyncio.iscoroutinefunction(filter_func):
                return await filter_func(event)
            else:
                return filter_func(event)
                
        except Exception as e:
            logger.error(f"Error applying event filter: {e}")
            return True  # Allow event on error
    
    async def _handle_event_retry(self, event: Dict[str, Any], error: str):
        """Handle event retry logic"""
        try:
            event_type = event.get("type")
            retry_count = event.get("retry_count", 0)
            max_retries = 3  # Default max retries
            
            if retry_count < max_retries:
                event["retry_count"] = retry_count + 1
                event["last_error"] = error
                
                # Add delay before retry
                await asyncio.sleep(self.retry_delay * (retry_count + 1))
                
                # Re-queue event
                await self._processing_queue.put(event)
                
                logger.info(f"Retrying event {event_type} (attempt {retry_count + 1})")
            else:
                logger.error(f"Event {event_type} failed after {max_retries} retries: {error}")
                
        except Exception as e:
            logger.error(f"Error handling event retry: {e}")
    
    async def _handle_handler_error(self, subscription: EventSubscription, 
                                  event: Dict[str, Any], error: str):
        """Handle subscription handler error"""
        try:
            # Could implement handler-specific retry logic here
            logger.error(f"Handler error for event {event.get('type')}: {error}")
            
        except Exception as e:
            logger.error(f"Error handling handler error: {e}")
    
    async def _notify_coordinator(self, event: Dict[str, Any]):
        """Notify agent coordinator of event"""
        try:
            if hasattr(self.agent_coordinator, '_trigger_event'):
                await self.agent_coordinator._trigger_event(
                    f"dispatcher_{event['type']}", 
                    event
                )
                
        except Exception as e:
            logger.error(f"Error notifying coordinator: {e}")
    
    def _add_to_history(self, event: Dict[str, Any]):
        """Add event to history"""
        try:
            self.event_history.append(event.copy())
            
            # Trim history if too large
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
                
        except Exception as e:
            logger.error(f"Error adding event to history: {e}")
    
    async def _update_publish_metrics(self, event_type: str):
        """Update publish metrics for event type"""
        try:
            if event_type not in self.event_metrics:
                self.event_metrics[event_type] = EventMetrics()
            
            metrics = self.event_metrics[event_type]
            metrics.total_published += 1
            metrics.last_event_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating publish metrics: {e}")
    
    async def _update_processing_metrics(self, event_type: str, 
                                       processing_time: float, success: bool):
        """Update processing metrics for event type"""
        try:
            if event_type not in self.event_metrics:
                self.event_metrics[event_type] = EventMetrics()
            
            metrics = self.event_metrics[event_type]
            
            if success:
                metrics.total_handled += 1
                # Update average processing time
                total_time = metrics.avg_processing_time * (metrics.total_handled - 1) + processing_time
                metrics.avg_processing_time = total_time / metrics.total_handled
            else:
                metrics.total_errors += 1
                
        except Exception as e:
            logger.error(f"Error updating processing metrics: {e}")
    
    async def _metrics_collector(self):
        """Collect and log metrics periodically"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Log summary metrics
                total_published = sum(m.total_published for m in self.event_metrics.values())
                total_handled = sum(m.total_handled for m in self.event_metrics.values())
                total_errors = sum(m.total_errors for m in self.event_metrics.values())
                
                if total_published > 0:
                    success_rate = (total_handled / total_published) * 100
                    logger.info(f"Event metrics: {total_published} published, "
                              f"{total_handled} handled, {total_errors} errors, "
                              f"{success_rate:.1f}% success rate, "
                              f"queue size: {self._processing_queue.qsize()}")
                    
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)
