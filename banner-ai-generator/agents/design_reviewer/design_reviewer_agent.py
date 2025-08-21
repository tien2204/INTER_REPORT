"""
Design Reviewer Agent

Provides comprehensive automated design critique and quality assessment
with multi-criteria evaluation and improvement recommendations.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from structlog import get_logger

from communication.protocol import Message, MessageType
from memory_manager.shared_memory import SharedMemory
from communication.message_queue import MessageQueue
from .visual_analyzer import VisualAnalyzer
from .brand_compliance_checker import BrandComplianceChecker
from .accessibility_auditor import AccessibilityAuditor
from .performance_evaluator import PerformanceEvaluator
from .quality_scorer import QualityScorer

logger = get_logger(__name__)


@dataclass
class ReviewCriteria:
    """Design review criteria configuration"""
    visual_hierarchy: float = 1.0
    brand_compliance: float = 1.0
    accessibility: float = 1.0
    performance: float = 1.0
    user_experience: float = 1.0
    technical_quality: float = 1.0


@dataclass
class ReviewResult:
    """Design review result"""
    overall_score: float
    criteria_scores: Dict[str, float]
    issues: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    approved: bool
    review_summary: str


class DesignReviewerAgent:
    """
    AI Agent for automated design review and quality assessment
    
    Capabilities:
    - Multi-criteria design evaluation
    - Visual hierarchy analysis
    - Brand compliance checking
    - Accessibility auditing
    - Performance assessment
    - Automated feedback generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = "design_reviewer"
        self.agent_name = "Design Reviewer Agent"
        
        # Review configuration
        self.review_criteria = ReviewCriteria(**config.get("review_criteria", {}))
        self.approval_threshold = config.get("approval_threshold", 0.75)
        self.strict_mode = config.get("strict_mode", False)
        self.auto_approve = config.get("auto_approve", True)
        
        # Initialize review components
        self.visual_analyzer = VisualAnalyzer(config.get("visual_analyzer", {}))
        self.brand_checker = BrandComplianceChecker(config.get("brand_compliance", {}))
        self.accessibility_auditor = AccessibilityAuditor(config.get("accessibility", {}))
        self.performance_evaluator = PerformanceEvaluator(config.get("performance", {}))
        self.quality_scorer = QualityScorer(config.get("quality_scorer", {}))
        
        # Communication
        self.shared_memory: Optional[SharedMemory] = None
        self.message_queue: Optional[MessageQueue] = None
        
        # State management
        self._running = False
        self._active_reviews = {}
        self._processing_queue = asyncio.Queue()
        
        # Statistics
        self._total_reviews = 0
        self._approved_designs = 0
        self._avg_review_time = 0.0
        self._common_issues = {}
        
        logger.info(f"Design Reviewer Agent initialized: {self.agent_id}")
    
    async def start(self):
        """Start the agent"""
        try:
            logger.info("Starting Design Reviewer Agent")
            
            # Initialize review components
            await self.visual_analyzer.initialize()
            await self.brand_checker.initialize()
            await self.accessibility_auditor.initialize()
            await self.performance_evaluator.initialize()
            await self.quality_scorer.initialize()
            
            self._running = True
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            asyncio.create_task(self._process_review_queue())
            
            logger.info("Design Reviewer Agent started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Design Reviewer Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the agent"""
        try:
            logger.info("Stopping Design Reviewer Agent")
            
            self._running = False
            
            # Wait for active reviews to complete
            for review_id in list(self._active_reviews.keys()):
                await self._cleanup_review(review_id)
            
            logger.info("Design Reviewer Agent stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Design Reviewer Agent: {e}")
    
    def set_communication(self, shared_memory: SharedMemory, message_queue: MessageQueue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
        
        # Initialize components with communication
        self.visual_analyzer.set_communication(shared_memory, message_queue)
        self.brand_checker.set_communication(shared_memory, message_queue)
        self.accessibility_auditor.set_communication(shared_memory, message_queue)
        self.performance_evaluator.set_communication(shared_memory, message_queue)
        self.quality_scorer.set_communication(shared_memory, message_queue)
    
    async def _process_messages(self):
        """Process incoming messages"""
        if not self.message_queue:
            return
        
        try:
            await self.message_queue.subscribe(
                f"agent.{self.agent_id}",
                self._handle_message
            )
        except Exception as e:
            logger.error(f"Error in message processing: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle incoming agent message"""
        try:
            # Extract action from payload
            action = message.payload.get("action", "")
            data = message.payload.get("data", {})
            
            logger.info(f"Design Reviewer received message: {action}")
            
            if action == "start_design_review_workflow":
                await self._start_design_review_workflow(data)
            
            elif action == "review_design":
                await self._review_design(data)
            
            elif action == "update_review_criteria":
                await self._update_review_criteria(data)
            
            elif action == "get_review_status":
                await self._get_review_status(data)
            
            else:
                logger.warning(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_review_queue(self):
        """Process the review queue"""
        while self._running:
            try:
                task = await asyncio.wait_for(self._processing_queue.get(), timeout=1.0)
                await self._process_review_task(task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing review task: {e}")
    
    async def _start_design_review_workflow(self, data: Dict[str, Any]):
        """Start the design review workflow"""
        try:
            session_id = data.get("session_id")
            design_id = data.get("design_id")
            
            logger.info(f"Starting design review workflow: session={session_id}, design={design_id}")
            
            # Create review session
            review_session = {
                "session_id": session_id,
                "design_id": design_id,
                "started_at": datetime.utcnow(),
                "current_phase": "initialization",
                "progress": 0,
                "context": data.get("context", {}),
                "review_result": None
            }
            
            self._active_reviews[session_id] = review_session
            
            # Add to processing queue
            await self._processing_queue.put({
                "type": "design_review",
                "session_id": session_id,
                "data": data
            })
            
        except Exception as e:
            logger.error(f"Error starting design review workflow: {e}")
    
    async def _process_review_task(self, task: Dict[str, Any]):
        """Process a review task"""
        try:
            task_type = task.get("type")
            session_id = task.get("session_id")
            
            if task_type == "design_review":
                await self._execute_design_review_workflow(session_id, task["data"])
            
        except Exception as e:
            logger.error(f"Error processing review task: {e}")
    
    async def _execute_design_review_workflow(self, session_id: str, data: Dict[str, Any]):
        """Execute the complete design review workflow"""
        try:
            review_session = self._active_reviews.get(session_id)
            if not review_session:
                logger.error(f"Review session not found: {session_id}")
                return
            
            design_id = review_session["design_id"]
            
            # Phase 1: Load design data
            await self._update_review_progress(session_id, 10, "Loading design data")
            
            design_data = await self._load_design_data(design_id)
            if not design_data:
                await self._mark_review_failed(session_id, "Failed to load design data")
                return
            
            # Phase 2: Visual analysis
            await self._update_review_progress(session_id, 25, "Analyzing visual hierarchy")
            
            visual_analysis = await self.visual_analyzer.analyze_design(design_data)
            
            # Phase 3: Brand compliance check
            await self._update_review_progress(session_id, 40, "Checking brand compliance")
            
            brand_analysis = await self.brand_checker.check_compliance(design_data)
            
            # Phase 4: Accessibility audit
            await self._update_review_progress(session_id, 55, "Auditing accessibility")
            
            accessibility_analysis = await self.accessibility_auditor.audit_design(design_data)
            
            # Phase 5: Performance evaluation
            await self._update_review_progress(session_id, 70, "Evaluating performance")
            
            performance_analysis = await self.performance_evaluator.evaluate_design(design_data)
            
            # Phase 6: Quality scoring
            await self._update_review_progress(session_id, 85, "Calculating quality scores")
            
            # Combine all analyses
            combined_analysis = {
                "visual": visual_analysis,
                "brand": brand_analysis,
                "accessibility": accessibility_analysis,
                "performance": performance_analysis
            }
            
            # Generate overall quality score and recommendations
            review_result = await self.quality_scorer.calculate_overall_score(
                combined_analysis, self.review_criteria
            )
            
            # Phase 7: Generate review report
            await self._update_review_progress(session_id, 95, "Generating review report")
            
            review_report = await self._generate_review_report(
                design_data, combined_analysis, review_result
            )
            
            # Phase 8: Make approval decision
            approval_decision = await self._make_approval_decision(review_result)
            
            # Finalize review
            final_result = {
                "review_result": review_result,
                "review_report": review_report,
                "approval_decision": approval_decision,
                "analyses": combined_analysis
            }
            
            review_session["review_result"] = final_result
            
            # Update design with review results
            await self._update_design_with_review(design_id, final_result)
            
            # Complete review
            await self._update_review_progress(session_id, 100, "Review completed")
            await self._complete_review(session_id)
            
            # Notify next steps
            await self._notify_review_completion(design_id, session_id, approval_decision)
            
            logger.info(f"Design review workflow completed: session={session_id}")
            
        except Exception as e:
            logger.error(f"Error in design review workflow: {e}")
            await self._mark_review_failed(session_id, str(e))
    
    async def _load_design_data(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Load complete design data for review"""
        try:
            if not self.shared_memory:
                return None
            
            design_data = await self.shared_memory.get_design_data(design_id)
            
            if not design_data:
                return None
            
            # Ensure we have all necessary data
            required_sections = ["blueprint", "generated_code", "strategy", "background_data"]
            for section in required_sections:
                if section not in design_data:
                    logger.warning(f"Missing design section: {section}")
            
            return design_data
            
        except Exception as e:
            logger.error(f"Error loading design data: {e}")
            return None
    
    async def _generate_review_report(self, design_data: Dict[str, Any], 
                                    analyses: Dict[str, Any], 
                                    review_result: ReviewResult) -> Dict[str, Any]:
        """Generate comprehensive review report"""
        try:
            report = {
                "design_id": design_data.get("design_id"),
                "reviewed_at": datetime.utcnow().isoformat(),
                "overall_score": review_result.overall_score,
                "approval_status": "approved" if review_result.approved else "needs_revision",
                
                "executive_summary": {
                    "score": review_result.overall_score,
                    "status": "approved" if review_result.approved else "needs_revision",
                    "key_strengths": [],
                    "main_concerns": [],
                    "priority_actions": []
                },
                
                "detailed_analysis": {
                    "visual_hierarchy": {
                        "score": review_result.criteria_scores.get("visual_hierarchy", 0),
                        "findings": analyses.get("visual", {}).get("findings", []),
                        "recommendations": analyses.get("visual", {}).get("recommendations", [])
                    },
                    "brand_compliance": {
                        "score": review_result.criteria_scores.get("brand_compliance", 0),
                        "findings": analyses.get("brand", {}).get("findings", []),
                        "recommendations": analyses.get("brand", {}).get("recommendations", [])
                    },
                    "accessibility": {
                        "score": review_result.criteria_scores.get("accessibility", 0),
                        "findings": analyses.get("accessibility", {}).get("findings", []),
                        "recommendations": analyses.get("accessibility", {}).get("recommendations", [])
                    },
                    "performance": {
                        "score": review_result.criteria_scores.get("performance", 0),
                        "findings": analyses.get("performance", {}).get("findings", []),
                        "recommendations": analyses.get("performance", {}).get("recommendations", [])
                    }
                },
                
                "issues_summary": {
                    "critical": [issue for issue in review_result.issues if issue.get("severity") == "critical"],
                    "major": [issue for issue in review_result.issues if issue.get("severity") == "major"],
                    "minor": [issue for issue in review_result.issues if issue.get("severity") == "minor"],
                    "suggestions": [issue for issue in review_result.issues if issue.get("severity") == "suggestion"]
                },
                
                "recommendations": {
                    "immediate_actions": [rec for rec in review_result.recommendations if rec.get("priority") == "high"],
                    "suggested_improvements": [rec for rec in review_result.recommendations if rec.get("priority") == "medium"],
                    "optional_enhancements": [rec for rec in review_result.recommendations if rec.get("priority") == "low"]
                },
                
                "technical_metrics": {
                    "file_sizes": self._extract_file_sizes(design_data),
                    "performance_metrics": analyses.get("performance", {}).get("metrics", {}),
                    "accessibility_score": analyses.get("accessibility", {}).get("score", 0),
                    "code_quality": analyses.get("performance", {}).get("code_quality", {})
                }
            }
            
            # Generate executive summary content
            report["executive_summary"].update(
                await self._generate_executive_summary(analyses, review_result)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating review report: {e}")
            return {"error": str(e)}
    
    async def _generate_executive_summary(self, analyses: Dict[str, Any], 
                                        review_result: ReviewResult) -> Dict[str, str]:
        """Generate executive summary content"""
        try:
            # Extract key strengths
            strengths = []
            if review_result.criteria_scores.get("visual_hierarchy", 0) > 0.8:
                strengths.append("Strong visual hierarchy and clear information flow")
            if review_result.criteria_scores.get("brand_compliance", 0) > 0.8:
                strengths.append("Excellent brand consistency and guidelines adherence")
            if review_result.criteria_scores.get("accessibility", 0) > 0.8:
                strengths.append("Good accessibility compliance and inclusive design")
            if review_result.criteria_scores.get("performance", 0) > 0.8:
                strengths.append("Optimized performance and technical implementation")
            
            # Extract main concerns
            concerns = []
            for issue in review_result.issues:
                if issue.get("severity") in ["critical", "major"]:
                    concerns.append(issue.get("description", ""))
            
            # Extract priority actions
            priority_actions = []
            for rec in review_result.recommendations:
                if rec.get("priority") == "high":
                    priority_actions.append(rec.get("description", ""))
            
            return {
                "key_strengths": strengths[:3],  # Top 3 strengths
                "main_concerns": concerns[:3],   # Top 3 concerns
                "priority_actions": priority_actions[:3]  # Top 3 actions
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {
                "key_strengths": [],
                "main_concerns": [],
                "priority_actions": []
            }
    
    def _extract_file_sizes(self, design_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract file sizes from design data"""
        try:
            generated_code = design_data.get("generated_code", {})
            formats = generated_code.get("formats", {})
            
            file_sizes = {}
            for format_name, format_data in formats.items():
                file_size = format_data.get("metadata", {}).get("file_size_bytes")
                if file_size:
                    file_sizes[format_name] = file_size
            
            return file_sizes
            
        except Exception as e:
            logger.error(f"Error extracting file sizes: {e}")
            return {}
    
    async def _make_approval_decision(self, review_result: ReviewResult) -> Dict[str, Any]:
        """Make approval decision based on review result"""
        try:
            # Check overall score against threshold
            score_based_approval = review_result.overall_score >= self.approval_threshold
            
            # Check for critical issues
            critical_issues = [issue for issue in review_result.issues 
                             if issue.get("severity") == "critical"]
            has_critical_issues = len(critical_issues) > 0
            
            # Make decision
            if self.strict_mode:
                # In strict mode, any critical issue blocks approval
                approved = score_based_approval and not has_critical_issues
            else:
                # In normal mode, use score threshold
                approved = score_based_approval
            
            # Auto-approve if enabled and criteria met
            if self.auto_approve and approved:
                decision_type = "auto_approved"
            elif approved:
                decision_type = "approved"
            else:
                decision_type = "needs_revision"
            
            decision = {
                "approved": approved,
                "decision_type": decision_type,
                "score": review_result.overall_score,
                "threshold": self.approval_threshold,
                "critical_issues_count": len(critical_issues),
                "reasoning": self._generate_approval_reasoning(
                    review_result, score_based_approval, has_critical_issues
                ),
                "next_steps": self._generate_next_steps(approved, review_result)
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making approval decision: {e}")
            return {
                "approved": False,
                "decision_type": "error",
                "reasoning": f"Error in approval process: {e}",
                "next_steps": ["Contact technical support"]
            }
    
    def _generate_approval_reasoning(self, review_result: ReviewResult, 
                                   score_based: bool, has_critical: bool) -> str:
        """Generate reasoning for approval decision"""
        if review_result.overall_score >= self.approval_threshold and not has_critical:
            return f"Design meets quality standards with score {review_result.overall_score:.2f} and no critical issues."
        elif review_result.overall_score < self.approval_threshold:
            return f"Design score {review_result.overall_score:.2f} is below threshold {self.approval_threshold}."
        elif has_critical:
            return "Design has critical issues that must be addressed before approval."
        else:
            return "Design requires revision based on review criteria."
    
    def _generate_next_steps(self, approved: bool, review_result: ReviewResult) -> List[str]:
        """Generate next steps based on approval decision"""
        if approved:
            return [
                "Design approved for production",
                "Proceed with final asset generation",
                "Prepare for campaign deployment"
            ]
        else:
            steps = ["Address review feedback"]
            
            # Add specific steps based on issues
            critical_issues = [issue for issue in review_result.issues 
                             if issue.get("severity") == "critical"]
            if critical_issues:
                steps.append("Fix critical issues immediately")
            
            high_priority_recs = [rec for rec in review_result.recommendations 
                                if rec.get("priority") == "high"]
            if high_priority_recs:
                steps.append("Implement high-priority recommendations")
            
            steps.append("Resubmit for review")
            
            return steps
    
    async def _update_review_progress(self, session_id: str, progress: int, current_phase: str):
        """Update review progress"""
        try:
            review_session = self._active_reviews.get(session_id)
            if not review_session:
                return
            
            review_session["progress"] = progress
            review_session["current_phase"] = current_phase
            review_session["updated_at"] = datetime.utcnow()
            
            # Update design progress in shared memory
            if self.shared_memory and review_session.get("design_id"):
                await self.shared_memory.update_design_progress(
                    review_session["design_id"],
                    {
                        "progress_percentage": progress,
                        "current_step": current_phase,
                        "current_agent": self.agent_id
                    }
                )
            
        except Exception as e:
            logger.error(f"Error updating review progress: {e}")
    
    async def _mark_review_failed(self, session_id: str, error_message: str):
        """Mark review as failed"""
        try:
            review_session = self._active_reviews.get(session_id)
            if not review_session:
                return
            
            review_session["status"] = "failed"
            review_session["error_message"] = error_message
            review_session["completed_at"] = datetime.utcnow()
            
            # Update design status
            if self.shared_memory and review_session.get("design_id"):
                await self.shared_memory.update_design_progress(
                    review_session["design_id"],
                    {
                        "status": "review_failed",
                        "error_message": error_message,
                        "current_agent": self.agent_id
                    }
                )
            
            logger.error(f"Review failed: {session_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"Error marking review failed: {e}")
    
    async def _complete_review(self, session_id: str):
        """Mark review as completed"""
        try:
            review_session = self._active_reviews.get(session_id)
            if not review_session:
                return
            
            review_session["status"] = "completed"
            review_session["completed_at"] = datetime.utcnow()
            
            # Update statistics
            self._total_reviews += 1
            
            review_result = review_session.get("review_result", {})
            approval_decision = review_result.get("approval_decision", {})
            
            if approval_decision.get("approved"):
                self._approved_designs += 1
            
            # Calculate review time
            review_time = (review_session["completed_at"] - review_session["started_at"]).total_seconds()
            self._avg_review_time = (
                (self._avg_review_time * (self._total_reviews - 1) + review_time) 
                / self._total_reviews
            )
            
            # Track common issues
            issues = review_result.get("review_result", {}).get("issues", [])
            for issue in issues:
                issue_type = issue.get("type", "unknown")
                self._common_issues[issue_type] = self._common_issues.get(issue_type, 0) + 1
            
            logger.info(f"Review completed: {session_id} (review time: {review_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error completing review: {e}")
    
    async def _update_design_with_review(self, design_id: str, review_result: Dict[str, Any]):
        """Update design data with review results"""
        try:
            if not self.shared_memory:
                return
            
            design_data = await self.shared_memory.get_design_data(design_id)
            if not design_data:
                return
            
            # Update with review results
            design_data["design_review"] = {
                "result": review_result,
                "reviewed_at": datetime.utcnow().isoformat(),
                "reviewer_agent": self.agent_id,
                "approval_status": review_result.get("approval_decision", {}).get("decision_type", "unknown")
            }
            
            # Update design status
            approval_decision = review_result.get("approval_decision", {})
            if approval_decision.get("approved"):
                design_data["status"] = "approved"
                design_data["progress_percentage"] = 100
                design_data["current_step"] = "Design approved"
            else:
                design_data["status"] = "needs_revision"
                design_data["current_step"] = "Awaiting revisions"
            
            await self.shared_memory.set_design_data(design_id, design_data)
            
        except Exception as e:
            logger.error(f"Error updating design data: {e}")
    
    async def _notify_review_completion(self, design_id: str, session_id: str, 
                                      approval_decision: Dict[str, Any]):
        """Notify about review completion"""
        try:
            if not self.message_queue:
                return
            
            # Notify campaign management system
            message = Message(
                sender=self.agent_id,
                recipient="campaign_manager",
                type=MessageType.NOTIFICATION,
                payload={
                    "action": "design_review_completed",
                    "data": {
                        "design_id": design_id,
                        "session_id": session_id,
                        "approved": approval_decision.get("approved", False),
                        "decision_type": approval_decision.get("decision_type"),
                        "score": approval_decision.get("score", 0)
                    }
                },
                timestamp=datetime.utcnow()
            )
            
            await self.message_queue.publish("system.campaign_manager", message)
            
        except Exception as e:
            logger.error(f"Error notifying review completion: {e}")
    
    async def _cleanup_review(self, session_id: str):
        """Clean up review resources"""
        try:
            if session_id in self._active_reviews:
                del self._active_reviews[session_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up review {session_id}: {e}")
    
    # Public API methods
    async def review_design(self, design_id: str, criteria: Optional[ReviewCriteria] = None) -> ReviewResult:
        """Review a design with specified criteria"""
        try:
            # Use provided criteria or defaults
            review_criteria = criteria or self.review_criteria
            
            # Load design data
            design_data = await self._load_design_data(design_id)
            if not design_data:
                raise ValueError(f"Design not found: {design_id}")
            
            # Perform analyses
            analyses = {}
            analyses["visual"] = await self.visual_analyzer.analyze_design(design_data)
            analyses["brand"] = await self.brand_checker.check_compliance(design_data)
            analyses["accessibility"] = await self.accessibility_auditor.audit_design(design_data)
            analyses["performance"] = await self.performance_evaluator.evaluate_design(design_data)
            
            # Calculate overall score
            review_result = await self.quality_scorer.calculate_overall_score(
                analyses, review_criteria
            )
            
            return review_result
            
        except Exception as e:
            logger.error(f"Error reviewing design: {e}")
            return ReviewResult(
                overall_score=0.0,
                criteria_scores={},
                issues=[{"type": "error", "description": str(e), "severity": "critical"}],
                recommendations=[],
                approved=False,
                review_summary=f"Review failed: {e}"
            )
    
    async def get_review_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a review session"""
        review_session = self._active_reviews.get(session_id)
        if not review_session:
            return {"error": "Review session not found"}
        
        return {
            "session_id": session_id,
            "design_id": review_session.get("design_id"),
            "status": review_session.get("status", "active"),
            "progress": review_session.get("progress", 0),
            "current_phase": review_session.get("current_phase"),
            "started_at": review_session.get("started_at"),
            "updated_at": review_session.get("updated_at"),
            "error_message": review_session.get("error_message")
        }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "running": self._running,
            "active_reviews": len(self._active_reviews),
            "total_reviews": self._total_reviews,
            "approved_designs": self._approved_designs,
            "approval_rate": (self._approved_designs / self._total_reviews * 100) if self._total_reviews > 0 else 0,
            "avg_review_time": self._avg_review_time,
            "queue_size": self._processing_queue.qsize(),
            "common_issues": self._common_issues,
            "review_criteria": {
                "visual_hierarchy": self.review_criteria.visual_hierarchy,
                "brand_compliance": self.review_criteria.brand_compliance,
                "accessibility": self.review_criteria.accessibility,
                "performance": self.review_criteria.performance,
                "user_experience": self.review_criteria.user_experience,
                "technical_quality": self.review_criteria.technical_quality
            },
            "approval_threshold": self.approval_threshold,
            "strict_mode": self.strict_mode
        }
    
    async def _review_design(self, data: Dict[str, Any]):
        """Handle review_design message"""
        try:
            design_id = data.get("design_id")
            if not design_id:
                logger.error("No design_id provided for review")
                return
            
            # Create a new review session for this request
            session_id = f"review_{design_id}_{int(datetime.utcnow().timestamp())}"
            await self._start_design_review_workflow({
                "session_id": session_id,
                "design_id": design_id,
                "context": data
            })
            
        except Exception as e:
            logger.error(f"Error handling review_design message: {e}")
    
    async def _update_review_criteria(self, data: Dict[str, Any]):
        """Handle update_review_criteria message"""
        try:
            criteria_data = data.get("criteria", {})
            
            # Update review criteria
            if "visual_hierarchy" in criteria_data:
                self.review_criteria.visual_hierarchy = criteria_data["visual_hierarchy"]
            if "brand_compliance" in criteria_data:
                self.review_criteria.brand_compliance = criteria_data["brand_compliance"]
            if "accessibility" in criteria_data:
                self.review_criteria.accessibility = criteria_data["accessibility"]
            if "performance" in criteria_data:
                self.review_criteria.performance = criteria_data["performance"]
            if "user_experience" in criteria_data:
                self.review_criteria.user_experience = criteria_data["user_experience"]
            if "technical_quality" in criteria_data:
                self.review_criteria.technical_quality = criteria_data["technical_quality"]
            
            # Update threshold if provided
            if "approval_threshold" in data:
                self.approval_threshold = data["approval_threshold"]
            
            # Update strict mode if provided
            if "strict_mode" in data:
                self.strict_mode = data["strict_mode"]
            
            logger.info("Review criteria updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating review criteria: {e}")
    
    async def _get_review_status(self, data: Dict[str, Any]):
        """Handle get_review_status message"""
        try:
            session_id = data.get("session_id")
            if not session_id:
                logger.error("No session_id provided for status request")
                return
            
            status = await self.get_review_status(session_id)
            
            # Send response back if message queue is available
            if self.message_queue:
                response_message = Message(
                    type=MessageType.RESPONSE,
                    sender=self.agent_id,
                    payload={
                        "success": True,
                        "data": status
                    }
                )
                await self.message_queue.publish("system.responses", response_message)
            
        except Exception as e:
            logger.error(f"Error getting review status: {e}")
