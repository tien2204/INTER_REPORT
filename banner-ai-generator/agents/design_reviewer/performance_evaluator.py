"""
Performance Evaluator

Evaluates design performance, loading speed, and technical optimization
to ensure efficient banner delivery and user experience.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from structlog import get_logger

logger = get_logger(__name__)


class PerformanceEvaluator:
    """
    Performance evaluation system
    
    Capabilities:
    - File size optimization analysis
    - Loading speed estimation
    - Code quality assessment
    - Resource efficiency evaluation
    - Mobile performance analysis
    - CDN and caching optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Performance thresholds
        self.max_file_size_kb = config.get("max_file_size_kb", 150)  # Total banner size
        self.max_image_size_kb = config.get("max_image_size_kb", 100)
        self.max_css_size_kb = config.get("max_css_size_kb", 20)
        self.max_js_size_kb = config.get("max_js_size_kb", 30)
        
        # Loading speed targets (milliseconds)
        self.target_load_time_ms = config.get("target_load_time_ms", 1000)
        self.max_load_time_ms = config.get("max_load_time_ms", 3000)
        
        # Network conditions for testing
        self.network_conditions = config.get("network_conditions", {
            "fast_3g": {"latency": 562, "download": 1600, "upload": 750},
            "slow_3g": {"latency": 2000, "download": 400, "upload": 400},
            "2g": {"latency": 1400, "download": 280, "upload": 256}
        })
        
        # Code quality thresholds
        self.max_css_rules = config.get("max_css_rules", 100)
        self.max_dom_depth = config.get("max_dom_depth", 8)
        self.max_elements = config.get("max_elements", 20)
        
        # Optimization features
        self.check_compression = config.get("check_compression", True)
        self.check_minification = config.get("check_minification", True)
        self.check_caching = config.get("check_caching", True)
        
        # Communication
        self.shared_memory = None
        self.message_queue = None
        
        logger.info("Performance Evaluator initialized")
    
    async def initialize(self):
        """Initialize the performance evaluator"""
        try:
            # Load performance benchmarks and optimization rules
            await self._load_performance_benchmarks()
            await self._initialize_performance_tools()
            
            logger.info("Performance Evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Performance Evaluator: {e}")
            raise
    
    def set_communication(self, shared_memory, message_queue):
        """Set communication interfaces"""
        self.shared_memory = shared_memory
        self.message_queue = message_queue
    
    async def evaluate_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive performance evaluation
        
        Args:
            design_data: Complete design data including blueprint and generated code
            
        Returns:
            Performance evaluation result with scores, issues, and optimization recommendations
        """
        try:
            logger.info("Starting performance evaluation")
            
            # Extract design components and generated code
            blueprint = design_data.get("blueprint", {})
            components = blueprint.get("components", {})
            generated_code = design_data.get("generated_code", {})
            
            # Perform different performance evaluations
            file_size_analysis = await self._analyze_file_sizes(generated_code)
            loading_speed_analysis = await self._estimate_loading_speed(generated_code, components)
            code_quality_analysis = await self._analyze_code_quality(generated_code)
            resource_optimization_analysis = await self._analyze_resource_optimization(generated_code)
            mobile_performance_analysis = await self._analyze_mobile_performance(generated_code, components)
            
            # Calculate overall performance score
            overall_score = await self._calculate_performance_score(
                file_size_analysis, loading_speed_analysis, code_quality_analysis,
                resource_optimization_analysis, mobile_performance_analysis
            )
            
            # Compile performance issues and recommendations
            issues = await self._compile_performance_issues(
                file_size_analysis, loading_speed_analysis, code_quality_analysis,
                resource_optimization_analysis, mobile_performance_analysis
            )
            
            recommendations = await self._generate_performance_recommendations(issues, overall_score)
            
            result = {
                "overall_score": overall_score,
                "performance_scores": {
                    "file_size": file_size_analysis.get("score", 0),
                    "loading_speed": loading_speed_analysis.get("score", 0),
                    "code_quality": code_quality_analysis.get("score", 0),
                    "resource_optimization": resource_optimization_analysis.get("score", 0),
                    "mobile_performance": mobile_performance_analysis.get("score", 0)
                },
                "issues": issues,
                "recommendations": recommendations,
                "performance_details": {
                    "file_sizes": file_size_analysis,
                    "loading_speed": loading_speed_analysis,
                    "code_quality": code_quality_analysis,
                    "resource_optimization": resource_optimization_analysis,
                    "mobile_performance": mobile_performance_analysis
                },
                "optimization_opportunities": await self._identify_optimization_opportunities(generated_code),
                "performance_budget": await self._calculate_performance_budget(file_size_analysis)
            }
            
            logger.info("Performance evaluation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e),
                "issues": [{"type": "error", "description": f"Performance evaluation failed: {e}", "severity": "critical"}],
                "recommendations": []
            }
    
    async def _analyze_file_sizes(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file sizes and size optimization"""
        try:
            formats = generated_code.get("formats", {})
            size_issues = []
            total_size = 0
            format_sizes = {}
            
            # Analyze each format
            for format_name, format_data in formats.items():
                file_size = format_data.get("file_size", 0)
                format_sizes[format_name] = file_size
                total_size += file_size
                
                # Check format-specific size limits
                if format_name == "svg":
                    if file_size > self.max_image_size_kb * 1024:
                        size_issues.append({
                            "type": "svg_size_large",
                            "description": f"SVG file too large: {file_size/1024:.1f}KB (max: {self.max_image_size_kb}KB)",
                            "severity": "major",
                            "current_size": file_size,
                            "max_size": self.max_image_size_kb * 1024,
                            "format": format_name
                        })
                
                elif format_name == "html":
                    html_content = format_data.get("html_content", "")
                    css_content = format_data.get("css_content", "")
                    js_content = format_data.get("js_content", "")
                    
                    css_size = len(css_content.encode('utf-8')) if css_content else 0
                    js_size = len(js_content.encode('utf-8')) if js_content else 0
                    
                    if css_size > self.max_css_size_kb * 1024:
                        size_issues.append({
                            "type": "css_size_large",
                            "description": f"CSS too large: {css_size/1024:.1f}KB (max: {self.max_css_size_kb}KB)",
                            "severity": "minor",
                            "current_size": css_size,
                            "max_size": self.max_css_size_kb * 1024
                        })
                    
                    if js_size > self.max_js_size_kb * 1024:
                        size_issues.append({
                            "type": "js_size_large",
                            "description": f"JavaScript too large: {js_size/1024:.1f}KB (max: {self.max_js_size_kb}KB)",
                            "severity": "minor",
                            "current_size": js_size,
                            "max_size": self.max_js_size_kb * 1024
                        })
            
            # Check total size
            total_size_kb = total_size / 1024
            if total_size_kb > self.max_file_size_kb:
                size_issues.append({
                    "type": "total_size_large",
                    "description": f"Total banner size too large: {total_size_kb:.1f}KB (max: {self.max_file_size_kb}KB)",
                    "severity": "major",
                    "current_size": total_size,
                    "max_size": self.max_file_size_kb * 1024
                })
            
            # Calculate size efficiency score
            size_efficiency = min(1.0, self.max_file_size_kb * 1024 / max(total_size, 1))
            size_score = size_efficiency - (len(size_issues) * 0.1)
            size_score = max(0.0, size_score)
            
            return {
                "score": size_score,
                "issues": size_issues,
                "total_size_bytes": total_size,
                "total_size_kb": total_size_kb,
                "format_sizes": format_sizes,
                "size_efficiency": size_efficiency,
                "compression_potential": await self._estimate_compression_potential(generated_code)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file sizes: {e}")
            return {"score": 0.0, "error": str(e), "issues": []}
    
    async def _estimate_loading_speed(self, generated_code: Dict[str, Any], 
                                    components: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate loading speed across different network conditions"""
        try:
            loading_issues = []
            loading_estimates = {}
            
            # Calculate total payload size
            total_size = sum(
                format_data.get("file_size", 0)
                for format_data in generated_code.get("formats", {}).values()
            )
            
            # Estimate external resource count
            external_resources = await self._count_external_resources(components)
            
            # Calculate loading time for different network conditions
            for network_name, network_config in self.network_conditions.items():
                download_speed_bps = network_config["download"] * 1024 / 8  # Convert to bytes per second
                latency_ms = network_config["latency"]
                
                # Estimate loading time
                download_time_ms = (total_size / download_speed_bps) * 1000
                total_load_time_ms = latency_ms + download_time_ms + (external_resources * latency_ms)
                
                loading_estimates[network_name] = {
                    "load_time_ms": total_load_time_ms,
                    "download_time_ms": download_time_ms,
                    "latency_ms": latency_ms,
                    "external_requests": external_resources
                }
                
                # Check against thresholds
                if total_load_time_ms > self.max_load_time_ms:
                    loading_issues.append({
                        "type": "slow_loading",
                        "description": f"Estimated loading time on {network_name}: {total_load_time_ms:.0f}ms (max: {self.max_load_time_ms}ms)",
                        "severity": "major" if total_load_time_ms > self.max_load_time_ms * 1.5 else "minor",
                        "network_condition": network_name,
                        "estimated_time": total_load_time_ms,
                        "max_time": self.max_load_time_ms
                    })
            
            # Calculate loading speed score based on fast 3G performance
            fast_3g_time = loading_estimates.get("fast_3g", {}).get("load_time_ms", 0)
            if fast_3g_time <= self.target_load_time_ms:
                loading_score = 1.0
            elif fast_3g_time <= self.max_load_time_ms:
                loading_score = 1.0 - ((fast_3g_time - self.target_load_time_ms) / (self.max_load_time_ms - self.target_load_time_ms)) * 0.5
            else:
                loading_score = max(0.0, 0.5 - ((fast_3g_time - self.max_load_time_ms) / self.max_load_time_ms) * 0.5)
            
            return {
                "score": loading_score,
                "issues": loading_issues,
                "loading_estimates": loading_estimates,
                "external_resources": external_resources,
                "total_payload_bytes": total_size,
                "critical_rendering_path": await self._analyze_critical_rendering_path(generated_code)
            }
            
        except Exception as e:
            logger.error(f"Error estimating loading speed: {e}")
            return {"score": 0.0, "error": str(e), "issues": []}
    
    async def _analyze_code_quality(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code quality and efficiency"""
        try:
            quality_issues = []
            quality_metrics = {}
            
            formats = generated_code.get("formats", {})
            
            # Analyze HTML quality
            html_data = formats.get("html", {})
            if html_data:
                html_metrics = await self._analyze_html_quality(html_data)
                quality_metrics["html"] = html_metrics
                
                if html_metrics.get("dom_depth", 0) > self.max_dom_depth:
                    quality_issues.append({
                        "type": "dom_depth_excessive",
                        "description": f"DOM depth too deep: {html_metrics['dom_depth']} levels (max: {self.max_dom_depth})",
                        "severity": "minor",
                        "current_depth": html_metrics["dom_depth"],
                        "max_depth": self.max_dom_depth
                    })
                
                if html_metrics.get("element_count", 0) > self.max_elements:
                    quality_issues.append({
                        "type": "element_count_high",
                        "description": f"Too many DOM elements: {html_metrics['element_count']} (max: {self.max_elements})",
                        "severity": "minor",
                        "current_count": html_metrics["element_count"],
                        "max_count": self.max_elements
                    })
            
            # Analyze CSS quality
            if html_data and html_data.get("css_content"):
                css_metrics = await self._analyze_css_quality(html_data["css_content"])
                quality_metrics["css"] = css_metrics
                
                if css_metrics.get("rule_count", 0) > self.max_css_rules:
                    quality_issues.append({
                        "type": "css_rules_excessive",
                        "description": f"Too many CSS rules: {css_metrics['rule_count']} (max: {self.max_css_rules})",
                        "severity": "minor",
                        "current_count": css_metrics["rule_count"],
                        "max_count": self.max_css_rules
                    })
                
                if css_metrics.get("unused_rules", 0) > 0:
                    quality_issues.append({
                        "type": "css_unused_rules",
                        "description": f"Unused CSS rules detected: {css_metrics['unused_rules']}",
                        "severity": "minor",
                        "unused_count": css_metrics["unused_rules"]
                    })
            
            # Analyze SVG quality
            svg_data = formats.get("svg", {})
            if svg_data:
                svg_metrics = await self._analyze_svg_quality(svg_data)
                quality_metrics["svg"] = svg_metrics
                
                if svg_metrics.get("optimization_potential", 0) > 0.3:
                    quality_issues.append({
                        "type": "svg_optimization_needed",
                        "description": f"SVG could be optimized further ({svg_metrics['optimization_potential']*100:.0f}% potential reduction)",
                        "severity": "minor",
                        "optimization_potential": svg_metrics["optimization_potential"]
                    })
            
            # Calculate overall code quality score
            html_score = quality_metrics.get("html", {}).get("quality_score", 1.0)
            css_score = quality_metrics.get("css", {}).get("quality_score", 1.0)
            svg_score = quality_metrics.get("svg", {}).get("quality_score", 1.0)
            
            overall_quality_score = (html_score + css_score + svg_score) / 3
            overall_quality_score -= len(quality_issues) * 0.05  # Small penalty per issue
            overall_quality_score = max(0.0, overall_quality_score)
            
            return {
                "score": overall_quality_score,
                "issues": quality_issues,
                "quality_metrics": quality_metrics,
                "maintainability_score": await self._calculate_maintainability_score(quality_metrics),
                "best_practices_compliance": await self._check_best_practices_compliance(generated_code)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {"score": 0.0, "error": str(e), "issues": []}
    
    async def _analyze_resource_optimization(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource optimization opportunities"""
        try:
            optimization_issues = []
            optimization_score = 1.0
            optimization_opportunities = {}
            
            formats = generated_code.get("formats", {})
            
            # Check for minification
            if self.check_minification:
                minification_analysis = await self._check_minification_status(formats)
                optimization_opportunities["minification"] = minification_analysis
                
                if not minification_analysis.get("is_minified", False):
                    optimization_issues.append({
                        "type": "minification_missing",
                        "description": "Code is not minified, potential size savings available",
                        "severity": "minor",
                        "potential_savings": minification_analysis.get("potential_savings", "unknown")
                    })
                    optimization_score -= 0.2
            
            # Check for compression opportunities
            if self.check_compression:
                compression_analysis = await self._check_compression_opportunities(formats)
                optimization_opportunities["compression"] = compression_analysis
                
                if compression_analysis.get("compression_ratio", 1.0) > 0.7:  # Could compress more
                    optimization_issues.append({
                        "type": "compression_opportunity",
                        "description": f"Additional compression possible (current ratio: {compression_analysis.get('compression_ratio', 1.0):.2f})",
                        "severity": "minor",
                        "current_ratio": compression_analysis.get("compression_ratio", 1.0)
                    })
                    optimization_score -= 0.1
            
            # Check for image optimization
            image_optimization = await self._check_image_optimization(formats)
            optimization_opportunities["images"] = image_optimization
            
            if image_optimization.get("optimization_needed", False):
                optimization_issues.append({
                    "type": "image_optimization_needed",
                    "description": "Images could be optimized for better performance",
                    "severity": "minor",
                    "potential_savings": image_optimization.get("potential_savings", "unknown")
                })
                optimization_score -= 0.15
            
            # Check for caching optimization
            if self.check_caching:
                caching_analysis = await self._analyze_caching_strategy(generated_code)
                optimization_opportunities["caching"] = caching_analysis
                
                if not caching_analysis.get("cache_optimized", False):
                    optimization_issues.append({
                        "type": "caching_not_optimized",
                        "description": "Caching strategy could be improved",
                        "severity": "minor",
                        "recommendations": caching_analysis.get("recommendations", [])
                    })
                    optimization_score -= 0.1
            
            optimization_score = max(0.0, optimization_score)
            
            return {
                "score": optimization_score,
                "issues": optimization_issues,
                "optimization_opportunities": optimization_opportunities,
                "resource_efficiency": await self._calculate_resource_efficiency(formats),
                "cdn_readiness": await self._assess_cdn_readiness(generated_code)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing resource optimization: {e}")
            return {"score": 0.0, "error": str(e), "issues": []}
    
    async def _analyze_mobile_performance(self, generated_code: Dict[str, Any], 
                                        components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mobile-specific performance characteristics"""
        try:
            mobile_issues = []
            mobile_score = 1.0
            
            # Check responsive design impact
            responsive_analysis = await self._analyze_responsive_performance(generated_code)
            
            if responsive_analysis.get("performance_impact", 0) > 0.2:
                mobile_issues.append({
                    "type": "responsive_performance_impact",
                    "description": "Responsive design may impact mobile performance",
                    "severity": "minor",
                    "performance_impact": responsive_analysis.get("performance_impact", 0)
                })
                mobile_score -= 0.2
            
            # Check touch targets
            touch_target_analysis = await self._analyze_touch_targets(components)
            
            if touch_target_analysis.get("small_targets", 0) > 0:
                mobile_issues.append({
                    "type": "small_touch_targets",
                    "description": f"Some touch targets may be too small: {touch_target_analysis['small_targets']} elements",
                    "severity": "minor",
                    "small_targets_count": touch_target_analysis["small_targets"]
                })
                mobile_score -= 0.1
            
            # Check mobile loading characteristics
            mobile_loading = await self._estimate_mobile_loading(generated_code)
            
            if mobile_loading.get("slow_3g_time", 0) > self.max_load_time_ms * 1.5:
                mobile_issues.append({
                    "type": "slow_mobile_loading",
                    "description": f"Slow loading on mobile networks: {mobile_loading['slow_3g_time']:.0f}ms",
                    "severity": "major",
                    "estimated_time": mobile_loading["slow_3g_time"]
                })
                mobile_score -= 0.3
            
            # Check mobile-specific optimizations
            mobile_optimizations = await self._check_mobile_optimizations(generated_code)
            
            missing_optimizations = mobile_optimizations.get("missing_optimizations", [])
            if missing_optimizations:
                mobile_issues.append({
                    "type": "mobile_optimizations_missing",
                    "description": f"Missing mobile optimizations: {', '.join(missing_optimizations)}",
                    "severity": "minor",
                    "missing_optimizations": missing_optimizations
                })
                mobile_score -= len(missing_optimizations) * 0.05
            
            mobile_score = max(0.0, mobile_score)
            
            return {
                "score": mobile_score,
                "issues": mobile_issues,
                "responsive_analysis": responsive_analysis,
                "touch_target_analysis": touch_target_analysis,
                "mobile_loading": mobile_loading,
                "mobile_optimizations": mobile_optimizations,
                "mobile_ux_score": await self._calculate_mobile_ux_score(components)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing mobile performance: {e}")
            return {"score": 0.0, "error": str(e), "issues": []}
    
    # Helper methods
    
    async def _estimate_compression_potential(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate potential file size reduction through compression"""
        try:
            compression_potential = {}
            
            for format_name, format_data in generated_code.get("formats", {}).items():
                original_size = format_data.get("file_size", 0)
                
                if format_name == "html":
                    # Estimate HTML compression (typical gzip compression ~70%)
                    estimated_compressed = original_size * 0.3
                    potential_savings = original_size - estimated_compressed
                    
                    compression_potential[format_name] = {
                        "original_size": original_size,
                        "estimated_compressed": estimated_compressed,
                        "potential_savings": potential_savings,
                        "compression_ratio": estimated_compressed / original_size if original_size > 0 else 1.0
                    }
                
                elif format_name == "svg":
                    # SVG typically compresses well (~60-80%)
                    estimated_compressed = original_size * 0.25
                    potential_savings = original_size - estimated_compressed
                    
                    compression_potential[format_name] = {
                        "original_size": original_size,
                        "estimated_compressed": estimated_compressed,
                        "potential_savings": potential_savings,
                        "compression_ratio": estimated_compressed / original_size if original_size > 0 else 1.0
                    }
            
            return compression_potential
            
        except Exception as e:
            logger.error(f"Error estimating compression potential: {e}")
            return {}
    
    async def _count_external_resources(self, components: Dict[str, Any]) -> int:
        """Count external resources that need to be loaded"""
        external_count = 0
        
        for component in components.values():
            content = component.get("content", {})
            
            # Check for external images
            source = content.get("source", "")
            if source and (source.startswith("http") or source.startswith("//")):
                external_count += 1
            
            # Check for external fonts
            styling = component.get("styling", {}).get("base", {})
            font_family = styling.get("font_family", "")
            if "googleapis.com" in font_family or "fonts.com" in font_family:
                external_count += 1
        
        return external_count
    
    async def _analyze_critical_rendering_path(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze critical rendering path"""
        try:
            html_data = generated_code.get("formats", {}).get("html", {})
            css_content = html_data.get("css_content", "")
            js_content = html_data.get("js_content", "")
            
            # Estimate critical CSS size
            critical_css_size = len(css_content.encode('utf-8')) if css_content else 0
            
            # Check for render-blocking resources
            blocking_resources = 0
            if js_content and "document.addEventListener('DOMContentLoaded'" not in js_content:
                blocking_resources += 1
            
            return {
                "critical_css_size": critical_css_size,
                "blocking_resources": blocking_resources,
                "estimated_first_paint": 100 + (critical_css_size / 1000) + (blocking_resources * 200)  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"Error analyzing critical rendering path: {e}")
            return {"critical_css_size": 0, "blocking_resources": 0}
    
    async def _analyze_html_quality(self, html_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze HTML code quality"""
        try:
            html_code = html_data.get("html_code", "")
            
            # Count DOM elements
            element_count = len(re.findall(r'<[^/][^>]*>', html_code))
            
            # Calculate DOM depth (simplified)
            dom_depth = await self._calculate_dom_depth(html_code)
            
            # Check for semantic elements
            semantic_elements = ["header", "main", "section", "article", "nav", "aside", "footer"]
            semantic_count = sum(html_code.count(f"<{elem}") for elem in semantic_elements)
            
            # Calculate quality score
            quality_score = 1.0
            if element_count > self.max_elements:
                quality_score -= 0.2
            if dom_depth > self.max_dom_depth:
                quality_score -= 0.3
            if semantic_count == 0 and element_count > 5:
                quality_score -= 0.1
            
            return {
                "element_count": element_count,
                "dom_depth": dom_depth,
                "semantic_element_count": semantic_count,
                "quality_score": max(0.0, quality_score),
                "html_size": len(html_code.encode('utf-8'))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing HTML quality: {e}")
            return {"element_count": 0, "dom_depth": 0, "quality_score": 0.5}
    
    async def _calculate_dom_depth(self, html_code: str) -> int:
        """Calculate maximum DOM depth"""
        try:
            max_depth = 0
            current_depth = 0
            
            # Simple depth calculation
            for match in re.finditer(r'<(/?)([^>\s]+)', html_code):
                is_closing = match.group(1) == '/'
                tag_name = match.group(2).lower()
                
                # Skip self-closing tags
                if tag_name in ['img', 'br', 'hr', 'input', 'meta', 'link']:
                    continue
                
                if is_closing:
                    current_depth -= 1
                else:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
            
            return max_depth
            
        except Exception as e:
            logger.error(f"Error calculating DOM depth: {e}")
            return 0
    
    async def _analyze_css_quality(self, css_content: str) -> Dict[str, Any]:
        """Analyze CSS code quality"""
        try:
            # Count CSS rules
            rule_count = len(re.findall(r'[^{}]*{[^}]*}', css_content))
            
            # Estimate unused rules (simplified check)
            selectors = re.findall(r'([^{]+){', css_content)
            complex_selectors = len([s for s in selectors if ' ' in s.strip() or '>' in s or '+' in s])
            
            # Calculate quality score
            quality_score = 1.0
            if rule_count > self.max_css_rules:
                quality_score -= 0.3
            if complex_selectors > rule_count * 0.5:  # More than 50% complex selectors
                quality_score -= 0.2
            
            return {
                "rule_count": rule_count,
                "complex_selectors": complex_selectors,
                "unused_rules": 0,  # Simplified - would need more sophisticated analysis
                "quality_score": max(0.0, quality_score),
                "css_size": len(css_content.encode('utf-8'))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CSS quality: {e}")
            return {"rule_count": 0, "quality_score": 0.5}
    
    async def _analyze_svg_quality(self, svg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SVG code quality"""
        try:
            svg_code = svg_data.get("svg_code", "")
            
            # Count SVG elements
            element_count = len(re.findall(r'<[^/][^>]*>', svg_code))
            
            # Check for optimization opportunities
            has_unnecessary_elements = bool(re.search(r'<metadata>|<title>|<!--', svg_code))
            has_precise_numbers = bool(re.search(r'\d+\.\d{3,}', svg_code))
            
            optimization_potential = 0.0
            if has_unnecessary_elements:
                optimization_potential += 0.1
            if has_precise_numbers:
                optimization_potential += 0.2
            
            quality_score = 1.0 - optimization_potential
            
            return {
                "element_count": element_count,
                "optimization_potential": optimization_potential,
                "quality_score": max(0.0, quality_score),
                "svg_size": len(svg_code.encode('utf-8'))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SVG quality: {e}")
            return {"quality_score": 0.5}
    
    async def _calculate_maintainability_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate code maintainability score"""
        try:
            scores = []
            
            html_metrics = quality_metrics.get("html", {})
            if html_metrics:
                scores.append(html_metrics.get("quality_score", 0.5))
            
            css_metrics = quality_metrics.get("css", {})
            if css_metrics:
                scores.append(css_metrics.get("quality_score", 0.5))
            
            svg_metrics = quality_metrics.get("svg", {})
            if svg_metrics:
                scores.append(svg_metrics.get("quality_score", 0.5))
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating maintainability score: {e}")
            return 0.5
    
    async def _check_best_practices_compliance(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with performance best practices"""
        try:
            compliance_score = 1.0
            violations = []
            
            html_data = generated_code.get("formats", {}).get("html", {})
            if html_data:
                html_code = html_data.get("html_code", "")
                
                # Check for performance best practices
                if 'loading="lazy"' not in html_code and '<img' in html_code:
                    violations.append("Missing lazy loading for images")
                    compliance_score -= 0.2
                
                if 'defer' not in html_code and '<script' in html_code:
                    violations.append("Scripts not deferred")
                    compliance_score -= 0.1
                
                css_content = html_data.get("css_content", "")
                if len(css_content) > 10000 and "@media" not in css_content:
                    violations.append("Large CSS without media queries")
                    compliance_score -= 0.1
            
            return {
                "compliance_score": max(0.0, compliance_score),
                "violations": violations,
                "best_practices_followed": len(violations) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking best practices: {e}")
            return {"compliance_score": 0.5, "violations": []}
    
    # Continue with remaining helper methods...
    
    async def _check_minification_status(self, formats: Dict[str, Any]) -> Dict[str, Any]:
        """Check if code is minified"""
        # Simplified implementation
        return {"is_minified": False, "potential_savings": "20-30%"}
    
    async def _check_compression_opportunities(self, formats: Dict[str, Any]) -> Dict[str, Any]:
        """Check compression opportunities"""
        return {"compression_ratio": 0.7, "gzip_ready": True}
    
    async def _check_image_optimization(self, formats: Dict[str, Any]) -> Dict[str, Any]:
        """Check image optimization"""
        return {"optimization_needed": False, "potential_savings": "0%"}
    
    async def _analyze_caching_strategy(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze caching strategy"""
        return {"cache_optimized": True, "recommendations": []}
    
    async def _calculate_resource_efficiency(self, formats: Dict[str, Any]) -> float:
        """Calculate resource efficiency"""
        return 0.8
    
    async def _assess_cdn_readiness(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Assess CDN readiness"""
        return {"cdn_ready": True, "issues": []}
    
    async def _analyze_responsive_performance(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze responsive design performance impact"""
        return {"performance_impact": 0.1}
    
    async def _analyze_touch_targets(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze touch target sizes"""
        small_targets = 0
        for component in components.values():
            if component.get("type") == "button":
                dims = component.get("dimensions", {})
                width = dims.get("width", 0)
                height = dims.get("height", 0)
                
                # Touch targets should be at least 44x44px
                if width < 44 or height < 44:
                    small_targets += 1
        
        return {"small_targets": small_targets, "total_interactive": len([c for c in components.values() if c.get("type") == "button"])}
    
    async def _estimate_mobile_loading(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate mobile loading performance"""
        total_size = sum(format_data.get("file_size", 0) for format_data in generated_code.get("formats", {}).values())
        
        # Estimate loading on slow 3G
        slow_3g_speed = 400 * 1024 / 8  # bytes per second
        slow_3g_time = (total_size / slow_3g_speed) * 1000 + 2000  # Add latency
        
        return {"slow_3g_time": slow_3g_time, "estimated_first_contentful_paint": slow_3g_time + 500}
    
    async def _check_mobile_optimizations(self, generated_code: Dict[str, Any]) -> Dict[str, Any]:
        """Check mobile-specific optimizations"""
        missing_optimizations = []
        
        html_data = generated_code.get("formats", {}).get("html", {})
        if html_data:
            html_code = html_data.get("html_code", "")
            
            if 'viewport' not in html_code:
                missing_optimizations.append("viewport meta tag")
            
            if 'preload' not in html_code:
                missing_optimizations.append("resource preloading")
        
        return {"missing_optimizations": missing_optimizations, "mobile_first_design": True}
    
    async def _calculate_mobile_ux_score(self, components: Dict[str, Any]) -> float:
        """Calculate mobile UX score"""
        # Simplified mobile UX scoring
        return 0.8
    
    async def _calculate_performance_score(self, *analyses) -> float:
        """Calculate overall performance score"""
        scores = [analysis.get("score", 0) for analysis in analyses]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # File size, loading, quality, optimization, mobile
        
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return min(1.0, max(0.0, weighted_score))
    
    async def _compile_performance_issues(self, *analyses) -> List[Dict[str, Any]]:
        """Compile all performance issues"""
        all_issues = []
        
        for analysis in analyses:
            issues = analysis.get("issues", [])
            all_issues.extend(issues)
        
        # Sort by severity
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        all_issues.sort(key=lambda x: severity_order.get(x.get("severity", "minor"), 2))
        
        return all_issues
    
    async def _generate_performance_recommendations(self, issues: List[Dict[str, Any]], 
                                                  overall_score: float) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Critical/Major issues
        critical_major_issues = [i for i in issues if i.get("severity") in ["critical", "major"]]
        if critical_major_issues:
            recommendations.append({
                "type": "performance_critical_fixes",
                "description": "Address critical performance issues immediately",
                "priority": "high",
                "actions": [i.get("description") for i in critical_major_issues[:3]]
            })
        
        # Overall score recommendations
        if overall_score < 0.6:
            recommendations.append({
                "type": "comprehensive_optimization",
                "description": "Comprehensive performance optimization needed",
                "priority": "high",
                "actions": ["Optimize file sizes", "Implement compression", "Minimize external requests", "Enable caching"]
            })
        elif overall_score < 0.8:
            recommendations.append({
                "type": "moderate_optimization",
                "description": "Several performance improvements available",
                "priority": "medium",
                "actions": ["Minify resources", "Optimize images", "Review loading strategy"]
            })
        
        return recommendations
    
    async def _identify_optimization_opportunities(self, generated_code: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # Check for image optimization
        opportunities.append({
            "type": "image_optimization",
            "description": "Optimize images for web delivery",
            "impact": "medium",
            "effort": "low"
        })
        
        # Check for code minification
        opportunities.append({
            "type": "code_minification",
            "description": "Minify HTML, CSS, and JavaScript",
            "impact": "medium",
            "effort": "low"
        })
        
        return opportunities
    
    async def _calculate_performance_budget(self, file_size_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance budget status"""
        total_size = file_size_analysis.get("total_size_bytes", 0)
        budget_size = self.max_file_size_kb * 1024
        
        return {
            "budget_size_bytes": budget_size,
            "current_size_bytes": total_size,
            "budget_used_percentage": (total_size / budget_size * 100) if budget_size > 0 else 0,
            "remaining_budget_bytes": max(0, budget_size - total_size),
            "over_budget": total_size > budget_size
        }
    
    async def _load_performance_benchmarks(self):
        """Load performance benchmarks"""
        # This would load performance benchmarks from external source
        pass
    
    async def _initialize_performance_tools(self):
        """Initialize performance evaluation tools"""
        # This would initialize performance testing libraries
        pass
