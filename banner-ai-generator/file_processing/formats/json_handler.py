"""
JSON Handler

Specialized handler for JSON files including parsing, validation,
transformation, and schema validation capabilities.
"""

import json
import re
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import jsonschema
from structlog import get_logger

logger = get_logger(__name__)


class JSONHandler:
    """
    Comprehensive JSON file handler
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # JSON processing options
        self.pretty_print = config.get("pretty_print", True)
        self.sort_keys = config.get("sort_keys", False)
        self.remove_null_values = config.get("remove_null_values", False)
        self.remove_empty_objects = config.get("remove_empty_objects", False)
        
        # Validation settings
        self.max_depth = config.get("max_depth", 100)
        self.max_size = config.get("max_size", 50 * 1024 * 1024)  # 50MB
        self.validate_encoding = config.get("validate_encoding", True)
        
        # Security settings
        self.allow_comments = config.get("allow_comments", False)
        self.sanitize_strings = config.get("sanitize_strings", True)
        self.max_string_length = config.get("max_string_length", 10000)
        
        # Transformation options
        self.minify_output = config.get("minify_output", False)
        self.ensure_ascii = config.get("ensure_ascii", False)
    
    async def process_json(self, 
                          json_data: Union[str, bytes],
                          options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process JSON data with validation and optimization
        
        Args:
            json_data: JSON content as string or bytes
            options: Processing options
        
        Returns:
            Processing result with parsed and optimized JSON
        """
        try:
            logger.info("Processing JSON file")
            
            # Convert bytes to string if needed
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8')
            
            # Validate file size
            if len(json_data.encode('utf-8')) > self.max_size:
                raise ValueError(f"JSON size exceeds maximum allowed size of {self.max_size} bytes")
            
            # Parse JSON
            parsed_data, parsing_info = await self._parse_json(json_data)
            
            # Extract metadata
            metadata = await self._extract_metadata(json_data, parsed_data)
            
            # Validate structure
            validation_result = await self._validate_structure(parsed_data)
            
            # Apply transformations
            processed_options = options or {}
            transformed_data = await self._transform_data(parsed_data, processed_options)
            
            # Generate optimized JSON
            optimized_json = await self._serialize_json(transformed_data, processed_options)
            
            # Calculate optimization stats
            original_size = len(json_data.encode('utf-8'))
            optimized_size = len(optimized_json.encode('utf-8'))
            compression_ratio = (original_size - optimized_size) / original_size * 100 if original_size > 0 else 0
            
            result = {
                "success": True,
                "original_json": json_data,
                "optimized_json": optimized_json,
                "parsed_data": transformed_data,
                "metadata": metadata,
                "parsing_info": parsing_info,
                "validation": validation_result,
                "optimization_stats": {
                    "original_size": original_size,
                    "optimized_size": optimized_size,
                    "compression_ratio": round(compression_ratio, 2),
                    "size_reduction": original_size - optimized_size
                }
            }
            
            logger.info(f"JSON processing completed: {compression_ratio:.1f}% size reduction")
            return result
            
        except Exception as e:
            logger.error(f"Error processing JSON: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_json": json_data if isinstance(json_data, str) else json_data.decode('utf-8', errors='replace')
            }
    
    async def validate_schema(self, 
                            json_data: Union[str, Dict[str, Any]],
                            schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate JSON against a schema
        
        Args:
            json_data: JSON data to validate
            schema: JSON schema for validation
        
        Returns:
            Validation result
        """
        try:
            logger.info("Validating JSON schema")
            
            # Parse JSON if string
            if isinstance(json_data, str):
                data, _ = await self._parse_json(json_data)
            else:
                data = json_data
            
            # Validate against schema
            validator = jsonschema.Draft7Validator(schema)
            errors = list(validator.iter_errors(data))
            
            validation_result = {
                "valid": len(errors) == 0,
                "errors": [],
                "schema_version": "draft-07"
            }
            
            # Format errors
            for error in errors:
                validation_result["errors"].append({
                    "path": list(error.absolute_path),
                    "message": error.message,
                    "invalid_value": str(error.instance),
                    "schema_path": list(error.schema_path)
                })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating JSON schema: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def transform_json(self, 
                           json_data: Union[str, Dict[str, Any]],
                           transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply transformations to JSON data
        
        Args:
            json_data: JSON data to transform
            transformations: List of transformation configurations
        
        Returns:
            Transformed JSON result
        """
        try:
            logger.info(f"Applying {len(transformations)} transformations to JSON")
            
            # Parse JSON if string
            if isinstance(json_data, str):
                data, _ = await self._parse_json(json_data)
            else:
                data = json_data
            
            # Apply each transformation
            for transformation in transformations:
                transform_type = transformation.get("type", "").lower()
                params = transformation.get("params", {})
                
                if transform_type == "filter_keys":
                    allowed_keys = params.get("keys", [])
                    data = await self._filter_keys(data, allowed_keys)
                
                elif transform_type == "rename_keys":
                    key_mapping = params.get("mapping", {})
                    data = await self._rename_keys(data, key_mapping)
                
                elif transform_type == "remove_null":
                    data = await self._remove_null_values(data)
                
                elif transform_type == "flatten":
                    separator = params.get("separator", "_")
                    data = await self._flatten_json(data, separator)
                
                elif transform_type == "sort_keys":
                    data = await self._sort_keys_recursive(data)
                
                elif transform_type == "convert_types":
                    type_conversions = params.get("conversions", {})
                    data = await self._convert_types(data, type_conversions)
                
                else:
                    logger.warning(f"Unknown transformation type: {transform_type}")
            
            # Serialize result
            transformed_json = await self._serialize_json(data, {})
            
            result = {
                "success": True,
                "transformed_data": data,
                "transformed_json": transformed_json,
                "transformations_applied": len(transformations)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error transforming JSON: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def extract_metadata(self, json_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from JSON
        
        Args:
            json_data: JSON content
        
        Returns:
            Extracted metadata
        """
        try:
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8')
            
            parsed_data, _ = await self._parse_json(json_data)
            return await self._extract_metadata(json_data, parsed_data)
            
        except Exception as e:
            logger.error(f"Error extracting JSON metadata: {e}")
            return {"error": str(e)}
    
    async def validate_json(self, json_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Validate JSON structure and content
        
        Args:
            json_data: JSON content
        
        Returns:
            Validation result
        """
        try:
            issues = []
            warnings = []
            
            # Convert bytes to string if needed
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8')
            
            # Check file size
            size = len(json_data.encode('utf-8'))
            if size > self.max_size:
                issues.append(f"JSON size ({size} bytes) exceeds maximum allowed ({self.max_size} bytes)")
            
            # Attempt to parse
            try:
                parsed_data, parsing_info = await self._parse_json(json_data)
            except Exception as e:
                issues.append(f"Invalid JSON syntax: {e}")
                return {"valid": False, "issues": issues}
            
            # Validate structure
            structure_validation = await self._validate_structure(parsed_data)
            issues.extend(structure_validation.get("issues", []))
            warnings.extend(structure_validation.get("warnings", []))
            
            # Check encoding
            if self.validate_encoding:
                encoding_issues = await self._check_encoding(json_data)
                warnings.extend(encoding_issues)
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "parsing_info": parsing_info,
                "structure_info": structure_validation.get("structure_info", {})
            }
            
        except Exception as e:
            logger.error(f"Error validating JSON: {e}")
            return {"valid": False, "error": str(e)}
    
    async def minify_json(self, json_data: Union[str, Dict[str, Any]]) -> str:
        """
        Minify JSON by removing whitespace
        
        Args:
            json_data: JSON data to minify
        
        Returns:
            Minified JSON string
        """
        try:
            if isinstance(json_data, str):
                data, _ = await self._parse_json(json_data)
            else:
                data = json_data
            
            return json.dumps(data, separators=(',', ':'), ensure_ascii=self.ensure_ascii)
            
        except Exception as e:
            logger.error(f"Error minifying JSON: {e}")
            raise
    
    async def prettify_json(self, json_data: Union[str, Dict[str, Any]], indent: int = 2) -> str:
        """
        Prettify JSON with proper indentation
        
        Args:
            json_data: JSON data to prettify
            indent: Number of spaces for indentation
        
        Returns:
            Prettified JSON string
        """
        try:
            if isinstance(json_data, str):
                data, _ = await self._parse_json(json_data)
            else:
                data = json_data
            
            return json.dumps(
                data, 
                indent=indent, 
                sort_keys=self.sort_keys,
                ensure_ascii=self.ensure_ascii
            )
            
        except Exception as e:
            logger.error(f"Error prettifying JSON: {e}")
            raise
    
    async def _parse_json(self, json_data: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse JSON and return data with parsing information"""
        try:
            start_time = datetime.now()
            
            # Remove comments if allowed
            if self.allow_comments:
                json_data = await self._remove_json_comments(json_data)
            
            # Parse JSON
            parsed_data = json.loads(json_data)
            
            end_time = datetime.now()
            parsing_time = (end_time - start_time).total_seconds() * 1000  # milliseconds
            
            parsing_info = {
                "parsing_time_ms": round(parsing_time, 2),
                "has_comments": self.allow_comments and "/*" in json_data or "//" in json_data,
                "parsed_at": end_time.isoformat()
            }
            
            return parsed_data, parsing_info
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON syntax: {e}")
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            raise
    
    async def _extract_metadata(self, json_string: str, parsed_data: Any) -> Dict[str, Any]:
        """Extract comprehensive metadata from JSON"""
        try:
            metadata = {
                "file_size": len(json_string.encode('utf-8')),
                "character_count": len(json_string),
                "line_count": json_string.count('\n') + 1,
                "data_type": type(parsed_data).__name__,
                "encoding": "utf-8"  # Assuming UTF-8
            }
            
            # Calculate hashes
            metadata["md5_hash"] = hashlib.md5(json_string.encode('utf-8')).hexdigest()
            metadata["sha256_hash"] = hashlib.sha256(json_string.encode('utf-8')).hexdigest()
            
            # Analyze structure
            structure_info = await self._analyze_structure(parsed_data)
            metadata.update(structure_info)
            
            # Count different value types
            type_counts = await self._count_value_types(parsed_data)
            metadata["value_types"] = type_counts
            
            # Find patterns
            patterns = await self._find_patterns(parsed_data)
            metadata["patterns"] = patterns
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e)}
    
    async def _validate_structure(self, data: Any) -> Dict[str, Any]:
        """Validate JSON structure"""
        try:
            issues = []
            warnings = []
            
            # Check depth
            depth = await self._calculate_depth(data)
            if depth > self.max_depth:
                issues.append(f"JSON depth ({depth}) exceeds maximum allowed ({self.max_depth})")
            elif depth > self.max_depth * 0.8:
                warnings.append(f"JSON depth ({depth}) is approaching maximum limit")
            
            # Check for circular references (shouldn't happen in valid JSON, but check anyway)
            if await self._has_circular_references(data):
                issues.append("Circular references detected")
            
            # Check string lengths if sanitization is enabled
            if self.sanitize_strings:
                long_strings = await self._find_long_strings(data)
                if long_strings:
                    warnings.extend([f"Long string found at path: {path}" for path in long_strings])
            
            # Analyze structure complexity
            complexity = await self._calculate_complexity(data)
            
            return {
                "issues": issues,
                "warnings": warnings,
                "structure_info": {
                    "depth": depth,
                    "complexity_score": complexity,
                    "total_elements": await self._count_total_elements(data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating structure: {e}")
            return {"issues": [f"Structure validation error: {e}"]}
    
    async def _transform_data(self, data: Any, options: Dict[str, Any]) -> Any:
        """Apply basic transformations to data"""
        try:
            transformed = data
            
            # Remove null values if requested
            if options.get("remove_null", self.remove_null_values):
                transformed = await self._remove_null_values(transformed)
            
            # Remove empty objects if requested
            if options.get("remove_empty", self.remove_empty_objects):
                transformed = await self._remove_empty_objects(transformed)
            
            # Sort keys if requested
            if options.get("sort_keys", self.sort_keys):
                transformed = await self._sort_keys_recursive(transformed)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return data
    
    async def _serialize_json(self, data: Any, options: Dict[str, Any]) -> str:
        """Serialize data back to JSON string"""
        try:
            serialize_options = {
                "ensure_ascii": self.ensure_ascii,
                "sort_keys": options.get("sort_keys", self.sort_keys)
            }
            
            if options.get("minify", self.minify_output):
                serialize_options["separators"] = (',', ':')
            else:
                serialize_options["indent"] = options.get("indent", 2 if self.pretty_print else None)
            
            return json.dumps(data, **serialize_options)
            
        except Exception as e:
            logger.error(f"Error serializing JSON: {e}")
            raise
    
    async def _remove_json_comments(self, json_string: str) -> str:
        """Remove comments from JSON string"""
        try:
            # Remove single-line comments
            json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
            
            # Remove multi-line comments (simple approach)
            json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
            
            return json_string
            
        except Exception as e:
            logger.error(f"Error removing comments: {e}")
            return json_string
    
    async def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure"""
        try:
            if isinstance(data, dict):
                return {
                    "root_type": "object",
                    "key_count": len(data),
                    "keys": list(data.keys())[:20],  # First 20 keys
                    "nested_objects": sum(1 for v in data.values() if isinstance(v, dict)),
                    "nested_arrays": sum(1 for v in data.values() if isinstance(v, list))
                }
            elif isinstance(data, list):
                return {
                    "root_type": "array",
                    "length": len(data),
                    "item_types": list(set(type(item).__name__ for item in data[:100])),  # Types of first 100 items
                    "nested_objects": sum(1 for item in data if isinstance(item, dict)),
                    "nested_arrays": sum(1 for item in data if isinstance(item, list))
                }
            else:
                return {
                    "root_type": type(data).__name__,
                    "value": str(data)[:100]  # First 100 chars
                }
                
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return {"error": str(e)}
    
    async def _count_value_types(self, data: Any) -> Dict[str, int]:
        """Count occurrences of different value types"""
        try:
            type_counts = {}
            
            def count_recursive(obj):
                obj_type = type(obj).__name__
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                
                if isinstance(obj, dict):
                    for value in obj.values():
                        count_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count_recursive(item)
            
            count_recursive(data)
            return type_counts
            
        except Exception as e:
            logger.error(f"Error counting value types: {e}")
            return {}
    
    async def _find_patterns(self, data: Any) -> Dict[str, Any]:
        """Find patterns in JSON data"""
        try:
            patterns = {
                "repeated_keys": [],
                "common_values": [],
                "array_patterns": []
            }
            
            # Find repeated keys in objects
            if isinstance(data, dict):
                key_counts = {}
                
                def count_keys(obj):
                    if isinstance(obj, dict):
                        for key in obj.keys():
                            key_counts[key] = key_counts.get(key, 0) + 1
                        for value in obj.values():
                            count_keys(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            count_keys(item)
                
                count_keys(data)
                patterns["repeated_keys"] = [
                    {"key": k, "count": v} for k, v in sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding patterns: {e}")
            return {}
    
    async def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of JSON structure"""
        try:
            if isinstance(data, dict):
                if not data:
                    return current_depth
                return max(await self._calculate_depth(value, current_depth + 1) for value in data.values())
            elif isinstance(data, list):
                if not data:
                    return current_depth
                return max(await self._calculate_depth(item, current_depth + 1) for item in data)
            else:
                return current_depth
                
        except Exception as e:
            logger.error(f"Error calculating depth: {e}")
            return current_depth
    
    async def _has_circular_references(self, data: Any, visited: set = None) -> bool:
        """Check for circular references"""
        # Note: This is more relevant for object references, 
        # JSON shouldn't have circular refs by design
        return False
    
    async def _find_long_strings(self, data: Any, path: str = "") -> List[str]:
        """Find strings that exceed maximum length"""
        try:
            long_strings = []
            
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    long_strings.extend(await self._find_long_strings(value, current_path))
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    long_strings.extend(await self._find_long_strings(item, current_path))
            elif isinstance(data, str) and len(data) > self.max_string_length:
                long_strings.append(path)
            
            return long_strings
            
        except Exception as e:
            logger.error(f"Error finding long strings: {e}")
            return []
    
    async def _calculate_complexity(self, data: Any) -> float:
        """Calculate complexity score for JSON structure"""
        try:
            complexity = 0.0
            
            def calc_recursive(obj, depth_factor=1.0):
                nonlocal complexity
                
                if isinstance(obj, dict):
                    complexity += len(obj) * depth_factor
                    for value in obj.values():
                        calc_recursive(value, depth_factor * 1.1)
                elif isinstance(obj, list):
                    complexity += len(obj) * depth_factor * 0.8
                    for item in obj:
                        calc_recursive(item, depth_factor * 1.05)
                else:
                    complexity += depth_factor * 0.5
            
            calc_recursive(data)
            return round(complexity, 2)
            
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return 0.0
    
    async def _count_total_elements(self, data: Any) -> int:
        """Count total number of elements in JSON structure"""
        try:
            count = 0
            
            def count_recursive(obj):
                nonlocal count
                count += 1
                
                if isinstance(obj, dict):
                    for value in obj.values():
                        count_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count_recursive(item)
            
            count_recursive(data)
            return count
            
        except Exception as e:
            logger.error(f"Error counting elements: {e}")
            return 0
    
    async def _check_encoding(self, json_string: str) -> List[str]:
        """Check for encoding issues"""
        warnings = []
        
        try:
            # Check for non-ASCII characters
            if not json_string.isascii():
                warnings.append("Contains non-ASCII characters")
            
            # Check for common encoding issues
            if '�' in json_string:
                warnings.append("Contains replacement characters (possible encoding issue)")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error checking encoding: {e}")
            return ["Encoding check failed"]
    
    # Transformation helper methods
    async def _remove_null_values(self, data: Any) -> Any:
        """Remove null/None values from data"""
        if isinstance(data, dict):
            return {k: await self._remove_null_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [await self._remove_null_values(item) for item in data if item is not None]
        else:
            return data
    
    async def _remove_empty_objects(self, data: Any) -> Any:
        """Remove empty objects and arrays"""
        if isinstance(data, dict):
            filtered = {}
            for k, v in data.items():
                cleaned_v = await self._remove_empty_objects(v)
                if cleaned_v or cleaned_v == 0 or cleaned_v == "" or cleaned_v is False:
                    filtered[k] = cleaned_v
            return filtered
        elif isinstance(data, list):
            filtered = []
            for item in data:
                cleaned_item = await self._remove_empty_objects(item)
                if cleaned_item or cleaned_item == 0 or cleaned_item == "" or cleaned_item is False:
                    filtered.append(cleaned_item)
            return filtered
        else:
            return data
    
    async def _sort_keys_recursive(self, data: Any) -> Any:
        """Sort keys recursively in dictionaries"""
        if isinstance(data, dict):
            return {k: await self._sort_keys_recursive(v) for k, v in sorted(data.items())}
        elif isinstance(data, list):
            return [await self._sort_keys_recursive(item) for item in data]
        else:
            return data
    
    async def _filter_keys(self, data: Any, allowed_keys: List[str]) -> Any:
        """Filter dictionary keys"""
        if isinstance(data, dict):
            return {k: await self._filter_keys(v, allowed_keys) for k, v in data.items() if k in allowed_keys}
        elif isinstance(data, list):
            return [await self._filter_keys(item, allowed_keys) for item in data]
        else:
            return data
    
    async def _rename_keys(self, data: Any, key_mapping: Dict[str, str]) -> Any:
        """Rename dictionary keys"""
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                new_key = key_mapping.get(k, k)
                result[new_key] = await self._rename_keys(v, key_mapping)
            return result
        elif isinstance(data, list):
            return [await self._rename_keys(item, key_mapping) for item in data]
        else:
            return data
    
    async def _flatten_json(self, data: Any, separator: str = "_", prefix: str = "") -> Dict[str, Any]:
        """Flatten nested JSON structure"""
        result = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.update(await self._flatten_json(value, separator, new_key))
                else:
                    result[new_key] = value
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{prefix}{separator}{i}" if prefix else str(i)
                if isinstance(item, (dict, list)):
                    result.update(await self._flatten_json(item, separator, new_key))
                else:
                    result[new_key] = item
        else:
            result[prefix] = data
        
        return result
    
    async def _convert_types(self, data: Any, conversions: Dict[str, str]) -> Any:
        """Convert data types based on mapping"""
        if isinstance(data, dict):
            return {k: await self._convert_types(v, conversions) for k, v in data.items()}
        elif isinstance(data, list):
            return [await self._convert_types(item, conversions) for item in data]
        else:
            current_type = type(data).__name__
            target_type = conversions.get(current_type)
            
            if target_type:
                try:
                    if target_type == "str":
                        return str(data)
                    elif target_type == "int":
                        return int(data)
                    elif target_type == "float":
                        return float(data)
                    elif target_type == "bool":
                        return bool(data)
                except (ValueError, TypeError):
                    pass
            
            return data
