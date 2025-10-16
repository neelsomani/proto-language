"""
Base registry pattern for decorator-based component registration.

Provides shared infrastructure for ConstraintRegistry, GeneratorRegistry, and ToolRegistry.
"""

from typing import Any, Dict, Generic, List, Type, TypeVar
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, computed_field


SpecType = TypeVar('SpecType', bound='BaseSpec')


class BaseSpec(BaseModel):
    """
    Base specification for registered components.

    This Pydantic model serves dual purposes:
    1. Internal: Stores component metadata in registry
    2. API: Automatically serialized by FastAPI to JSON

    Subclasses extend this to add component-specific metadata
    """

    # Public fields - exposed in API
    key: str = Field(description="Internal identifier (e.g., 'mcmc', 'gc-content')")
    label: str = Field(description="External UI display name (e.g., 'MCMC Optimizer', 'GC Content Range')")
    description: str = Field(description="Detailed description of component functionality")

    # Private field - excluded from serialization
    config_model: Type[BaseModel] = Field(exclude=True)

    model_config = {
        "extra": "allow",  # Allow subclasses to add fields
        "arbitrary_types_allowed": True,  # Allow Type[BaseModel] in config_model
    }

    # TODO: we can remove this and use standard json schema once client is synced.
    @computed_field
    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Parameter schema in flattened, form-friendly format.

        Transforms Pydantic's standard JSON Schema (which has 'properties' and 'required'
        at the top level) into a flattened format where each parameter name maps directly
        to its schema with 'required' as a boolean field. This format is more convenient
        for client form generators.

        Returns:
            Dict mapping parameter names to their schema definitions:
            {
                "param_name": {
                    "type": "number",
                    "description": "Parameter description",
                    "required": true,
                    "default": 42,
                    "minimum": 0,
                    "maximum": 100,
                    ...
                }
            }

        Note:
            This is a convenience transformation of the standard JSON Schema from
            config_model.model_json_schema(). For the full JSON Schema, access
            spec.config_model.model_json_schema() directly.
        """
        # Get standard JSON Schema from Pydantic
        schema = self.config_model.model_json_schema()
        properties = schema.get("properties", {})
        required_set = set(schema.get("required", []))

        # Transform to parameter-centric format by adding 'required' field
        return {
            param_name: {
                **param_schema,  # Spread all fields from JSON Schema
                "required": param_name in required_set,  # Add required as boolean
            }
            for param_name, param_schema in properties.items()
        }


class BaseRegistry(ABC, Generic[SpecType]):
    """
    Base registry for decorator-based component registration.
    
    Provides discovery, schema generation, and factory methods for constraints,
    generators, and tools. Registration happens at import time via decorators.
    
    Abstract Methods (implemented by subclasses):
    - register(): Decorator to register components
    - list_all(): List all components with metadata
    
    Public Methods:
    - get(): Retrieve component spec by key
    - get_schema(): Get JSON schema for component configuration
    - count(): Get number of registered components
    """
    
    # Subclasses must define their own _registry class variable
    _registry: Dict[str, SpecType] = {}

    @classmethod
    @abstractmethod
    def register(cls, key: str, **kwargs):
        """Decorator to register a component. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.register() must be implemented by subclass")
    
    @classmethod
    @abstractmethod
    def list_all(cls) -> List[SpecType]:
        """List all components as Pydantic models. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.list_all() must be implemented by subclass")
    
    @classmethod
    def get(cls, key: str) -> SpecType:
        """
        Get component spec by key.
        
        Args:
            key: Component identifier
            
        Returns:
            Component specification object
            
        Raises:
            ValueError: If key not found in registry
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) # List all registered keys
            component_type = cls._component_type() # Get the component type (e.g. "constraint", "generator", "tool")
            raise ValueError(f"Unknown {component_type}: '{key}'. Available {component_type}s: {available}")
        return cls._registry[key]
    
    @classmethod
    def get_schema(cls, key: str) -> Dict[str, Any]:
        """
        Get the JSON schema for a specific component's configuration.
        
        The schema includes parameter names, types, defaults, validation rules,
        and descriptions - everything needed to generate a client form.
        
        Args:
            key: Component identifier
            
        Returns:
            JSON Schema dict with structure:
            {
                "properties": {
                    "param_name": {
                        "type": "number",
                        "description": "Parameter description",
                        "default": 42,
                        ...
                    },
                    ...
                },
                "required": ["param1", "param2"],
                "title": "ConfigModelName",
                ...
            }
        
        Examples:
            >>> schema = MyRegistry.get_schema("my_component")
            >>> # Client uses this to generate form fields:
            >>> for param_name, param_info in schema["properties"].items():
            ...     print(f"{param_name}: {param_info['type']}")
        """
        spec = cls.get(key)
        return spec.config_model.model_json_schema()
    
    @classmethod
    def count(cls) -> int:
        """
        Get count of registered components.
        
        Returns:
            Number of registered components
        """
        return len(cls._registry)
    
    @classmethod
    def _check_duplicate(cls, key: str, attempted_component_name: str = None) -> None:
        """
        Check for duplicate registration.
        
        Args:
            key: Component identifier to check
            attempted_component_name: Name of component attempting registration (optional)
            
        Raises:
            ValueError: If key already exists in registry
        """
        if key in cls._registry:
            component_type = cls._component_type()
            existing_spec = cls._registry[key]
            
            # Try to get name from the existing spec label
            existing_name = getattr(existing_spec, 'label', 'unknown')
            
            error_msg = (
                f"{component_type.capitalize()} '{key}' is already registered. "
                f"Duplicate registration is not allowed."
            )
            
            if attempted_component_name:
                error_msg += f"\nExisting: {existing_name}, Attempted: {attempted_component_name}"
            else:
                error_msg += f"\nExisting component: {existing_name}"
            
            raise ValueError(error_msg)
        
    @classmethod
    def _component_type(cls) -> str:
        """
        Get component type derived from registry class name.
        
        Returns:
            Component type string (e.g., 'constraint', 'generator', 'tool')
        """
        return cls.__name__.replace('Registry', '').lower()
