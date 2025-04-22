from abc import ABC, abstractmethod
from typing import Any, List, Dict
from language.sequence import ProgramSequence

class ProgramGenerator(ABC):
    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters: Dict[str, Any] = hyperparameters
        self._is_initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        self._is_initialized = True
        raise NotImplementedError("Subclasses must implement the initialize method.")

    @abstractmethod
    def sample(self) -> List[ProgramSequence]:
        if not self._is_initialized:
            raise RuntimeError(f"Generator {self.__class__.__name__} has not been initialized. Call initialize() first.")
        raise NotImplementedError("Subclasses must implement the sample method.")
    
class MCMCGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)
        
class Evo2Generator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.num_generations: int = hyperparameters.get("num_generations", 100)
        self.population_size: int = hyperparameters.get("population_size", 100)
        self.mutation_rate: float = hyperparameters.get("mutation_rate", 0.1)
        

class SemanticMiningGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        
        
class BindCraftGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)
        self.temperature: float = hyperparameters.get("temperature", 1.0)