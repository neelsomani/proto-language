from typing import Any, Dict, List, Optional, Tuple

from .base import ProgramIterativeGenerator, ProgramSequence


class Program:
    """
    Small wrapper class that samples from an EBM-typed generator. Can keep track of state across
    multiple calls to `sample()`.

    TODO(@brianhie): Decide if this class is even needed.
    """
    def __init__(
        self,
        ebm: ProgramIterativeGenerator,
        **kwargs: Any,
    ) -> None:
        self.ebm: ProgramIterativeGenerator = ebm
        self.config: Dict[str, Any] = kwargs
    
    def _validate_ebm(self) -> None:
        if not isinstance(self.ebm, ProgramIterativeGenerator):
            raise ValueError("ebm must be a ProgramIterativeGenerator")

    def run(self) -> Dict[str, Any]:
        """
        Run MCMC on an EBM generator while keeping track of state.
        """
        # Get initial state for printing
        initial_sequence = tuple(output.sequence for output in self.ebm.get_outputs())
        initial_energy = self.ebm.score_energy()

        print(f"Initial sequence: {initial_sequence}")
        print(f"Initial energy: {initial_energy:.4f}")

        # Run MCMC
        data = self.ebm.sample()
        sequence_history = data["sequence_history"]
        energy_history = data["energy_history"]
        steps_history = data["steps_history"]

        # Get the final sequence.
        final_sequences = tuple(output.sequence for output in self.ebm.get_outputs())

        print(f"Final sequence: {final_sequences}")
        print(f"Final energy: {self.ebm.score_energy():.4f}")

        return {
            "sequence_history": sequence_history,
            "energy_history": energy_history,
            "steps_history": steps_history,
        }

