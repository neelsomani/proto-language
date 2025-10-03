"""
Chained Generator

Extracted from generator.py for better code organization.
"""

from typing import Any, Dict, List, Optional, final
import copy
import time
import json

from ..base import IterativeGenerator, Construct


@final
class ChainedGenerator:
    """
    Pipeline orchestrator that chains multiple IterativeGenerator stages in sequence.

    This meta-generator runs a series of independent IterativeGenerator objects (stages)
    in order, automatically propagating sequences between stages. Each stage can use
    different optimization strategies, constraints, and generators, enabling
    multi-stage design workflows.

    Examples:
        Creating a two-stage optimization pipeline:
        >>> from proto_language.language.generator import ChainedGenerator, MCMCGenerator, UniformMutationGenerator
        >>> from proto_language.language.constraint import gc_content_constraint
        >>> from proto_language.language.base import Construct, Segment
        >>> from proto_language.language.base.sequence import SequenceType
        >>>
        >>> # Create segments and generators
        >>> segment1 = Segment(sequence="A" * 50, sequence_type=SequenceType.DNA)
        >>> segment2 = Segment(sequence="A" * 50, sequence_type=SequenceType.DNA)
        >>>
        >>> gen1 = UniformMutationGenerator(sequence_length=50, batch_size=3)
        >>> gen2 = UniformMutationGenerator(sequence_length=50, batch_size=3)
        >>> gen1.assign(segment1)
        >>> gen2.assign(segment2)
        >>>
        >>> # Stage 1: Increase GC content
        >>> construct1 = Construct([segment1])
        >>> constraint1 = Constraint(
        ...     inputs=[segment1],
        ...     scoring_function=gc_content_constraint,
        ...     scoring_function_config={"min_gc": 70.0, "max_gc": 90.0}
        ... )
        >>> stage1 = MCMCGenerator(
        ...     constructs=[construct1],
        ...     generators=[gen1],
        ...     constraints=[constraint1],
        ...     num_steps=100,
        ...     verbose=False
        ... )
        >>>
        >>> # Stage 2: Further optimize with different constraints
        >>> construct2 = Construct([segment2])
        >>> constraint2 = Constraint(
        ...     inputs=[segment2],
        ...     scoring_function=gc_content_constraint,
        ...     scoring_function_config={"min_gc": 80.0, "max_gc": 95.0}
        ... )
        >>> stage2 = MCMCGenerator(
        ...     constructs=[construct2],
        ...     generators=[gen2],
        ...     constraints=[constraint2],
        ...     num_steps=50,
        ...     verbose=False
        ... )
        >>>
        >>> # Create and run the chained pipeline
        >>> pipeline = ChainedGenerator([stage1, stage2], verbose=True)
        >>> pipeline.run()
        >>>
        >>> # Get results
        >>> final_sequences = pipeline.get_final_sequences()
        >>> stage_results = pipeline.get_stage_results()
        >>> print(f"Generated {len(final_sequences)} sequences")
        >>> print(f"Stage 1 execution time: {stage_results[0]['execution_time']:.2f}s")

        Using different generator types per stage:
        >>> # Stage 1: Point mutations with MCMC
        >>> # Stage 2: Structure-based generation (if available)
        >>> # Stage 3: Final refinement with different constraints

        Accessing detailed metadata:
        >>> pipeline = ChainedGenerator([stage1, stage2])
        >>> pipeline.run()
        >>> metadata = pipeline.get_stage_metadata()
        >>> for stage_meta in metadata:
        ...     print(f"Stage {stage_meta['stage']}: {stage_meta['stage_type']}")
        ...     print(f"Execution time: {stage_meta['execution_summary']['execution_time']:.2f}s")

    Note:
        Unlike IterativeGenerator subclasses, ChainedGenerator is a pipeline orchestrator
        that manages the execution flow between multiple generators rather than generating
        sequences itself. It does not inherit from IterativeGenerator.
    """

    def __init__(
        self,
        generator_stages: List[IterativeGenerator],
        verbose: bool = True,
        capture_metadata: bool = True,
    ) -> None:
        """
        Initialize the chained generator.

        Args:
            generator_stages: List of IterativeGenerator objects to run in sequence.
                             Each stage must have the same batch_size and compatible constructs.
                             Individual stages can have their own energy_threshold parameters for early stopping.
            verbose: Whether to print progress information during execution.
            capture_metadata: Whether to capture detailed metadata from each stage.
                             If False, only basic stage results are stored.
        """
        if not generator_stages:
            raise ValueError("At least one generator stage must be provided")
        
        self.generator_stages = generator_stages
        self.verbose = verbose
        self.capture_metadata = capture_metadata
        self.stage_results = []
        self._execution_start_time = None

        # Validate stage compatibility
        self._validate_stages()
        
    def _validate_stages(self) -> None:
        """
        Validate that all stages are compatible.
        
        Checks:
            - All stages have the same batch_size
            - All stages have compatible construct structures
            - All stages are IterativeGenerator instances
            
        Raises:
            ValueError: If validation fails.
        """
        # Check that all stages are IterativeGenerator instances
        for i, stage in enumerate(self.generator_stages):
            if not isinstance(stage, IterativeGenerator):
                raise ValueError(
                    f"Stage {i} must be an IterativeGenerator, got {type(stage).__name__}"
                )
        
        # Check batch size consistency by looking at each stage's batch_size
        batch_sizes = []
        for stage in self.generator_stages:
            # IterativeGenerator has a batch_size attribute
            batch_sizes.append(stage.batch_size)
        
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"All stages must have the same batch_size. Found: {batch_sizes}"
            )
        
        # Check construct structure compatibility
        first_stage = self.generator_stages[0]
        for i, stage in enumerate(self.generator_stages[1:], 1):
            if len(stage.constructs) != len(first_stage.constructs):
                raise ValueError(
                    f"Stage {i} must have the same number of constructs as stage 0. "
                    f"Found: {len(stage.constructs)} vs {len(first_stage.constructs)}"
                )
            
            # Check that construct structures match
            for j, (first_construct, stage_construct) in enumerate(zip(first_stage.constructs, stage.constructs)):
                if len(first_construct.segments) != len(stage_construct.segments):
                    raise ValueError(
                        f"Stage {i} construct {j} must have the same number of segments as stage 0. "
                        f"Found: {len(stage_construct.segments)} vs {len(first_construct.segments)}"
                    )
                
                # Check sequence types and lengths
                for k, (first_segment, stage_segment) in enumerate(zip(first_construct.segments, stage_construct.segments)):
                    if first_segment.sequence_type != stage_segment.sequence_type:
                        raise ValueError(
                            f"Stage {i} construct {j} segment {k} must have the same sequence type as stage 0. "
                            f"Found: {stage_segment.sequence_type} vs {first_segment.sequence_type}"
                        )
    
    def run(self) -> None:
        """
        Execute all generator stages in sequence.
        
        For each stage:
        1. Runs the stage's sample() method (which may execute multiple internal steps)
        2. Captures the stage's final results and metadata
        3. Propagates sequences to the next stage as starting points
        4. Stores stage results in self.stage_results
        """
        if self.verbose:
            print(f"Starting chained generation with {len(self.generator_stages)} stages")
        
        self._execution_start_time = time.time()
        self.stage_results = []
        
        for i, stage in enumerate(self.generator_stages):
            stage_start_time = time.time()
            
            if self.verbose:
                print(f"\n--- Stage {i+1}/{len(self.generator_stages)} ---")
                print(f"Running {stage.__class__.__name__}")
                if hasattr(stage, 'num_steps'):
                    print(f"Steps: {stage.num_steps}")
                if hasattr(stage, 'temperature'):
                    print(f"Temperature: {stage.temperature}")
            
            # Run this stage
            stage.sample()
            
            # Capture stage results and metadata
            stage_result = self._capture_stage_result(i, stage, stage_start_time)
            self.stage_results.append(stage_result)

            if self.verbose:
                execution_time = stage_result['execution_time']
                final_energy = stage_result['final_energy']
                best_energy = min(final_energy) if final_energy else None
                print(f"Stage {i+1} completed in {execution_time:.2f}s")
                print(f"Best energy: {best_energy:.4f}")

            # Propagate sequences to next stage (if not the last stage)
            if i < len(self.generator_stages) - 1:
                next_stage = self.generator_stages[i + 1]
                self._propagate_sequences_to_next_stage(stage, next_stage)
        
        total_time = time.time() - self._execution_start_time
        if self.verbose:
            print(f"\n--- Chained Generation Complete ---")
            print(f"Total execution time: {total_time:.2f}s")
            if self.stage_results:
                final_energy = self.stage_results[-1]['final_energy']
                if final_energy:
                    print(f"Final best energy: {min(final_energy):.4f}")
                else:
                    print("Final energy: No energy scores available")
            else:
                print("Final energy: No stages completed")
    
    def _capture_stage_result(
        self, 
        stage_index: int, 
        stage: IterativeGenerator, 
        stage_start_time: float
    ) -> Dict:
        """
        Capture comprehensive results and metadata from a completed stage.
        
        Args:
            stage_index: Index of the stage.
            stage: The completed stage.
            stage_start_time: When the stage started execution.
            
        Returns:
            Dictionary containing stage results and metadata.
        """
        execution_time = time.time() - stage_start_time
        
        # Get final constructs and energy
        constructs = copy.deepcopy(stage.constructs)
        stage.score_energy() if hasattr(stage, 'score_energy') else []
        
        # Capture stage configuration
        stage_config = self._extract_stage_config(stage)
        
        # Capture outputs metadata
        outputs_metadata = []
        if self.capture_metadata:
            try:
                outputs = stage.get_generator_outputs()
                for output in outputs:
                    per_batch_meta = []
                    for seq in output.batch_sequences:
                        # Deep copy metadata to avoid reference issues
                        seq_metadata = copy.deepcopy(seq._metadata) if hasattr(seq, '_metadata') else {}
                        per_batch_meta.append(seq_metadata)
                    outputs_metadata.append(per_batch_meta)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not capture outputs metadata for stage {stage_index}: {e}")
                outputs_metadata = []
        
        return {
            'stage': stage_index,
            'stage_type': stage.__class__.__name__,
            'constructs': constructs,
            'final_energy': stage.energy_scores,
            'num_steps': getattr(stage, 'num_steps', None),
            'execution_time': execution_time,
            'stage_config': stage_config,
            'outputs_metadata': outputs_metadata,
        }
    
    def _extract_stage_config(self, stage: IterativeGenerator) -> Dict:
        """
        Extract configuration parameters from a stage.
        
        Args:
            stage: The stage to extract configuration from.
            
        Returns:
            Dictionary of configuration parameters.
        """
        config = {}
        
        # Common IterativeGenerator parameters
        common_params = ['temperature', 'temperature_min', 'track_step_size', 'verbose']
        for param in common_params:
            if hasattr(stage, param):
                config[param] = getattr(stage, param)
        
        # Generator-specific parameters
        if hasattr(stage, 'generators'):
            config['generator_types'] = [gen.__class__.__name__ for gen in stage.generators]
        
        if hasattr(stage, 'constraints'):
            config['constraint_types'] = [constraint.__class__.__name__ for constraint in stage.constraints]
        
        if hasattr(stage, 'constraint_weights'):
            config['constraint_weights'] = stage.constraint_weights
        
        # Evo2Generator specific
        if hasattr(stage, 'evo2_type'):
            config['evo2_type'] = stage.evo2_type
        
        # ESM2Generator specific
        if hasattr(stage, 'esm2_type'):
            config['esm2_type'] = stage.esm2_type

        return config
    
    def _propagate_sequences_to_next_stage(
        self, 
        current_stage: IterativeGenerator, 
        next_stage: IterativeGenerator
    ) -> None:
        """
        Copy final sequences from current stage to next stage's constructs.
        
        Args:
            current_stage: The stage that just completed.
            next_stage: The stage that will run next.
        """
        try:
            # Get the final sequences from current stage
            current_outputs = current_stage.get_generator_outputs()
            
            # Update next stage's constructs with these sequences
            next_outputs = next_stage.get_generator_outputs()
            
            # Copy sequences from current stage outputs to next stage outputs
            for curr_out, next_out in zip(current_outputs, next_outputs):
                for i, seq in enumerate(curr_out.batch_sequences):
                    if i < len(next_out.batch_sequences):
                        next_out.batch_sequences[i].sequence = seq.sequence
                        # Preserve existing metadata and add new metadata
                        if hasattr(seq, '_metadata') and hasattr(next_out.batch_sequences[i], '_metadata'):
                            next_out.batch_sequences[i]._metadata.update(seq._metadata.copy())
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not propagate sequences between stages: {e}")
    
    def get_final_constructs(self) -> List[Construct]:
        """
        Get the final constructs after all stages have completed.
        
        Returns:
            List of Construct objects from the final stage.
            
        Raises:
            RuntimeError: If run() hasn't been called yet.
        """
        if not self.stage_results:
            raise RuntimeError("run() must be called before accessing final constructs")
        return self.stage_results[-1]['constructs']
    
    def get_final_sequences(self) -> List[str]:
        """
        Get the final sequences from all constructs.
        
        Returns:
            List of concatenated sequences from the final stage.
            
        Raises:
            RuntimeError: If run() hasn't been called yet.
        """
        constructs = self.get_final_constructs()
        sequences = []
        for construct in constructs:
            # Concatenate all segment sequences
            full_sequence = ""
            for segment in construct.segments:
                # Access sequence through batch_sequences
                if hasattr(segment, 'batch_sequences') and segment.batch_sequences:
                    full_sequence += segment.batch_sequences[0].sequence
                elif hasattr(segment, 'sequence'):
                    # Fallback for segments with direct sequence attribute
                    full_sequence += segment.sequence
                else:
                    # If no sequence found, add empty string
                    full_sequence += ""
            sequences.append(full_sequence)
        return sequences
    
    def get_stage_results(self) -> List[Dict]:
        """
        Get comprehensive results from all stages.
        
        Returns:
            List of dictionaries, one per stage, containing:
            - 'stage': stage index (0-based)
            - 'stage_type': class name of the stage
            - 'constructs': final constructs from this stage
            - 'final_energy': final energy scores from this stage
            - 'num_steps': number of steps this stage executed
            - 'execution_time': time taken for this stage
            - 'outputs_metadata': detailed metadata per output and batch element
            - 'stage_config': configuration parameters of this stage
        """
        return self.stage_results.copy()
    
    def get_stage_metadata(self) -> List[Dict]:
        """
        Get metadata-focused information from all stages.
        
        Returns:
            List of dictionaries with metadata per stage:
            - 'stage': stage index
            - 'stage_type': class name
            - 'outputs_metadata': List[List[Dict]] with metadata per output and batch
            - 'execution_summary': high-level execution stats
        """
        metadata = []
        for result in self.stage_results:
            stage_meta = {
                'stage': result['stage'],
                'stage_type': result['stage_type'],
                'outputs_metadata': result.get('outputs_metadata', []),
                'execution_summary': {
                    'execution_time': result['execution_time'],
                    'num_steps': result.get('num_steps'),
                    'final_energy': result.get('final_energy')
                }
            }
            metadata.append(stage_meta)
        return metadata
    
    def get_stage(self, stage_index: int) -> Optional[IterativeGenerator]:
        """
        Get a specific stage by index.
        
        Args:
            stage_index: Index of the stage to retrieve.
            
        Returns:
            The IterativeGenerator at the specified index, or None if invalid.
        """
        if 0 <= stage_index < len(self.generator_stages):
            return self.generator_stages[stage_index]
        return None
    
    def get_stage_result(self, stage_index: int) -> Optional[Dict]:
        """
        Get results from a specific stage.
        
        Args:
            stage_index: Index of the stage to retrieve results from.
            
        Returns:
            Stage results dictionary, or None if stage hasn't run or index is invalid.
        """
        if 0 <= stage_index < len(self.stage_results):
            return self.stage_results[stage_index]
        return None
    
    def get_execution_summary(self) -> Dict:
        """
        Get a high-level summary of the entire pipeline execution.
        
        Returns:
            Dictionary with:
            - 'total_stages': number of stages
            - 'total_execution_time': sum of all stage execution times
            - 'final_energy': best energy from final stage
            - 'energy_progression': list of best energies from each stage
            - 'stage_types': list of stage class names
        """
        if not self.stage_results:
            return {
                'total_stages': len(self.generator_stages),
                'total_execution_time': 0.0,
                'final_energy': None,
                'energy_progression': [],
                'stage_types': [stage.__class__.__name__ for stage in self.generator_stages]
            }
        
        total_execution_time = sum(result['execution_time'] for result in self.stage_results)
        energy_progression = []
        
        for result in self.stage_results:
            if result['final_energy']:
                best_energy = min(result['final_energy'])
                energy_progression.append(best_energy)
            else:
                energy_progression.append(None)
        
        final_energy = energy_progression[-1] if energy_progression else None
        
        return {
            'total_stages': len(self.generator_stages),
            'total_execution_time': total_execution_time,
            'final_energy': final_energy,
            'energy_progression': energy_progression,
            'stage_types': [result['stage_type'] for result in self.stage_results]
        }
    
    def get_energy_progression(self) -> List[float]:
        """
        Get the progression of best energies across all stages.
        
        Returns:
            List of best energy values, one per stage.
        """
        if not self.stage_results:
            return []
        
        energy_progression = []
        for result in self.stage_results:
            if result['final_energy']:
                best_energy = min(result['final_energy'])
                energy_progression.append(best_energy)
            else:
                energy_progression.append(None)
        
        return energy_progression
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export all results to a file.
        
        Args:
            filepath: Path to save the results.
            format: Export format ('json', 'pickle').
        """
        if not self.stage_results:
            raise RuntimeError("No results to export. Call run() first.")
        
        if format.lower() == 'json':
            # Convert constructs to serializable format
            exportable_results = []
            for result in self.stage_results:
                exportable_result = result.copy()
                # Convert constructs to sequence strings for JSON export
                exportable_result['constructs'] = []
                for construct in result['constructs']:
                    construct_sequences = []
                    for seg in construct.segments:
                        # Access sequence through batch_sequences
                        if hasattr(seg, 'batch_sequences') and seg.batch_sequences:
                            construct_sequences.append(seg.batch_sequences[0].sequence)
                        elif hasattr(seg, 'sequence'):
                            construct_sequences.append(seg.sequence)
                        else:
                            construct_sequences.append("")
                    exportable_result['constructs'].append(construct_sequences)
                exportable_results.append(exportable_result)
            
            with open(filepath, 'w') as f:
                json.dump(exportable_results, f, indent=2)
                
        elif format.lower() == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.stage_results, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.")

