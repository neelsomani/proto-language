import json
import os
import pytest
from api.core.parser import DarwinParser


@pytest.fixture(scope="session")
def toy_json():
    with open(os.path.join(os.path.dirname(__file__), "../../examples/json_schemas/toy.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def comprehensive_darwin_json():
    """
    Comprehensive Darwin JSON that tests various edge cases:
    - Multiple constructs (DNA and protein)
    - Multiple segments per construct
    - Multiple generators assigned to the same segment
    - Mix of DNA and protein constraints
    - Proper batch_size at optimization level (not generator level)
    """
    return {
        "name": "comprehensive_darwin_test",
        "description": "Comprehensive test covering all generators, constraints, and edge cases",
        "version": "1.0",
        "optimization": {
            "method": "mcmc",
            "num_steps": 5,
            "batch_size": 4,
            "num_candidates": 8,
            "track_step_size": 1,
            "temperature": 1.0,
            "temperature_min": 0.001
        },
        "constructs": [
            {
                "id": "dna_construct",
                "type": "dna",
                "segments": [
                    {
                        "id": "dna_segment1",
                        "initial_sequence": "ATCGATCG"
                    },
                    {
                        "id": "dna_segment2",
                        "initial_sequence": "GCTAGCTA"
                    }
                ]
            },
            {
                "id": "protein_construct",
                "type": "protein",
                "segments": [
                    {
                        "id": "protein_segment1",
                        "initial_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV"
                    }
                ]
            }
        ],
        "constraints": [
            # DNA constraints on segment 1
            {
                "id": "gc_content_constraint_seg1",
                "key": "gc-content",
                "config": {
                    "min_gc": 40.0,
                    "max_gc": 60.0
                },
                "targets": ["dna_segment1"]
            },
            {
                "id": "sequence_length_constraint_seg1",
                "key": "sequence-length",
                "config": {
                    "target_length": 100
                },
                "targets": ["dna_segment1"]
            },
            {
                "id": "max_homopolymer_constraint_seg1",
                "key": "max-homopolymer",
                "config": {
                    "max_length": 3
                },
                "targets": ["dna_segment1"]
            },
            # DNA constraints on segment 2
            {
                "id": "gc_content_constraint_seg2",
                "key": "gc-content",
                "config": {
                    "min_gc": 45.0,
                    "max_gc": 55.0
                },
                "targets": ["dna_segment2"]
            },
            {
                "id": "dinucleotide_frequency_constraint_seg2",
                "key": "dinucleotide-frequency",
                "config": {
                    "min_freq": 0.0,
                    "max_freq": 0.3
                },
                "targets": ["dna_segment2"]
            },
            {
                "id": "tetranucleotide_usage_constraint_seg2",
                "key": "tetranucleotide-usage",
                "config": {
                    "tetranucleotide": "ATCG",
                    "min_tud": 0.5,
                    "max_tud": 2.0
                },
                "targets": ["dna_segment2"]
            },
            # Multi-segment constraint (tests constraint spanning multiple segments)
            {
                "id": "multi_segment_length_constraint",
                "key": "sequence-length",
                "config": {
                    "target_length": 200
                },
                "targets": ["dna_segment1", "dna_segment2"]
            },
            # Protein constraints
            {
                "id": "esmfold_plddt_constraint",
                "key": "esmfold-plddt",
                "config": {
                    "n_replications": 1
                },
                "targets": ["protein_segment1"]
            },
            {
                "id": "esmfold_ptm_constraint",
                "key": "esmfold-ptm",
                "config": {
                    "n_replications": 1
                },
                "targets": ["protein_segment1"]
            },
            {
                "id": "protein_symmetry_ring_constraint",
                "key": "protein-symmetry-ring",
                "config": {
                    "n_replications": 3,
                    "all_to_all_protomer_symmetry": False
                },
                "targets": ["protein_segment1"]
            },
            {
                "id": "protein_globularity_constraint",
                "key": "protein-globularity",
                "config": {
                    "n_replications": 1
                },
                "targets": ["protein_segment1"]
            }
        ],
        "generators": [
            # Generator for dna_segment1
            {
                "id": "dna_seg1_generator",
                "key": "uniform-mutation",
                "config": {
                    "sequence_length": 100
                },
                "targets": ["dna_segment1"]
            },
            # Two generators for dna_segment2 (tests multiple generators on same segment)
            {
                "id": "dna_seg2_generator_1",
                "key": "uniform-mutation",
                "config": {
                    "sequence_length": 50
                },
                "targets": ["dna_segment2"]
            },
            {
                "id": "dna_seg2_generator_2",
                "key": "uniform-mutation",
                "config": {
                    "sequence_length": 50
                },
                "targets": ["dna_segment2"]
            },
            # Generator for protein segment
            {
                "id": "protein_generator",
                "key": "uniform-mutation",
                "config": {
                    "sequence_length": 33
                },
                "targets": ["protein_segment1"]
            }
        ]
    }

@pytest.fixture(scope="session")
def orfipy_mmseqs_darwin_json():
    """
    Darwin JSON that tests the ORFipy + MMseqs constraints.
    Note: This requires a valid MMseqs database path to work properly.
    """
    return {
        "name": "orfipy_mmseqs_darwin_test",
        "description": "Test ORFipy + MMseqs constraints",
        "version": "1.0",
        "optimization": {
            "method": "mcmc",
            "num_steps": 3,
            "track_step_size": 1,
            "temperature": 1.0
        },
        "constructs": [
            {
                "id": "dna_construct",
                "type": "dna",
                "segments": [
                    {
                        "id": "dna_segment1"
                    }
                ]
            }
        ],
        "constraints": [
            {
                "id": "orfipy_mmseqs_gene_hit_count_constraint",
                "key": "orfipy-mmseqs-gene-hit-count",
                "config": {
                    "min_hits": 1,
                    "max_hits": 10,
                    "mmseqs_kwargs": {
                        "database": "/path/to/test/database"
                    }
                },
                "targets": ["dna_segment1"]
            },
            {
                "id": "orfipy_mmseqs_gene_homology_constraint",
                "key": "orfipy-mmseqs-gene-homology",
                "config": {
                    "min_homology": 50.0,
                    "max_homology": 90.0,
                    "mmseqs_kwargs": {
                        "database": "/path/to/test/database"
                    }
                },
                "targets": ["dna_segment1"]
            }
        ],
        "generators": [
            {
                "id": "uniform_mutation_generator",
                "key": "uniform-mutation",
                "config": {
                    "sequence_length": 200
                },
                "targets": ["dna_segment1"]
            }
        ]
    }


def test_darwin_parser_runs(toy_json):
    """
    Test that the basic Darwin parser runs with toy JSON data.
    """
    parser = DarwinParser(toy_json)
    program = parser.parse()
    program.run()
    assert isinstance(program.history, list)
    assert len(program.history) > 0


def test_comprehensive_darwin_parser_parse(comprehensive_darwin_json):
    """
    Test that the comprehensive Darwin JSON can be parsed successfully.
    This test validates:
    - Multiple constructs (DNA and protein)
    - Multiple segments per construct (DNA has 2 segments)
    - Multiple generators on the same segment (dna_segment2 has 2 generators)
    - Multi-segment constraints (constraint spanning both DNA segments)
    - Proper batch_size configuration at optimization level
    """
    parser = DarwinParser(comprehensive_darwin_json)
    program = parser.parse()
    
    # Check that we have the expected number of constructs
    assert len(program.constructs) == 2, f"Expected 2 constructs, got {len(program.constructs)}"
    
    # Check construct types
    assert program.constructs[0].segments[0].sequence_type.value == "dna"
    assert program.constructs[1].segments[0].sequence_type.value == "protein"
    
    # Check that DNA construct has 2 segments
    assert len(program.constructs[0].segments) == 2, f"Expected DNA construct to have 2 segments, got {len(program.constructs[0].segments)}"
    
    # Check that protein construct has 1 segment
    assert len(program.constructs[1].segments) == 1, f"Expected protein construct to have 1 segment, got {len(program.constructs[1].segments)}"
    
    # Check that we have the expected number of generators (4 total: 1 for seg1, 2 for seg2, 1 for protein)
    assert len(program.generators) == 4, f"Expected 4 generators, got {len(program.generators)}"
    
    # Check that we have the expected number of constraints (11 total: 7 DNA + 4 protein)
    assert len(program.constraints) == 11, f"Expected 11 constraints, got {len(program.constraints)}"
    
    # Check that the optimization method is correct
    assert program.iterative_generator_type.__name__ == "MCMCGenerator"
    
    # Check that batch_size is set correctly at the MCMC level (not generator level)
    assert program.ebm.batch_size == 4, f"Expected batch_size=4, got {program.ebm.batch_size}"
    
    # Verify that all generators have the correct batch_size (should be overridden by MCMC)
    for gen in program.generators:
        assert gen.batch_size == 4, f"Generator {gen.__class__.__name__} should have batch_size=4, got {gen.batch_size}"
    
    # Verify that segments have been batched correctly
    for construct in program.constructs:
        for segment in construct.segments:
            assert len(segment.batch_sequences) == 4, f"Each segment should have batch_size=4 sequences, got {len(segment.batch_sequences)}"
            # Verify all batch sequences have content
            for batch_seq in segment.batch_sequences:
                assert len(batch_seq.sequence) > 0, "Batch sequences should not be empty"


def test_parser_registry_completeness():
    """
    Test that all expected generators and constraints are registered in the parser.
    """
    # Check generator registry
    expected_generators = [
        "uniform-mutation",
        "evo2",
        "nim-evo2"
    ]
    
    for gen_key in expected_generators:
        assert gen_key in DarwinParser.generator_registry, f"Generator '{gen_key}' not registered"
    
    # Check constraint registry
    expected_constraints = [
        "gc-content",
        "sequence-length",
        "max-homopolymer",
        "dinucleotide-frequency",
        "tetranucleotide-usage",
        "esmfold-plddt",
        "esmfold-ptm",
        "protein-symmetry-ring",
        "protein-globularity",
        "orfipy-mmseqs-gene-hit-count",
        "orfipy-mmseqs-gene-homology"
    ]
    
    for constraint_key in expected_constraints:
        assert constraint_key in DarwinParser.constraint_registry, f"Constraint '{constraint_key}' not registered"
    
    # Check optimization method registry
    expected_optimization_methods = [
        "mcmc",
        "beam-search"
    ]
    
    for opt_key in expected_optimization_methods:
        assert opt_key in DarwinParser.optimization_method_registry, f"Optimization method '{opt_key}' not registered"


# @pytest.mark.skip(reason="Requires external dependencies (MMseqs database)")
def test_orfipy_mmseqs_parser_parse(orfipy_mmseqs_darwin_json):
    """
    Test that the ORFipy + MMseqs Darwin JSON can be parsed successfully.
    This test is skipped by default as it requires external dependencies.
    """
    parser = DarwinParser(orfipy_mmseqs_darwin_json)
    program = parser.parse()
    
    # Check that we have the expected number of constructs
    assert len(program.constructs) == 1
    
    # Check that we have the expected number of generators
    assert len(program.generators) == 1
    
    # Check that we have the expected number of constraints
    assert len(program.constraints) == 2


def test_parser_error_handling():
    """
    Test that the parser handles unknown generators and constraints gracefully.
    """
    # Test unknown generator
    invalid_generator_json = {
        "name": "invalid_generator_test",
        "description": "Test invalid generator",
        "version": "1.0",
        "optimization": {
            "method": "mcmc",
            "num_steps": 1,
            "track_step_size": 1,
            "temperature": 1.0
        },
        "constructs": [
            {
                "id": "construct1",
                "type": "dna",
                "segments": [{"id": "segment1"}]
            }
        ],
        "constraints": [],
        "generators": [
            {
                "id": "unknown_generator",
                "key": "unknown-generator",
                "config": {},
                "targets": ["segment1"]
            }
        ]
    }
    
    parser = DarwinParser(invalid_generator_json)
    with pytest.raises(ValueError, match="Unknown generator key"):
        parser.parse()
    
    # Test unknown constraint
    invalid_constraint_json = {
        "name": "invalid_constraint_test",
        "description": "Test invalid constraint",
        "version": "1.0",
        "optimization": {
            "method": "mcmc",
            "num_steps": 1,
            "track_step_size": 1,
            "temperature": 1.0
        },
        "constructs": [
            {
                "id": "construct1",
                "type": "dna",
                "segments": [{"id": "segment1"}]
            }
        ],
        "constraints": [
            {
                "id": "unknown_constraint",
                "key": "unknown-constraint",
                "config": {},
                "targets": ["segment1"]
            }
        ],
        "generators": []
    }
    
    parser = DarwinParser(invalid_constraint_json)
    with pytest.raises(ValueError, match="Unknown constraint key"):
        parser.parse()
    
    # Test unknown optimization method
    invalid_optimization_json = {
        "name": "invalid_optimization_test",
        "description": "Test invalid optimization method",
        "version": "1.0",
        "optimization": {
            "method": "unknown-method",
            "num_steps": 1,
            "track_step_size": 1,
            "temperature": 1.0
        },
        "constructs": [
            {
                "id": "construct1",
                "type": "dna",
                "segments": [{"id": "segment1"}]
            }
        ],
        "constraints": [],
        "generators": []
    }
    
    parser = DarwinParser(invalid_optimization_json)
    with pytest.raises(ValueError, match="Unknown optimization method"):
        parser.parse()