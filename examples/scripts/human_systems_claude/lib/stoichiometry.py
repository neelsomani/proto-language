"""
Stoichiometry inference for human protein complexes.

This module provides stoichiometry information based on:
1. Naming conventions in complex labels (e.g., _trimer, _tetramer)
2. Known biological structures
3. Default assumptions for unknown cases
"""

from typing import Dict, List, Tuple
import re


# =============================================================================
# KNOWN STOICHIOMETRIES - Explicit overrides for complexes where inference fails
# =============================================================================
#
# NOTE: Only include entries here that would NOT be correctly inferred by
# _infer_stoichiometry_from_name(). The inference function handles:
#   - MONOMER:: → 1 copy each
#   - HOMO_FAMILY::*_dimer/trimer/tetramer with single gene → uses suffix
#   - HOMO_FAMILY:: with multiple genes → 1 each (isoforms)
#   - HETERO_FAMILY:: → 1 each or evenly distributed
#   - COMPLEX:: → 1 copy each
#
# Add entries here for:
#   1. COMPLEX:: with non-1:1:1 stoichiometry (some subunits have >1 copy)
#   2. HETERO_FAMILY:: with uneven distribution
#   3. Any complex with known non-trivial stoichiometry from PDB structures
#
# Format: complex_id -> {gene: copy_number}
# =============================================================================

KNOWN_STOICHIOMETRIES: Dict[str, Dict[str, int]] = {

    # =========================================================================
    # I. GENETIC INFORMATION PROCESSING
    # =========================================================================

    # --- Translation (Ribosome) ---

    # 60S ribosomal subunit: P-stalk has 2 copies of RPLP1 and RPLP2
    # Without this override, inference gives 1 each. PDB: 4UG0
    "COMPLEX::60S_ribosomal_subunit": {
        "RPL3": 1, "RPL4": 1, "RPL5": 1, "RPL6": 1, "RPL7": 1, "RPL7A": 1,
        "RPL8": 1, "RPL9": 1, "RPL10": 1, "RPL10A": 1, "RPL11": 1, "RPL12": 1,
        "RPL13": 1, "RPL13A": 1, "RPL14": 1, "RPL15": 1, "RPL17": 1, "RPL18": 1,
        "RPL18A": 1, "RPL19": 1, "RPL21": 1, "RPL22": 1, "RPL23": 1, "RPL23A": 1,
        "RPL24": 1, "RPL26": 1, "RPL27": 1, "RPL27A": 1, "RPL28": 1, "RPL29": 1,
        "RPL30": 1, "RPL31": 1, "RPL32": 1, "RPL34": 1, "RPL35": 1, "RPL35A": 1,
        "RPL36": 1, "RPL36A": 1, "RPL37": 1, "RPL37A": 1, "RPL38": 1, "RPL39": 1,
        "RPL41": 1,
        "RPLP0": 1,   # P0 (1 copy)
        "RPLP1": 2,   # P1 (2 copies in P-stalk) **
        "RPLP2": 2,   # P2 (2 copies in P-stalk) **
        "UBA52": 1,
    },

    #"COMPLEX::U5_tri_snRNP_core": {
    #    #"EFTUD2": 1, "SNRNP40": 1, "SNRNP200": 1, "PRPF8": 1, # This is correct, but OOMs.
    #    "EFTUD2": 1, "SNRNP40": 0, "SNRNP200": 1, "PRPF8": 1,
    #},

    # =========================================================================
    # II. PROTEIN HOMEOSTASIS
    # =========================================================================

    # Proteasome 20S core particle: α7β7β7α7 = 2 copies of each subunit
    # Inference would give 1 each. PDB: 6RGQ
    "COMPLEX::Proteasome_20S_core": {
        "PSMA1": 2, "PSMA2": 2, "PSMA3": 2, "PSMA4": 2,
        "PSMA5": 2, "PSMA6": 2, "PSMA7": 2,
        "PSMB1": 2, "PSMB2": 2, "PSMB3": 2, "PSMB4": 2,
        "PSMB5": 2, "PSMB6": 2, "PSMB7": 2,
    },

    # =========================================================================
    # III. BIOENERGETICS
    # =========================================================================

    # --- Glycolysis ---

    # LDH tetramer: 2+2 distribution (not 1:1 even split)
    "HETERO_FAMILY::LDH_tetramer": {
        "LDHA": 2,
        "LDHB": 2,
    },

    # PFK tetramer: tissue-specific, not evenly distributed
    "HETERO_FAMILY::PFK_tetramer": {
        "PFKM": 2, "PFKL": 1, "PFKP": 1,
    },

    # --- TCA Cycle ---

    # IDH3: α2βγ heterotetramer (NOT 1:1:1)
    "COMPLEX::IDH3_complex": {
        "IDH3A": 2,  # Catalytic, 2 copies **
        "IDH3B": 1,
        "IDH3G": 1,
    },

    # Succinyl-CoA synthetase: α2β2 heterotetramer
    "COMPLEX::Succinyl_CoA_synthetase": {
        "SUCLG1": 2,  # α subunit **
        "SUCLG2": 2,  # β subunit (GDP-forming) **
    },

    # --- Electron Transport Chain ---

    # Complex I: NDUFAB1 (acyl carrier protein) has 2 copies
    "COMPLEX::Complex_I_NADH_dehydrogenase": {
        "NDUFS1": 1, "NDUFS2": 1, "NDUFS3": 1, "NDUFS7": 1, "NDUFS8": 1,
        "NDUFV1": 1, "NDUFV2": 1,
        "MT-ND1": 1, "MT-ND2": 1, "MT-ND3": 1, "MT-ND4": 1,
        "MT-ND4L": 1, "MT-ND5": 1, "MT-ND6": 1,
        "NDUFA1": 1, "NDUFA2": 1, "NDUFA3": 1, "NDUFA5": 1, "NDUFA6": 1,
        "NDUFA7": 1, "NDUFA8": 1, "NDUFA9": 1, "NDUFA10": 1, "NDUFA11": 1,
        "NDUFA12": 1, "NDUFA13": 1,
        "NDUFAB1": 2,  # ** Acyl carrier protein - 2 copies **
        "NDUFB1": 1, "NDUFB2": 1, "NDUFB3": 1, "NDUFB4": 1, "NDUFB5": 1,
        "NDUFB6": 1, "NDUFB7": 1, "NDUFB8": 1, "NDUFB9": 1, "NDUFB10": 1,
        "NDUFB11": 1, "NDUFC1": 1, "NDUFC2": 1,
        "NDUFS4": 1, "NDUFS5": 1, "NDUFS6": 1, "NDUFV3": 1,
    },

    # Complex III: functional homodimer - 2 copies of each
    "COMPLEX::Complex_III_cytochrome_bc1": {
        "MT-CYB": 2, "UQCRC1": 2, "UQCRC2": 2, "UQCRFS1": 2, "CYC1": 2,
        "UQCRB": 2, "UQCRH": 2, "UQCRQ": 2, "UQCR10": 2, "UQCR11": 2,
    },

    # ATP synthase F1F0: α3β3 head + c8-ring
    "COMPLEX::ATP_synthase_F1F0": {
        "ATP5F1A": 3, "ATP5F1B": 3, "ATP5F1C": 1, "ATP5F1D": 1, "ATP5F1E": 1,
        "ATP5MC1": 8, # c-ring (8 copies) **
        "ATP5MJ": 1, "ATP5PB": 1, "ATP5PD": 1, "ATP5PF": 1, "ATP5PO": 1,
        "MT-ATP6": 1, "MT-ATP8": 1, "ATP5IF1": 1,
    },

    # =========================================================================
    # IV. STRUCTURE & TRANSPORT
    # =========================================================================

    # Clathrin triskelion: 3 heavy + 3 light chains
    "COMPLEX::Clathrin_triskelion": {
        "CLTC": 3, "CLTA": 3,
    },

    "HETERO_FAMILY::Alpha_Beta_tubulin_heterodimer": {
        "TUBA1A": 1, "TUBB": 1,
    },

    "COMPLEX::COPII_outer_coat": {
        "SEC13": 2, "SEC31A": 2,
    },

    "COMPLEX::NMDAR_GluN1_GluN2A": {
        "GRIN1": 2, "GRIN2A": 2,
    },

    "COMPLEX::NMDAR_GluN1_GluN2B": {
        "GRIN1": 2, "GRIN2B": 2,
    },

    "COMPLEX::NMDAR_GluN1_GluN2C": {
        "GRIN1": 2, "GRIN2C": 2,
    },

    "COMPLEX::NMDAR_GluN1_GluN2D": {
        "GRIN1": 2, "GRIN2D": 2,
    },

    "COMPLEX::AMPAR_GluA1_GluA2": {
        "GRIA1": 2, "GRIA2": 2,
    },

    "COMPLEX::AMPAR_GluA2_GluA3": {
        "GRIA2": 2, "GRIA3": 2,
    },

    "COMPLEX::AMPAR_GluA1_GluA4": {
        "GRIA1": 2, "GRIA4": 2,
    },

    "COMPLEX::GABAA_receptor_synaptic": {
        "GABRA1": 2, "GABRB2": 2, "GABRG2": 1,
    },

    "COMPLEX::GABAA_receptor_extrasynaptic": {
        "GABRA4": 1, "GABRB3": 3, "GABRD": 1,
    },

    # V-ATPase V1: A3B3 + E2G2 (peripheral stalks)
    "COMPLEX::V_ATPase_V1": {
        "ATP6V1A": 3, "ATP6V1B2": 3,
        "ATP6V1E1": 2, "ATP6V1G1": 2,
        "ATP6V1C1": 1, "ATP6V1D": 1, "ATP6V1F": 1, "ATP6V1H": 1,
    },
    # V-ATPase V0: c-ring (~6-10 copies, structure has 10)
    "COMPLEX::V_ATPase_V0": {
        "ATP6V0C": 10,
        "ATP6V0A1": 1, "ATP6V0D1": 1, "ATP6V0E1": 1,
    },

    # Kv Channels are tetramers
    "HOMOMER::Kv1.1": {"KCNA1": 4},
    "HOMOMER::Kv1.2": {"KCNA2": 4, "KCNAB2": 4},
    "HOMOMER::Kv2.1": {"KCNB1": 4},
    "HOMOMER::Kv7.1": {"KCNQ1": 4},
    "HOMOMER::hERG":  {"KCNH2": 4},

    # ABCG2 (BCRP) is a half-transporter, forms homodimer
    "HOMOMER::ABCG2": {"ABCG2": 2},
    # MDR1, CFTR, etc are monomers (handled by default inference)

    # =========================================================================
    # V. CELL CYCLE & FATE
    # =========================================================================

    # TPR lobe contains 2 copies each of CDC27, CDC16, CDC23, ANAPC7
    "COMPLEX::APC_TPR_lobe": {
        "CDC27": 2, "CDC16": 2, "CDC23": 2, "ANAPC7": 2,
    },
    # Platform and Catalytic modules are generally 1:1, so default inference works

    # Apoptosome: APAF1 forms heptameric wheel
    "COMPLEX::Apoptosome_core_partial": {
        "APAF1": 7, "CYCS": 7,
    },
    "COMPLEX::Apoptosome_monomer_unit": {
        "APAF1": 1, "CYCS": 1,
    },

    # =========================================================================
    # VI. SIGNALING PATHWAYS
    # =========================================================================

    "HETERO_FAMILY::ClassIA_PI3K_heterodimer": {
        "PIK3CA": 1, "PIK3R1": 1,
    },

    "COMPLEX::mTORC1_core": {
        #"MTOR": 2, "RPTOR": 2, "MLST8": 2,  # This is correct, but OOMs.
        "MTOR": 1, "RPTOR": 1, "MLST8": 1,
    },

    "COMPLEX::mTORC2_core": {
        #"MTOR": 2, "RICTOR": 2, "MLST8": 2, "MAPKAP1": 2,  # This is correct, but OOMs.
        "MTOR": 1, "RICTOR": 1, "MLST8": 1, "MAPKAP1": 1,
    },

    "COMPLEX::IKK_complex": {
        "CHUK": 1, "IKBKB": 1, "IKBKG": 2,
    },

    "COMPLEX::TCR_CD3": {
        "TRAC": 1, "TRBC1": 1, "CD3D": 1, "CD3E": 2, "CD3G": 1, "CD247": 2,
    },

    "COMPLEX::NLRP3_inflammasome": {
        #"NLRP3": 10, "NEK7": 10,  # This is correct, but OOMs.
        "NLRP3": 3, "NEK7": 3,
    },

    "COMPLEX::ASC_Caspase1_Interface": {
        "PYCARD": 4, "CASP1": 4,
    },

    "COMPLEX::NLRC4_inflammasome": {
        "NLRC4": 11, "NAIP": 1,
    },

    "COMPLEX::AIM2_inflammasome": {
        "AIM2": 8, "PYCARD": 8,
    },

    "COMPLEX::TGFB_receptor_active": {
        "TGFBR2": 2, "TGFBR1": 2,
    },

    "COMPLEX::PKA_RIa_Holoenzyme": {
        "PRKACA": 2, "PRKAR1A": 2,
    },

    "HOMOMER::CREB_dimer": {
        "CREB1": 2,
    },

    # =========================================================================
    # VII. CHROMATIN REGULATION AND GENOME ORGANIZATION
    # =========================================================================

    # Nucleosome core: (H3-H4)2 tetramer + 2x (H2A-H2B) dimers
    "COMPLEX::Nucleosome_core": {
        "H3C1": 2, "H4C1": 2, "H2AC1": 2, "H2BC1": 2,
    },

    # INO80 Core: RuvBL1/2 form a hexameric ring (3 copies each)
    "COMPLEX::INO80_Core": {
        "RUVBL1": 3, "RUVBL2": 3, "INO80": 1,
    },

    "COMPLEX::TIP60_core": {
        "EP400": 1,    # Central scaffold, ATPase motor
        "RUVBL1": 3,   # AAA+ ATPase hexamer component
        "RUVBL2": 3,   # AAA+ ATPase hexamer component
        "EPC1": 1,     # ARP module
        "DMAP1": 1,    # ARP module, connects SWR1L and NuA4L
        "VPS72": 1,    # H2A.Z-H2B chaperone (also known as YL1)
        #"ACTB": 2,     # β-Actin, ARP module
        #"ACTL6A": 2,   # Actin-like 6A (BAF53a), ARP module
        # Remove due to OOM:
        "ACTB": 0,
        "ACTL6A": 0,
    },

    # =========================================================================
    # VIII. DNA REPAIR
    # =========================================================================

    "COMPLEX::BRCA1_A_complex": {
        "BRCA1": 2,
        "BARD1": 2,
        "ABRAXAS1": 2,
        "RAP80": 2,
        "BRCC3": 2,
        "MERIT40": 2,
    },

    "COMPLEX::MRN_complex": {
        "MRE11": 2, "RAD50": 2, "NBN": 2,
    },

    "COMPLEX::Ligase_IV_complex": {
        "LIG4": 1, "XRCC4": 2,
    },

    "COMPLEX::XRCC4_XLF_filament": {
        "XRCC4": 2, "NHEJ1": 2,
    },
}


def _infer_stoichiometry_from_name(complex_id: str, gene_ids: List[str]) -> Dict[str, int]:
    """
    Infer stoichiometry from complex naming conventions.

    Patterns recognized:
    - _trimer, _trimeric -> 3 copies (if single gene)
    - _dimer, _dimeric -> 2 copies (if single gene)
    - _tetramer, _tetrameric -> 4 copies (if single gene)
    - _hexamer, _hexameric -> 6 copies (if single gene)
    - _heptamer, _heptameric -> 7 copies (if single gene)
    - _octamer, _octameric -> 8 copies (if single gene)
    - HOMO_FAMILY with multiple genes -> 1 each (they're isoforms, not co-assembled)
    - HETERO_FAMILY -> 1 each (unless pattern suggests otherwise)
    - COMPLEX -> 1 each (heteromeric complex)
    - MONOMER -> 1 copy
    """
    stoichiometry = {}

    # Check for oligomeric state patterns
    oligomer_patterns = {
        r'_dimer|_dimeric': 2,
        r'_trimer|_trimeric': 3,
        r'_tetramer|_tetrameric': 4,
        r'_pentamer|_pentameric': 5,
        r'_hexamer|_hexameric': 6,
        r'_heptamer|_heptameric': 7,
        r'_octamer|_octameric': 8,
    }

    inferred_copies = 1
    for pattern, copies in oligomer_patterns.items():
        if re.search(pattern, complex_id, re.IGNORECASE):
            inferred_copies = copies
            break

    # Determine complex type
    if complex_id.startswith("MONOMER::"):
        # Monomers: 1 copy each
        for gene in gene_ids:
            stoichiometry[gene] = 1

    elif complex_id.startswith("HOMO_FAMILY::") or complex_id.startswith("HOMOMER::"):
        # Homo-oligomers: if single gene, use inferred copies
        # If multiple genes, they're isoforms (1 each, pick one in practice)
        if len(gene_ids) == 1:
            stoichiometry[gene_ids[0]] = inferred_copies
        else:
            # Multiple isoforms - default to 1 each
            for gene in gene_ids:
                stoichiometry[gene] = 1

    elif complex_id.startswith("HETERO_FAMILY::"):
        # Hetero-oligomers: distribute copies if pattern found
        if inferred_copies > 1 and len(gene_ids) > 1:
            # Try to distribute evenly or default to 1 each
            copies_per_gene = max(1, inferred_copies // len(gene_ids))
            for gene in gene_ids:
                stoichiometry[gene] = copies_per_gene
        else:
            for gene in gene_ids:
                stoichiometry[gene] = 1

    else:  # COMPLEX:: or unknown
        # Default: 1 copy of each subunit
        for gene in gene_ids:
            stoichiometry[gene] = 1

    return stoichiometry


def get_stoichiometry(complex_id: str, gene_ids: List[str]) -> Tuple[Dict[str, int], bool]:
    """
    Get stoichiometry for a complex.

    Args:
        complex_id: The complex identifier (e.g., "COMPLEX::ORC_core")
        gene_ids: List of gene IDs in the complex

    Returns:
        Tuple of:
        - Dict mapping gene_id -> copy number
        - Boolean indicating if stoichiometry was inferred (True) or known (False)
    """
    # Check known stoichiometries first
    if complex_id in KNOWN_STOICHIOMETRIES:
        known = KNOWN_STOICHIOMETRIES[complex_id]
        # Fill in any missing genes with 1 copy
        result = {gene: known.get(gene, 1) for gene in gene_ids}
        return result, False

    # Infer from naming
    inferred = _infer_stoichiometry_from_name(complex_id, gene_ids)
    return inferred, True


def expand_gene_ids_by_stoichiometry(gene_ids: List[str], stoichiometry: Dict[str, int]) -> List[str]:
    """
    Expand gene IDs according to stoichiometry for AF3 scoring.

    For example:
        gene_ids = ["PCNA"], stoichiometry = {"PCNA": 3}
        -> ["PCNA", "PCNA", "PCNA"]

    Args:
        gene_ids: List of unique gene IDs
        stoichiometry: Dict mapping gene_id -> copy number

    Returns:
        Expanded list with genes repeated according to copy number
    """
    expanded = []
    for gene in gene_ids:
        copies = stoichiometry.get(gene, 1)
        expanded.extend([gene] * copies)
    return expanded
