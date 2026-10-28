"""Graph schema as the LLM sees it, taken from the loaded PrimeKG graph (counts rounded)."""

LABELS = (
    "Drug",
    "Disease",
    "Gene",
    "Phenotype",
    "Exposure",
    "Pathway",
    "BiologicalProcess",
    "MolecularFunction",
    "CellularComponent",
    "Anatomy",
)

# (source label, relationship type, target label). Relationships are stored once; query them
# undirected (-[:T]-) unless the direction below matters.
TRIPLES = (
    ("Drug", "INDICATION", "Disease"),
    ("Drug", "CONTRAINDICATION", "Disease"),
    ("Drug", "OFF_LABEL_USE", "Disease"),
    ("Drug", "PREDICTED_INDICATION", "Disease"),
    ("Drug", "TARGET", "Gene"),
    ("Drug", "ENZYME", "Gene"),
    ("Drug", "TRANSPORTER", "Gene"),
    ("Drug", "CARRIER", "Gene"),
    ("Drug", "DRUG_EFFECT", "Phenotype"),
    ("Drug", "DRUG_DRUG", "Drug"),
    ("Disease", "DISEASE_PROTEIN", "Gene"),
    ("Disease", "DISEASE_PHENOTYPE_POSITIVE", "Phenotype"),
    ("Disease", "DISEASE_PHENOTYPE_NEGATIVE", "Phenotype"),
    ("Disease", "DISEASE_DISEASE", "Disease"),
    ("Disease", "EXPOSURE_DISEASE", "Exposure"),
    ("Gene", "PROTEIN_PROTEIN", "Gene"),
    ("Gene", "PATHWAY_PROTEIN", "Pathway"),
    ("Gene", "PHENOTYPE_PROTEIN", "Phenotype"),
    ("Gene", "BIOPROCESS_PROTEIN", "BiologicalProcess"),
    ("Gene", "MOLFUNC_PROTEIN", "MolecularFunction"),
    ("Gene", "CELLCOMP_PROTEIN", "CellularComponent"),
    ("Gene", "ANATOMY_PROTEIN_PRESENT", "Anatomy"),
    ("Gene", "ANATOMY_PROTEIN_ABSENT", "Anatomy"),
    ("Gene", "EXPOSURE_PROTEIN", "Exposure"),
    ("Phenotype", "PHENOTYPE_PHENOTYPE", "Phenotype"),
    ("Pathway", "PATHWAY_PATHWAY", "Pathway"),
    ("BiologicalProcess", "BIOPROCESS_BIOPROCESS", "BiologicalProcess"),
    ("MolecularFunction", "MOLFUNC_MOLFUNC", "MolecularFunction"),
    ("CellularComponent", "CELLCOMP_CELLCOMP", "CellularComponent"),
    ("Anatomy", "ANATOMY_ANATOMY", "Anatomy"),
    ("Exposure", "EXPOSURE_EXPOSURE", "Exposure"),
    ("Exposure", "EXPOSURE_BIOPROCESS", "BiologicalProcess"),
    ("Exposure", "EXPOSURE_MOLFUNC", "MolecularFunction"),
    ("Exposure", "EXPOSURE_CELLCOMP", "CellularComponent"),
)
REL_TYPES = tuple(sorted({t for _, t, _ in TRIPLES}))


def schema_prompt() -> str:
    lines = [
        "Node labels (every node has properties id: int, name: string, source: string): "
        + ", ".join(LABELS),
        "Relationships (source)-[TYPE]->(target):",
        *(f"  ({a})-[:{t}]->({b})" for a, t, b in TRIPLES),
        "PREDICTED_INDICATION has properties score: float, rank: int (1 = best), "
        "model_version: string, and optionally ct_studies: int, ct_max_phase: float.",
        "Indexes: unique id per label; full-text index 'entityNames' on name.",
    ]
    return "\n".join(lines)
