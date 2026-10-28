// Constraints and indexes for the RepurposeKG graph. Idempotent; apply with `make kg-schema`.
CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (n:Gene) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT phenotype_id IF NOT EXISTS FOR (n:Phenotype) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT exposure_id IF NOT EXISTS FOR (n:Exposure) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (n:Pathway) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT biological_process_id IF NOT EXISTS FOR (n:BiologicalProcess) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT molecular_function_id IF NOT EXISTS FOR (n:MolecularFunction) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT cellular_component_id IF NOT EXISTS FOR (n:CellularComponent) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT anatomy_id IF NOT EXISTS FOR (n:Anatomy) REQUIRE n.id IS UNIQUE;
CREATE INDEX drug_name IF NOT EXISTS FOR (n:Drug) ON (n.name);
CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name);
CREATE FULLTEXT INDEX entityNames IF NOT EXISTS
FOR (n:Drug|Disease|Gene|Phenotype|Exposure|Pathway|BiologicalProcess|MolecularFunction|CellularComponent|Anatomy)
ON EACH [n.name];
// German Wikidata names (`make labels-de`), so search also finds e.g. "Kurzsichtigkeit".
CREATE FULLTEXT INDEX entityNamesDe IF NOT EXISTS
FOR (n:Drug|Disease)
ON EACH [n.name_de];
