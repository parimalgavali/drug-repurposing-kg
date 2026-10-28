const COLORS: Record<string, string> = {
  Drug: "#2563eb",
  Disease: "#dc2626",
  Gene: "#16a34a",
  Phenotype: "#9333ea",
  Pathway: "#ea580c",
  BiologicalProcess: "#0891b2",
  MolecularFunction: "#0d9488",
  CellularComponent: "#65a30d",
  Anatomy: "#a16207",
  Exposure: "#db2777",
};

export const color = (type: string) => COLORS[type] ?? "#6b7280";
