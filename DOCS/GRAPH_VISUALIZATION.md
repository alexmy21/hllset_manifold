# Graph Visualization Tool

## Overview

The graph visualization tool converts HLLSet lattice structures into property graphs, enabling visual analysis and consistency validation. All lattice structures (AM, W, HRT, database hierarchies) are fundamentally graphs, and this tool provides a unified way to visualize and analyze them.

## Why Property Graphs?

**Property graphs** extend basic graphs by attaching metadata to both nodes and edges:

```python
Node: {
    id: "table_sales",
    type: "table",
    label: "sales",
    cardinality: 1000000,
    artifact_id: "abc123...",
    sha1: "def456..."
}

Edge: {
    source: "database",
    target: "table_sales",
    type: "contains",
    weight: 1000000,
    properties: {...}
}
```

This allows rich analysis beyond simple connectivity.

## Key Features

### 1. Multiple Lattice Types

The tool handles various lattice structures:

- **Adjacency Matrix (AM)**: Token adjacency relationships
- **W Lattice**: Row/column HLLSet structures  
- **Database Hierarchy**: Tables, columns, entanglements
- **Custom graphs**: Extend with your own structures

### 2. Consistency Testing

Automatically validates graph structure:

✅ All edges reference existing nodes  
✅ No duplicate node IDs  
✅ Containment hierarchies are acyclic  
✅ Property schemas consistent within node types  

```python
from core.graph_visualizer import test_consistency

consistency = test_consistency(builder)
if consistency['passed']:
    print("✓ Graph is consistent!")
else:
    print("Issues found:", consistency['errors'])
```

### 3. Multiple Export Formats

Export to standard graph formats:

- **GraphML**: Property graph format (Gephi, Neo4j)
- **DOT**: Graphviz format (visualization)
- **JSON**: Custom analysis and processing

### 4. Relationship Discovery

Automatically detect patterns:

- **Foreign Keys**: Column similarity analysis
- **Data Overlap**: Cross-table comparisons
- **Hierarchy Validation**: Check containment structure

## Architecture

```text
Lattice Structure
      ↓
LatticeGraphBuilder  ← Converts to property graph
      ↓
NetworkX DiGraph     ← Standard graph library
      ↓
      ├→ LatticeVisualizer  ← Matplotlib visualization
      ├→ Export (GraphML/DOT/JSON)
      └→ Consistency Testing
```

## Usage

### Basic Workflow

```python
from core.graph_visualizer import LatticeGraphBuilder, LatticeVisualizer

# 1. Create builder
builder = LatticeGraphBuilder()

# 2. Load from lattice structure
builder.from_database_hierarchy(ingestion_result, manifold)

# 3. Test consistency
from core.graph_visualizer import test_consistency
consistency = test_consistency(builder)

# 4. Get statistics
viz = LatticeVisualizer(builder)
viz.print_statistics()

# 5. Visualize
viz.plot(figsize=(12, 8), layout='spring')

# 6. Export
builder.export_graphml(Path('./graph.graphml'))
```

### From Adjacency Matrix

```python
from core.hrt import AdjacencyMatrix

# Create AM
am = AdjacencyMatrix(["Alice", "Bob", "Carol"])
am.add_adjacency(0, 1)  # Alice -> Bob
am.add_adjacency(1, 2)  # Bob -> Carol

# Build graph
builder = LatticeGraphBuilder()
builder.from_adjacency_matrix(am)

# Visualize
viz = LatticeVisualizer(builder)
viz.plot()
```

**Result**: Graph with token nodes and adjacency edges.

### From W Lattice

```python
# Create W lattice
class MyWLattice:
    def __init__(self):
        self.rows = [...]     # Row HLLSets
        self.columns = [...]  # Column HLLSets

w_lattice = MyWLattice()

# Build graph
builder = LatticeGraphBuilder()
builder.from_w_lattice(w_lattice)

# Visualize
viz = LatticeVisualizer(builder)
viz.plot(layout='hierarchical')
```

**Result**: Graph with row nodes, column nodes, and cell nodes showing intersections.

### From Database Hierarchy

```python
# After database ingestion
builder = LatticeGraphBuilder()
builder.from_database_hierarchy(ingestion_result, manifold)

# Detect foreign keys
fk_count = builder.detect_potential_foreign_keys(threshold=0.7)
print(f"Found {fk_count} potential FKs")

# Visualize subgraph around specific table
viz = LatticeVisualizer(builder)
viz.plot_subgraph(
    node_ids=["table_sales"],
    depth=2,  # Include related nodes
    figsize=(12, 8)
)
```

**Result**: Graph showing database → tables → columns with FK relationships.

## Node Types

Different lattice structures create different node types:

| Lattice Type | Node Types |
| ------------- | ----------- |
| Adjacency Matrix | `token` |
| W Lattice | `row_hllset`, `col_hllset`, `cell_hllset` |
| Database | `database`, `table`, `column` |

Each node type has consistent properties:

**Token node**:

```python
{
    'id': 'token_0_Alice',
    'type': 'token',
    'label': 'Alice',
    'index': 0,
    'token': 'Alice'
}
```

**Table node**:

```python
{
    'id': 'table_sales',
    'type': 'table',
    'label': 'sales',
    'cardinality': 1000000,
    'artifact_id': 'abc123...',
    'sha1': 'def456...',
    'column_count': 15
}
```

**Column node**:

```python
{
    'id': 'column_sales_revenue',
    'type': 'column',
    'label': 'sales.revenue',
    'table': 'sales',
    'column': 'revenue',
    'data_type': 'DECIMAL',
    'cardinality': 50000,
    'distinct_count': 50000
}
```

## Edge Types

| Edge Type | Meaning | Example |
| ----------- | --------- | --------- |
| `contains` | Hierarchical containment | Database → Table → Column |
| `adjacent` | Token adjacency | Token A → Token B |
| `entangled` | Explicit entanglement | Column data ↔ Metadata |
| `potential_fk` | Detected foreign key | Column A → Column B |
| `morphism` | Lattice morphism | HLLSet A → HLLSet B |

Edges have weights based on cardinality or similarity.

## Visualization Layouts

Different layouts emphasize different aspects:

### Spring Layout (default)

```python
viz.plot(layout='spring')
```

- Force-directed layout
- Good for general structure
- Natural clustering

### Hierarchical Layout

```python
viz.plot(layout='hierarchical')
```

- Top-down tree structure
- Good for containment hierarchies
- Requires graphviz/pydot

### Circular Layout

```python
viz.plot(layout='circular')
```

- Nodes arranged in circle
- Good for seeing all connections
- Works with any graph

### Kamada-Kawai Layout

```python
viz.plot(layout='kamada_kawai')
```

- Energy-based layout
- Good for symmetric structures
- Slower for large graphs

## Subgraph Visualization

For large graphs, visualize portions:

```python
# Get table node IDs
table_nodes = [
    nid for nid, meta in builder.nodes.items() 
    if meta.node_type == 'table'
]

# Visualize first table and its neighborhood
viz.plot_subgraph(
    node_ids=[table_nodes[0]],
    depth=2,  # Include nodes up to 2 hops away
    figsize=(10, 8),
    show_labels=True
)
```

**depth=1**: Node + immediate neighbors  
**depth=2**: Node + neighbors + neighbors' neighbors  
**depth=3+**: Quickly grows to full graph

## Export Formats

### GraphML (Recommended)

```python
builder.export_graphml(Path('./graph.graphml'))
```

**Use with**:

- **Gephi**: Import → Graph File → Open
- **Neo4j**: LOAD CSV or apoc.import.graphml
- **igraph**: read_graph()
- **networkx**: read_graphml()

**Benefits**: Preserves all properties, widely supported

### DOT (Graphviz)

```python
builder.export_dot(Path('./graph.dot'))
```

**Render with**:

```bash
dot -Tpng graph.dot -o graph.png
dot -Tsvg graph.dot -o graph.svg
```

**Benefits**: Publication-quality layouts, LaTeX integration

**Requires**: pydot package (`pip install pydot`)

### JSON

```python
builder.export_json(Path('./graph.json'))
```

**Structure**:

```json
{
  "nodes": [
    {"id": "...", "type": "...", "label": "...", ...},
    ...
  ],
  "edges": [
    {"source": "...", "target": "...", "type": "...", ...},
    ...
  ]
}
```

**Benefits**: Custom analysis, web visualization (D3.js, etc.)

## Consistency Testing

The `test_consistency()` function validates graph structure:

```python
results = test_consistency(builder)

# Check if passed
if results['passed']:
    print("✓ All consistency checks passed")
else:
    print("✗ Consistency issues found:")
    for error in results['errors']:
        print(f"  - {error}")

# Warnings are non-fatal
if results['warnings']:
    print("Warnings:")
    for warning in results['warnings']:
        print(f"  - {warning}")
```

### Checks Performed

1. **Node ID Uniqueness**: No duplicate IDs
2. **Edge Validity**: All edges reference existing nodes
3. **No Self-Loops**: (Except where expected)
4. **Acyclic Containment**: Containment hierarchy has no cycles
5. **Property Consistency**: Same node type has same properties

### Why This Matters

Consistency testing ensures:

- Your lattice structures are well-formed
- Data can be safely traversed
- Export formats will be valid
- Graph algorithms will work correctly

## Foreign Key Detection

Automatically discover relationships between columns:

```python
# After building database graph
fk_count = builder.detect_potential_foreign_keys(threshold=0.7)

# Get FK edges
fk_edges = [e for e in builder.edges if e.edge_type == 'potential_fk']

for edge in fk_edges:
    source_label = builder.nodes[edge.source].label
    target_label = builder.nodes[edge.target].label
    confidence = edge.properties.get('confidence', 0)
    print(f"{source_label} → {target_label} ({confidence:.2f})")
```

### Detection Heuristics

Current implementation uses name-based heuristics:

1. Column ends with `_id` and matches table name
2. Both columns named `id` and one has suffix
3. Columns with identical names containing `id`

**Future**: Use HLLSet similarity for value-based FK detection:

```python
# Load column HLLSets
hll1 = HLLSet.from_bytes(manifold.retrieve_artifact(col1_id))
hll2 = HLLSet.from_bytes(manifold.retrieve_artifact(col2_id))

# High overlap suggests FK relationship
intersection = hll1.intersection(hll2)
jaccard = intersection.cardinality() / hll1.union(hll2).cardinality()

if jaccard > 0.8:  # High similarity
    print(f"Potential FK: {jaccard:.2f} overlap")
```

## Statistics

Get graph statistics:

```python
stats = builder.get_statistics()

print(f"Nodes: {stats['node_count']}")
print(f"Edges: {stats['edge_count']}")
print(f"Density: {stats['density']:.4f}")
print(f"Connected: {stats['is_connected']}")
print(f"Average degree: {stats['avg_degree']:.2f}")

# By type
for node_type, count in stats['node_types'].items():
    print(f"  {node_type}: {count}")
```

**Density**: Ratio of actual edges to possible edges  
**Connected**: Can reach any node from any other node  
**Degree**: Number of edges per node (average, min, max)

## Integration Points

### With ManifoldOS

```python
# After ingestion
builder.from_database_hierarchy(ingestion_result, manifold)

# Load artifacts through manifold
for node_id, metadata in builder.nodes.items():
    if 'artifact_id' in metadata.properties:
        artifact = manifold.retrieve_artifact(metadata.properties['artifact_id'])
        # Process artifact...
```

### With NetworkX

The underlying graph is a NetworkX DiGraph:

```python
# Access directly
G = builder.graph

# Use NetworkX algorithms
import networkx as nx

# Centrality
centrality = nx.betweenness_centrality(G)
most_central = max(centrality, key=centrality.get)

# Shortest path
path = nx.shortest_path(G, source, target)

# Communities
communities = nx.community.greedy_modularity_communities(G.to_undirected())
```

### With External Tools

**Gephi**:

1. Export GraphML: `builder.export_graphml('graph.graphml')`
2. Open Gephi → File → Open → Select file
3. Visualize with force-directed layout
4. Color by node type, size by cardinality

**Neo4j**:

1. Export GraphML: `builder.export_graphml('graph.graphml')`
2. Use APOC: `CALL apoc.import.graphml('graph.graphml', {})`
3. Query: `MATCH (n:table)-[:contains]->(m:column) RETURN n, m`

**Custom D3.js**:

1. Export JSON: `builder.export_json('graph.json')`
2. Load in JavaScript: `d3.json('graph.json').then(data => {...})`
3. Render with force layout

## Performance

### Memory

- Node: ~200 bytes (metadata + properties)
- Edge: ~100 bytes (source, target, properties)
- Graph overhead: ~50 bytes per node/edge

**Example**: 10,000 nodes, 50,000 edges:

- Nodes: 2 MB
- Edges: 5 MB
- Total: ~7-10 MB

### Speed

| Operation | Time (10K nodes) | Time (100K nodes) |
| ----------- | ----------------- | ------------------ |
| Build from AM | ~0.1s | ~1s |
| Build from W | ~0.5s | ~5s |
| Build from DB | ~1s | ~10s |
| Consistency test | ~0.05s | ~0.5s |
| Export GraphML | ~0.2s | ~2s |
| Visualize (spring) | ~1s | ~30s |

**Recommendation**: For graphs >1000 nodes, use subgraph visualization.

## Examples

See these notebooks for complete examples:

- **[demo_graph_visualization.ipynb](demo_graph_visualization.ipynb)** - Full testing suite
- **[workbook_db_ingestion.ipynb](workbook_db_ingestion.ipynb)** - Real database visualization

## Future Enhancements

### Planned Features

1. **Value-based FK detection**: Use HLLSet similarity instead of heuristics
2. **Graph algorithms**: Centrality, communities, paths
3. **Interactive visualization**: Plotly/Dash integration
4. **Temporal graphs**: Track evolution over time
5. **Diff visualization**: Compare two graph versions

### Extensibility

Add custom node/edge types:

```python
# Custom node type
builder.add_node(NodeMetadata(
    node_id='custom_1',
    node_type='my_type',
    label='Custom Node',
    properties={'custom_prop': 'value'}
))

# Custom edge type
builder.add_edge(EdgeMetadata(
    source='node_a',
    target='node_b',
    edge_type='my_relationship',
    weight=1.0,
    properties={'custom': 'data'}
))
```

## Conclusion

The graph visualization tool provides:

✅ **Unified representation** - All lattices as property graphs  
✅ **Consistency validation** - Verify structure integrity  
✅ **Visual analysis** - See relationships and patterns  
✅ **Multiple exports** - Use with external tools  
✅ **Relationship discovery** - Automatic FK detection  

Use it to:

- Debug lattice structures
- Understand data relationships
- Validate system consistency
- Export for further analysis
- Communicate structure visually

The tool ensures all our lattice structures follow consistent patterns, making the entire system more reliable and easier to reason about.
