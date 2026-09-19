import dagre from '@dagrejs/dagre'

// Node size must match the CSS of <RiskNode> in RiskNetwork.jsx (w-[190px] ~
// 92px tall including padding/border).
export const NODE_WIDTH = 190
export const NODE_HEIGHT = 96

/**
 * Hierarchical left-to-right layout for the supply-chain network.
 *
 * Dagre assigns each node a "rank" (column) using the longest path from a
 * source, so tiers naturally come out as:
 *   suppliers -> factories -> (ports/routes) -> warehouses -> customers
 * Ranks run left to right (rankdir: 'LR'); nodes inside a rank are spread
 * vertically (nodesep) and columns are spaced by ranksep - no overlaps,
 * readable edges, and the flow direction always matches supply direction.
 *
 * Pure function: takes whatever nodes/edges the backend API returns and
 * returns the same nodes with `position` filled in. Nothing is hardcoded.
 */
export function layoutGraph(nodes, edges, options = {}) {
  const {
    direction = 'LR',
    nodeSep = 60, // vertical gap between nodes in the same rank (px)
    rankSep = 120, // horizontal gap between stages (px)
    nodeWidth = NODE_WIDTH,
    nodeHeight = NODE_HEIGHT,
  } = options

  if (!nodes?.length) return []

  const g = new dagre.graphlib.Graph()
  g.setGraph({
    rankdir: direction,
    nodesep: nodeSep,
    ranksep: rankSep,
    marginx: 20,
    marginy: 20,
  })
  g.setDefaultEdgeLabel(() => ({}))

  for (const node of nodes) {
    g.setNode(node.id, { width: nodeWidth, height: nodeHeight })
  }
  for (const edge of edges) {
    // dagre needs every endpoint to exist; guard against stale edges
    if (g.hasNode(edge.source) && g.hasNode(edge.target)) {
      g.setEdge(edge.source, edge.target)
    }
  }

  dagre.layout(g)

  return nodes.map((node) => {
    const pos = g.node(node.id)
    // dagre returns the node CENTER; React Flow wants the top-left corner
    return {
      ...node,
      position: {
        x: pos.x - nodeWidth / 2,
        y: pos.y - nodeHeight / 2,
      },
    }
  })
}
