import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  MarkerType,
  useReactFlow,
  useNodesInitialized,
} from '@xyflow/react'
import { hexFor01, bandOf01, RISK_COLORS } from '../lib/risk'
import { layoutGraph, NODE_WIDTH, NODE_HEIGHT } from '../lib/layout'

const TYPE_ICONS = {
  supplier: '🏭',
  factory: '⚙️',
  port: '⚓',
  route: '🚆',
  warehouse: '📦',
  customer: '🛒',
}

const TYPE_LABELS = {
  supplier: 'Supplier',
  factory: 'Factory',
  port: 'Port',
  route: 'Route',
  warehouse: 'Warehouse',
  customer: 'Customer',
}

function RiskNode({ data, selected }) {
  const baselineHex = hexFor01(data.risk)
  const simulated = data.afterRisk != null && Math.abs(data.afterRisk - data.risk) > 1e-9
  const displayRisk = simulated ? data.afterRisk : data.risk
  const displayHex = hexFor01(displayRisk)
  const bandLabel = bandOf01(displayRisk)

  return (
    <div
      className={`w-[190px] rounded-lg border-2 bg-white px-3 py-2.5 shadow-sm transition-shadow ${
        selected ? 'shadow-lg ring-2 ring-slate-300' : ''
      }`}
      style={{ borderColor: displayHex }}
    >
      <Handle type="target" position={Position.Left} className="!bg-slate-300" />
      <div className="flex items-center justify-between gap-2">
        <span className="text-base leading-none">{TYPE_ICONS[data.type] || '📍'}</span>
        <span
          className="num rounded-full px-1.5 py-0.5 text-[10px] font-bold text-white"
          style={{ backgroundColor: displayHex }}
          title={`baseline ${Math.round((data.risk ?? 0) * 100)}%`}
        >
          {Math.round(displayRisk * 100)}
        </span>
      </div>
      <p className="mt-1.5 truncate text-xs font-semibold text-slate-800" title={data.label}>
        {data.label}
      </p>
      <div className="mt-1 flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wide text-slate-400">
          {TYPE_LABELS[data.type] || data.type}
        </span>
        {simulated ? (
          <span className="rounded bg-indigo-100 px-1 text-[9px] font-bold uppercase text-indigo-700">
            simulated
          </span>
        ) : (
          <span className="text-[9px] font-semibold uppercase" style={{ color: baselineHex }}>
            {bandLabel}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-slate-300" />
    </div>
  )
}

const nodeTypes = { risk: RiskNode }

/**
 * Supply-chain network on React Flow.
 *
 * Layout: Dagre, left-to-right (see src/lib/layout.js) - whatever nodes and
 * edges the backend API returns are positioned hierarchically, then the view
 * auto-fits the container once the nodes are measured.
 *
 * Props:
 *  - graph:        { nodes, edges } from GET /api/graph (required)
 *  - simulation:   Stage 5 result or null (adds before/after overlay)
 *  - onSelectNode: (nodeId) => void - preserved click behavior
 *  - selectedNodeId
 */
function RiskNetworkInner({
  graph,
  simulation,
  onSelectNode,
  selectedNodeId,
  className = '',
}) {
  const [flowNodes, setFlowNodes] = useState([])
  const [flowEdges, setFlowEdges] = useState([])
  const { fitView } = useReactFlow()

  const affectedMap = useMemo(() => {
    if (!simulation?.affected_nodes) return null
    return new Map(simulation.affected_nodes.map((a) => [a.node_id, a.after_risk]))
  }, [simulation])

  // Rebuild + re-layout when the API data or the simulation overlay changes.
  useEffect(() => {
    if (!graph?.nodes?.length) {
      setFlowNodes([])
      setFlowEdges([])
      return
    }
    const affected = affectedMap || new Map()

    const nodes = layoutGraph(
      graph.nodes.map((n) => ({
        id: n.id,
        type: 'risk',
        position: n.position ?? { x: 0, y: 0 }, // dagre overwrites this
        data: { ...n.data, afterRisk: affected.get(n.id) },
      })),
      graph.edges,
    )

    const affectedIds = new Set(affected.keys())
    setFlowNodes(nodes)
    setFlowEdges(
      graph.edges.map((e) => {
        const onPath = affectedIds.has(e.target)
        return {
          id: e.id ?? `${e.source}->${e.target}`,
          source: e.source,
          target: e.target,
          label: e.label,
          type: 'smoothstep',
          animated: onPath,
          style: {
            stroke: simulation ? (onPath ? '#6366f1' : '#cbd5e1') : '#94a3b8',
            strokeWidth: onPath ? 2 : 1.5,
          },
          labelStyle: { fontSize: 9, fill: '#64748b', fontWeight: 600 },
          labelBgStyle: { fill: '#f8fafc', fillOpacity: 0.9 },
          labelBgPadding: [3, 1],
          labelBgBorderRadius: 3,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: simulation ? (onPath ? '#6366f1' : '#cbd5e1') : '#64748b',
            width: 15,
            height: 15,
          },
        }
      }),
    )
  }, [graph, affectedMap, simulation])

  // Auto-fit once the nodes are actually measured (avoids fitting an empty
  // canvas or mid-layout state). Keeps zoom sane: never too far out, never
  // blown up, complete network centered with breathing room.
  const nodesInitialized = useNodesInitialized()
  const [hasFitted, setHasFitted] = useState(false)
  const FIT_OPTS = { padding: 0.15, minZoom: 0.2, maxZoom: 1.1, duration: 400 }
  useEffect(() => {
    if (nodesInitialized && flowNodes.length && !hasFitted) {
      setHasFitted(true)
      const id = requestAnimationFrame(() => fitView(FIT_OPTS))
      return () => cancelAnimationFrame(id)
    }
    return undefined
  }, [nodesInitialized, flowNodes.length, hasFitted, fitView])

  // Re-fit when a new simulation lands so the whole network stays visible.
  useEffect(() => {
    if (simulation && flowNodes.length) {
      const id = requestAnimationFrame(() => fitView(FIT_OPTS))
      return () => cancelAnimationFrame(id)
    }
    return undefined
  }, [simulation]) // eslint-disable-line react-hooks/exhaustive-deps

  const onNodeClick = useCallback(
    (_event, node) => onSelectNode?.(node.id),
    [onSelectNode],
  )

  if (!graph) return null

  return (
    <div className={`h-full w-full ${className}`}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.15, minZoom: 0.2, maxZoom: 1.1 }}
        minZoom={0.2}
        maxZoom={2}
        nodesDraggable
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
        className="bg-slate-50"
      >
        <Background color="#e2e8f0" gap={18} />
        <Controls showInteractive={false} position="bottom-right" />
        <MiniMap
          pannable
          zoomable
          style={{ width: 140, height: 84 }}
          nodeColor={(n) => hexFor01(n.data?.afterRisk ?? n.data?.risk ?? 0)}
          maskColor="rgba(241,245,249,0.7)"
          className="!bottom-3 !left-3"
        />
      </ReactFlow>
    </div>
  )
}

// useReactFlow/useNodesInitialized require a provider ancestor
export default function RiskNetwork(props) {
  return (
    <ReactFlowProvider>
      <RiskNetworkInner {...props} />
    </ReactFlowProvider>
  )
}
