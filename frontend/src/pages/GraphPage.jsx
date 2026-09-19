import { useEffect, useState } from 'react'
import { fetchGraph } from '../api'
import RiskNetwork from '../components/RiskNetwork'
import NodeDetails from '../components/NodeDetails'
import { Card, LoadingState, ErrorState, EmptyState, Badge } from '../components/ui'
import { bandOf01 } from '../lib/risk'

export default function GraphPage({ simulation }) {
  const [graph, setGraph] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState(null)

  const load = () => {
    setLoading(true)
    setError(null)
    fetchGraph()
      .then(setGraph)
      .catch((e) => setError(e?.message || 'Request failed'))
      .finally(() => setLoading(false))
  }
  useEffect(load, [])

  if (loading) return <LoadingState label="Loading supply-chain network…" />
  if (error) return <ErrorState message={error} onRetry={load} />
  if (!graph) return null

  // explicit empty state when the API returns no network
  if (!graph.nodes?.length) {
    return (
      <Card title="Supply-chain network">
        <EmptyState>No supply-chain network data available.</EmptyState>
      </Card>
    )
  }

  const selectedNode = graph.nodes.find((n) => n.id === selectedId)?.data
  const counts = { LOW: 0, MEDIUM: 0, HIGH: 0 }
  graph.nodes.forEach((n) => { counts[bandOf01(n.data.risk)] += 1 })

  return (
    <div className="relative flex h-[calc(100vh-170px)] min-h-[480px] gap-4">
      <Card
        className="flex min-w-0 flex-1 flex-col overflow-hidden"
        bodyClassName="min-h-0 flex-1 !p-0"
        title="Supply-chain network"
        subtitle="left-to-right supply flow · drag to pan · scroll to zoom · click a node for details"
        actions={
          <div className="flex items-center gap-3 text-[11px] font-semibold">
            <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />LOW {counts.LOW}</span>
            <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-amber-500" />MEDIUM {counts.MEDIUM}</span>
            <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-rose-600" />HIGH {counts.HIGH}</span>
            {simulation && <Badge tone="MEDIUM">simulation overlay active</Badge>}
          </div>
        }
      >
        {/* the card body is the only size constraint; the canvas fills it */}
        <RiskNetwork
          graph={graph}
          simulation={simulation}
          selectedNodeId={selectedId}
          onSelectNode={setSelectedId}
        />
      </Card>
      {selectedNode && (
        <div className="absolute inset-y-0 right-0 z-20 w-[340px] max-w-[88%] shadow-xl xl:static xl:max-w-none xl:shadow-none">
          <NodeDetails node={selectedNode} graph={graph} onClose={() => setSelectedId(null)} />
        </div>
      )}
    </div>
  )
}
