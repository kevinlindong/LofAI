import { notFound } from "next/navigation"
import { DesignWorkspace } from "@/components/design-workspace"
import { DESIGNS } from "@/lib/designs"

export function generateStaticParams() {
  return DESIGNS.filter((design) => design.id !== "1").map((design) => ({ design: design.id }))
}

export const dynamicParams = false

export function generateMetadata({ params }: { params: { design: string } }) {
  const design = DESIGNS.find((entry) => entry.id === params.design)
  return { title: design ? `${design.name} — lofAI` : "lofAI" }
}

export default function DesignPage({ params }: { params: { design: string } }) {
  const design = DESIGNS.find((entry) => entry.id === params.design)
  if (!design || design.id === "1") notFound()
  return <DesignWorkspace key={design.id} design={design} />
}
