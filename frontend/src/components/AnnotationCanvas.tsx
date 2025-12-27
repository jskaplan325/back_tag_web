import { useState, useRef, useCallback } from 'react'
import clsx from 'clsx'

export interface Annotation {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  tag_name: string
  color: 'green' | 'yellow' | 'red'
  annotation_type: 'positive' | 'negative' | 'uncertain'
}

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

interface AnnotationCanvasProps {
  annotations: Annotation[]
  isDrawingEnabled: boolean
  onAnnotationCreate: (box: BoundingBox) => void
  onAnnotationSelect: (annotation: Annotation) => void
  onAnnotationDelete?: (id: string) => void
  selectedAnnotationId?: string | null
}

export function AnnotationCanvas({
  annotations,
  isDrawingEnabled,
  onAnnotationCreate,
  onAnnotationSelect,
  onAnnotationDelete,
  selectedAnnotationId
}: AnnotationCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null)
  const [currentBox, setCurrentBox] = useState<BoundingBox | null>(null)

  // Convert mouse event to normalized coordinates (0-1)
  const getRelativeCoords = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return { x: 0, y: 0 }

    const rect = svgRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height

    // Clamp to 0-1
    return {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y))
    }
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDrawingEnabled) return

    // Ignore if clicking on an existing annotation
    if ((e.target as Element).classList.contains('annotation-rect')) return

    const coords = getRelativeCoords(e)
    setIsDrawing(true)
    setStartPoint(coords)
    setCurrentBox({
      x1: coords.x,
      y1: coords.y,
      x2: coords.x,
      y2: coords.y
    })
  }, [isDrawingEnabled, getRelativeCoords])

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDrawing || !startPoint) return

    const coords = getRelativeCoords(e)

    // Create box with proper min/max so x1,y1 is always top-left
    setCurrentBox({
      x1: Math.min(startPoint.x, coords.x),
      y1: Math.min(startPoint.y, coords.y),
      x2: Math.max(startPoint.x, coords.x),
      y2: Math.max(startPoint.y, coords.y)
    })
  }, [isDrawing, startPoint, getRelativeCoords])

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !currentBox) {
      setIsDrawing(false)
      setStartPoint(null)
      setCurrentBox(null)
      return
    }

    // Only create if box has some minimum size (prevent accidental clicks)
    const width = currentBox.x2 - currentBox.x1
    const height = currentBox.y2 - currentBox.y1

    if (width > 0.01 && height > 0.01) {
      onAnnotationCreate(currentBox)
    }

    setIsDrawing(false)
    setStartPoint(null)
    setCurrentBox(null)
  }, [isDrawing, currentBox, onAnnotationCreate])

  const handleAnnotationClick = useCallback((e: React.MouseEvent, annotation: Annotation) => {
    e.stopPropagation()
    onAnnotationSelect(annotation)
  }, [onAnnotationSelect])

  const handleDeleteClick = useCallback((e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    onAnnotationDelete?.(id)
  }, [onAnnotationDelete])

  // Color classes for annotations
  const getColorClasses = (color: string, isSelected: boolean) => {
    const baseClasses = {
      green: 'fill-green-500/20 stroke-green-500',
      yellow: 'fill-yellow-500/20 stroke-yellow-500',
      red: 'fill-red-500/20 stroke-red-500'
    }

    const selectedClasses = {
      green: 'fill-green-500/40 stroke-green-600',
      yellow: 'fill-yellow-500/40 stroke-yellow-600',
      red: 'fill-red-500/40 stroke-red-600'
    }

    return isSelected
      ? selectedClasses[color as keyof typeof selectedClasses] || selectedClasses.green
      : baseClasses[color as keyof typeof baseClasses] || baseClasses.green
  }

  return (
    <svg
      ref={svgRef}
      className={clsx(
        'absolute inset-0 w-full h-full',
        isDrawingEnabled ? 'cursor-crosshair' : 'pointer-events-none'
      )}
      style={{ touchAction: 'none' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Existing annotations */}
      {annotations.map(ann => {
        const isSelected = ann.id === selectedAnnotationId
        return (
          <g key={ann.id}>
            <rect
              className={clsx(
                'annotation-rect cursor-pointer transition-all',
                getColorClasses(ann.color, isSelected),
                isSelected && 'stroke-2'
              )}
              x={`${ann.x1 * 100}%`}
              y={`${ann.y1 * 100}%`}
              width={`${(ann.x2 - ann.x1) * 100}%`}
              height={`${(ann.y2 - ann.y1) * 100}%`}
              strokeWidth={isSelected ? 3 : 2}
              style={{ pointerEvents: 'auto' }}
              onClick={(e) => handleAnnotationClick(e, ann)}
            />
            {/* Tag label */}
            <foreignObject
              x={`${ann.x1 * 100}%`}
              y={`${ann.y1 * 100}%`}
              width={`${(ann.x2 - ann.x1) * 100}%`}
              height="24"
              style={{ pointerEvents: 'none', overflow: 'visible' }}
            >
              <div
                className={clsx(
                  'inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-medium rounded-b',
                  ann.color === 'green' && 'bg-green-500 text-white',
                  ann.color === 'yellow' && 'bg-yellow-500 text-black',
                  ann.color === 'red' && 'bg-red-500 text-white'
                )}
                style={{ transform: 'translateY(-100%)' }}
              >
                <span className="truncate max-w-[100px]">{ann.tag_name}</span>
                {isSelected && onAnnotationDelete && (
                  <button
                    className="ml-1 hover:bg-black/20 rounded px-0.5"
                    style={{ pointerEvents: 'auto' }}
                    onClick={(e) => handleDeleteClick(e, ann.id)}
                  >
                    ×
                  </button>
                )}
              </div>
            </foreignObject>
          </g>
        )
      })}

      {/* Drawing preview */}
      {currentBox && (
        <rect
          className="fill-blue-500/20 stroke-blue-500 stroke-dashed pointer-events-none"
          x={`${currentBox.x1 * 100}%`}
          y={`${currentBox.y1 * 100}%`}
          width={`${(currentBox.x2 - currentBox.x1) * 100}%`}
          height={`${(currentBox.y2 - currentBox.y1) * 100}%`}
          strokeWidth={2}
          strokeDasharray="5,5"
        />
      )}
    </svg>
  )
}
