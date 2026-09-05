"use client"

import { useId, type CSSProperties } from "react"

interface DotSliderProps {
  label: string
  readout: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  segments?: number
  disabled?: boolean
  // called when the drag ends, for controls that settle onto fixed positions
  onRelease?: () => void
}

// A continuous rail lets the value flow from one stop to the next. Three tiny
// nodes preserve a trace of the old dot vocabulary without making the whole
// control another matrix. The native input stays on top for keyboard, touch
// and screen-reader behaviour.
export function DotSlider({
  label,
  readout,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  segments = 16,
  disabled = false,
  onRelease,
}: DotSliderProps) {
  const id = useId()
  const fraction = max === min ? 0 : (value - min) / (max - min)
  const position = `${Math.min(1, Math.max(0, fraction)) * 100}%`
  const markerCount = Math.max(2, Math.min(3, segments))
  const rangeStyle = { "--range-position": position } as CSSProperties

  return (
    <div className={disabled ? "opacity-40" : undefined}>
      <div className="flex items-baseline justify-between mb-2">
        <label htmlFor={id} className="label">
          {label}
        </label>
        <span className="readout text-xs">{readout}</span>
      </div>

      <div className="flow-range-shell" style={rangeStyle}>
        <div className="flow-range-visual" aria-hidden>
          <span className="flow-range-active" />
          {Array.from({ length: markerCount }, (_, i) => (
            <span
              key={i}
              className="flow-range-node"
              style={{ left: `${(i / (markerCount - 1)) * 100}%` }}
            />
          ))}
          <span className="flow-range-thumb" />
        </div>

        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          onPointerUp={onRelease}
          onPointerCancel={onRelease}
          onKeyUp={onRelease}
          onBlur={onRelease}
          aria-label={label}
          className="flow-range-input disabled:cursor-not-allowed"
        />
      </div>
    </div>
  )
}

export default DotSlider
