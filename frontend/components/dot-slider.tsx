"use client"

import { useId } from "react"

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

// a value shown the only way this interface knows how to show anything: as a
// run of lit dots. the real control underneath is a native range input, kept
// transparent on top so keyboard, touch and screen readers all still work.
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
  const lit = Math.round(fraction * (segments - 1))

  return (
    <div className={disabled ? "opacity-40" : undefined}>
      <div className="flex items-baseline justify-between mb-2">
        <label htmlFor={id} className="label">
          {label}
        </label>
        <span className="readout text-xs">{readout}</span>
      </div>

      <div className="relative h-5 flex items-center">
        <div className="flex w-full items-center justify-between pointer-events-none">
          {Array.from({ length: segments }, (_, i) => (
            <span
              key={i}
              className="rounded-full transition-colors duration-75"
              style={{
                width: i === lit ? 7 : 5,
                height: i === lit ? 7 : 5,
                background:
                  i === lit
                    ? "var(--accent)"
                    : i < lit
                      ? "var(--dot-3)"
                      : "var(--dot-1)",
              }}
            />
          ))}
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
          onKeyUp={onRelease}
          aria-label={label}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        />
      </div>
    </div>
  )
}

export default DotSlider
