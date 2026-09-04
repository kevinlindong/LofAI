"use client"

import { useEffect, useMemo, useState } from "react"
import { DotGlyph } from "@/components/dot-glyph"
import type { PetEvent } from "@/components/pet"

interface Todo {
  id: string
  text: string
  done: boolean
}

interface TodoListProps {
  // the cat is watching; tell it what happened
  onEvent: (kind: PetEvent) => void
}

// what came back from storage, which may have been written by an older build
type StoredTodo = Partial<Todo> & { completed?: boolean }

// tasks are never thrown away when they are finished, only struck through and
// moved down. the point of a list you keep beside a focus timer is to be able
// to look at it at the end of a session and see what you got through - a list
// that deletes the evidence is a list that only ever shows you what is left.
export function TodoList({ onEvent }: TodoListProps) {
  const [todos, setTodos] = useState<Todo[]>([])
  const [draft, setDraft] = useState("")
  const [mounted, setMounted] = useState(false)

  // read storage after mount so the server and first client render agree
  useEffect(() => {
    setMounted(true)
    try {
      const saved = localStorage.getItem("todos")
      if (saved) {
        const parsed: StoredTodo[] = JSON.parse(saved)
        setTodos(
          parsed
            .filter((t) => typeof t?.text === "string")
            .map((t, i) => ({
              id: t.id ?? `${Date.now()}-${i}`,
              text: t.text as string,
              // an older build called this `completed` and deleted the task
              // rather than keeping it, so in practice this is always false -
              // but a list that silently drops the user's tasks on an upgrade
              // is not worth the two lines it saves
              done: t.done ?? t.completed ?? false,
            })),
        )
      }
      // the score and multiplier this list used to keep are gone. they counted
      // up for ever and reset to nothing, which measured neither progress nor
      // anything else.
      localStorage.removeItem("todoScore")
    } catch {
      // corrupt or unavailable storage is not worth losing the app over
    }
  }, [])

  useEffect(() => {
    if (mounted) localStorage.setItem("todos", JSON.stringify(todos))
  }, [todos, mounted])

  const done = todos.filter((t) => t.done).length
  const left = todos.length - done

  // finished tasks sink. sort is stable, so within each half the order is
  // still the order they were typed in.
  const ordered = useMemo(
    () => [...todos].sort((a, b) => Number(a.done) - Number(b.done)),
    [todos],
  )

  const add = (e: React.FormEvent) => {
    e.preventDefault()
    const text = draft.trim()
    if (!text) return
    setTodos((prev) => [...prev, { id: `${Date.now()}`, text, done: false }])
    setDraft("")
    onEvent("add")
  }

  const toggle = (id: string) => {
    const todo = todos.find((t) => t.id === id)
    if (!todo) return
    setTodos((prev) => prev.map((t) => (t.id === id ? { ...t, done: !t.done } : t)))
    // clearing the last one open is worth more than finishing another one
    if (todo.done) onEvent("undo")
    else onEvent(left === 1 ? "clear" : "complete")
  }

  const remove = (id: string) => {
    setTodos((prev) => prev.filter((t) => t.id !== id))
  }

  const clearDone = () => {
    setTodos((prev) => prev.filter((t) => !t.done))
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="label">Tasks</span>
        <div className="flex items-baseline gap-3">
          {mounted && todos.length > 0 && (
            <span className="readout text-xs">
              {left === 0 ? "all done" : `${done}/${todos.length} done`}
            </span>
          )}
          {done > 0 && (
            <button
              type="button"
              onClick={clearDone}
              className="label transition-colors hover:text-[var(--accent)]"
              style={{ letterSpacing: "0.18em" }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <form onSubmit={add} className="flex gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="what needs doing"
          maxLength={80}
          aria-label="New task"
          className="panel-inset min-w-0 flex-1 bg-transparent px-3 py-2 text-sm outline-none placeholder:text-[var(--text-dim)] focus:border-[var(--accent)]"
        />
        <button type="submit" className="key px-3 py-2 text-xs" aria-label="Add task">
          <DotGlyph name="plus" dot={2} />
        </button>
      </form>

      <ul className="dot-scroll min-h-0 flex-1 overflow-y-auto pr-1">
        {todos.length === 0 && (
          <li className="label py-3" style={{ letterSpacing: "0.14em" }}>
            {mounted ? "nothing queued" : ""}
          </li>
        )}
        {ordered.map((todo) => (
          <li key={todo.id} className="group flex items-center gap-3 py-1.5">
            <button
              type="button"
              onClick={() => toggle(todo.id)}
              aria-pressed={todo.done}
              className="flex min-w-0 flex-1 items-center gap-3 text-left text-sm transition-colors hover:text-[var(--accent)]"
            >
              <span className="relative shrink-0" style={{ width: 11, height: 11 }}>
                <DotGlyph
                  name="box"
                  dot={1}
                  className={`absolute inset-0 ${todo.done ? "opacity-25" : "opacity-50"}`}
                />
                <DotGlyph
                  name="check"
                  dot={1}
                  className={`absolute inset-0 transition-opacity ${
                    todo.done ? "opacity-100" : "opacity-0 group-hover:opacity-60"
                  }`}
                  color="var(--accent)"
                />
              </span>
              <span
                className="truncate"
                style={
                  todo.done
                    ? { textDecoration: "line-through", color: "var(--text-dim)" }
                    : undefined
                }
              >
                {todo.text}
              </span>
            </button>
            <button
              type="button"
              onClick={() => remove(todo.id)}
              aria-label={`Delete ${todo.text}`}
              className="shrink-0 opacity-30 transition-opacity hover:opacity-100 hover:text-[var(--bad)]"
            >
              <DotGlyph name="cross" dot={1} />
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default TodoList
