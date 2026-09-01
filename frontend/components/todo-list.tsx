"use client"

import { useEffect, useState } from "react"
import { DotGlyph } from "@/components/dot-glyph"
import type { PetEvent } from "@/components/pet"

interface Todo {
  id: string
  text: string
  completed: boolean
}

interface TodoListProps {
  // the cat is watching; tell it what happened
  onEvent: (kind: PetEvent) => void
}

export function TodoList({ onEvent }: TodoListProps) {
  const [todos, setTodos] = useState<Todo[]>([])
  const [newTodo, setNewTodo] = useState("")
  const [score, setScore] = useState(0)
  const [multiplier, setMultiplier] = useState(1)
  const [mounted, setMounted] = useState(false)

  // read storage after mount so the server and first client render agree
  useEffect(() => {
    setMounted(true)
    try {
      const saved = localStorage.getItem("todos")
      if (saved) setTodos(JSON.parse(saved))
      const savedScore = localStorage.getItem("todoScore")
      if (savedScore) {
        const parsed = JSON.parse(savedScore)
        setScore(parsed.score)
        setMultiplier(parsed.multiplier)
      }
    } catch {
      // corrupt or unavailable storage is not worth losing the app over
    }
  }, [])

  useEffect(() => {
    if (mounted) localStorage.setItem("todos", JSON.stringify(todos))
  }, [todos, mounted])

  useEffect(() => {
    if (mounted) localStorage.setItem("todoScore", JSON.stringify({ score, multiplier }))
  }, [score, multiplier, mounted])

  const addTodo = (e: React.FormEvent) => {
    e.preventDefault()
    const text = newTodo.trim()
    if (!text) return
    setTodos((prev) => [...prev, { id: Date.now().toString(), text, completed: false }])
    setNewTodo("")
    onEvent("add")
  }

  const completeTodo = (id: string) => {
    setScore((prev) => prev + multiplier)
    setMultiplier((prev) => prev + 1)
    setTodos((prev) => prev.filter((todo) => todo.id !== id))
    // clearing the last one is worth more than finishing another one
    onEvent(todos.length === 1 ? "clear" : "complete")
  }

  const resetPoints = () => {
    setScore(0)
    setMultiplier(1)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <span className="label">Tasks</span>
        <div className="flex items-baseline gap-3">
          <span className="readout text-xs">
            {score} pts &middot; x{multiplier}
          </span>
          <button
            type="button"
            onClick={resetPoints}
            className="label hover:text-[var(--accent)] transition-colors"
            style={{ letterSpacing: "0.18em" }}
          >
            Reset
          </button>
        </div>
      </div>

      <form onSubmit={addTodo} className="flex gap-2">
        <input
          value={newTodo}
          onChange={(e) => setNewTodo(e.target.value)}
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
        {todos.map((todo) => (
          <li key={todo.id}>
            <button
              type="button"
              onClick={() => completeTodo(todo.id)}
              className="group flex w-full items-center gap-3 py-1.5 text-left text-sm hover:text-[var(--accent)] transition-colors"
            >
              <span className="relative shrink-0" style={{ width: 11, height: 11 }}>
                <DotGlyph name="box" dot={1} className="absolute inset-0 opacity-50" />
                <DotGlyph
                  name="check"
                  dot={1}
                  className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
                  color="var(--accent)"
                />
              </span>
              <span className="truncate">{todo.text}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default TodoList
