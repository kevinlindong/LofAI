import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import { RotateCcw, Trophy } from "lucide-react"

interface Todo {
  id: string
  text: string
  completed: boolean
}

export function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([])
  const [newTodo, setNewTodo] = useState("")
  const [score, setScore] = useState(0)
  const [multiplier, setMultiplier] = useState(1)
  const [mounted, setMounted] = useState(false)

  // Load from localStorage after mount to prevent hydration errors
  useEffect(() => {
    setMounted(true)
    const saved = localStorage.getItem("todos")
    if (saved) {
      setTodos(JSON.parse(saved))
    }
    const savedScore = localStorage.getItem("todoScore")
    if (savedScore) {
      const { score: savedScoreValue, multiplier: savedMultiplier } = JSON.parse(savedScore)
      setScore(savedScoreValue)
      setMultiplier(savedMultiplier)
    }
  }, [])

  useEffect(() => {
    if (mounted) {
      localStorage.setItem("todos", JSON.stringify(todos))
    }
  }, [todos, mounted])

  useEffect(() => {
    if (mounted) {
      localStorage.setItem("todoScore", JSON.stringify({ score, multiplier }))
    }
  }, [score, multiplier, mounted])

  const addTodo = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newTodo.trim()) return

    setTodos((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        text: newTodo,
        completed: false,
      },
    ])
    setNewTodo("")
  }

  const completeTodo = (id: string) => {
    setScore((prev) => prev + 1 * multiplier)
    setMultiplier((prev) => prev + 1)
    setTodos((prev) => prev.filter((todo) => todo.id !== id))
  }

  const resetPoints = () => {
    setScore(0)
    setMultiplier(1)
    localStorage.setItem("todoScore", JSON.stringify({ score: 0, multiplier: 1 }))
  }

  return (
    <div className="w-full h-full relative flex flex-col items-center justify-start">
      <div className="z-10 bg-card rounded-full w-[90%] h-[95%] relative overflow-hidden">
        {/* Circular mask for bottom cutoff */}
        <div
          className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-[2000%] h-[0%] z-20"
          style={{
            background: "radial-gradient(ellipse at top, transparent 0%, var(--background) 70%)",
          }}
        />
        <div className="absolute inset-0 flex flex-col items-center pt-6">
          {/* Score and Reset Section */}
          <div className="w-[45%] flex items-center justify-between mb-1">
            <div className="flex items-center gap-0.5">
              <Trophy className="h-3 w-3 text-yellow-500" />
              <span className="text-xs font-medium dark:text-gray-300">
                {score} (x{multiplier})
              </span>
            </div>
            <Button variant="ghost" size="sm" onClick={resetPoints} className="h-6 w-6 p-0">
              <RotateCcw className="h-3 w-3" />
            </Button>
          </div>

          {/* Input Form */}
          <form onSubmit={addTodo} className="flex gap-1 w-[66%] mb-2">
            <Input
              value={newTodo}
              onChange={(e) => setNewTodo(e.target.value)}
              placeholder="Add a task..."
              className="flex-grow text-xs h-6 px-2 py-0.5 min-w-0"
            />
            <Button type="submit" size="sm" className="h-6 text-xs px-2 py-0 shrink-0">
              Add
            </Button>
          </form>

          {/* Todo List with ScrollArea */}
          <div className="w-[66%] flex-1 overflow-hidden relative h-[calc(100%-100px)]">
            <ScrollArea className="h-full w-full">
              <div className="space-y-1 pr-4 pb-4 min-h-full">
                <div className="min-h-[200px]">
                  {todos.map((todo) => (
                    <div key={todo.id} className="flex items-center gap-1 py-0.5">
                      <Checkbox
                        checked={todo.completed}
                        onCheckedChange={() => completeTodo(todo.id)}
                        id={todo.id}
                        className="w-3 h-3 shrink-0"
                      />
                      <label htmlFor={todo.id} className="flex-grow text-xs cursor-pointer dark:text-gray-300 truncate">
                        {todo.text}
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TodoList