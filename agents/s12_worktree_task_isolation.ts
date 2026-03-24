// Harness: directory isolation -- parallel execution lanes that never collide.
/*
s12_worktree_task_isolation.ts- Worktree + Task Isolation

Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.

    .tasks/task_12.json
      {
        "id": 12,
        "subject": "Implement auth refactor",
        "status": "in_progress",
        "worktree": "auth-refactor"
      }

    .worktrees/index.json
      {
        "worktrees": [
          {
            "name": "auth-refactor",
            "path": ".../.worktrees/auth-refactor",
            "branch": "wt/auth-refactor",
            "task_id": 12,
            "status": "active"
          }
        ]
      }

Key insight: "Isolate by directory, coordinate by task ID."
 */

import Anthropic from "@anthropic-ai/sdk";
import { config as loadEnv } from "dotenv";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, relative, resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

loadEnv({ override: true, quiet: true });

if (process.env.ANTHROPIC_BASE_URL) {
  delete process.env.ANTHROPIC_AUTH_TOKEN;
}

const WORKDIR = resolve(process.cwd());
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || undefined,
});
const MODEL = process.env.MODEL_ID ?? "claude-sonnet-4-20250514";

function detect_repo_root(cwd: string): string | null {
  /* Return git repo root if cwd is inside a repo, else None. */
  try {
    const result = spawnSync("git", ["rev-parse", "--show-toplevel"], {
      cwd,
      encoding: "utf8",
      timeout: 10_000,
    });
    if (result.status !== 0) {
      return null;
    }
    const root = resolve(result.stdout.trim());
    return existsSync(root) ? root : null;
  } catch {
    return null;
  }
}

const REPO_ROOT = detect_repo_root(WORKDIR) ?? WORKDIR;

const SYSTEM =
  `You are a coding agent at ${WORKDIR}. ` +
  "Use task + worktree tools for multi-task work. " +
  "For parallel or risky changes: create tasks, allocate worktree lanes, " +
  "run commands in those lanes, then choose keep/remove for closeout. " +
  "Use worktree_events when you need lifecycle visibility.";

// -- EventBus: append-only lifecycle events for observability --
class EventBus {
  path: string;

  constructor(event_log_path: string) {
    this.path = event_log_path;
    mkdirSync(dirname(this.path), { recursive: true });
    if (!existsSync(this.path)) {
      writeFileSync(this.path, "", "utf8");
    }
  }

  emit(
    event: string,
    task: Record<string, unknown> | null = null,
    worktree: Record<string, unknown> | null = null,
    error: string | null = null
  ): void {
    const payload: Record<string, unknown> = {
      event,
      ts: Date.now() / 1000,
      task: task || {},
      worktree: worktree || {},
    };
    if (error) {
      payload.error = error;
    }
    appendFileSync(this.path, `${JSON.stringify(payload)}\n`, "utf8");
  }

  list_recent(limit = 20): string {
    const n = Math.max(1, Math.min(Number(limit || 20), 200));
    const lines = readFileSync(this.path, "utf8").split(/\r?\n/).filter(Boolean);
    const recent = lines.slice(-n);
    const items = [];
    for (const line of recent) {
      try {
        items.push(JSON.parse(line));
      } catch {
        items.push({ event: "parse_error", raw: line });
      }
    }
    return JSON.stringify(items, null, 2);
  }
}

// -- TaskManager: persistent task board with optional worktree binding --
class TaskManager {
  dir: string;
  _next_id: number;

  constructor(tasks_dir: string) {
    this.dir = tasks_dir;
    mkdirSync(this.dir, { recursive: true });
    this._next_id = this._max_id() + 1;
  }

  _max_id(): number {
    const ids: number[] = [];
    for (const file of readdirSync(this.dir).filter((entry) => /^task_\d+\.json$/.test(entry))) {
      try {
        ids.push(Number(file.replace(/^task_(\d+)\.json$/, "$1")));
      } catch {
        continue;
      }
    }
    return ids.length ? Math.max(...ids) : 0;
  }

  _path(task_id: number): string {
    return join(this.dir, `task_${task_id}.json`);
  }

  _load(task_id: number): Record<string, unknown> {
    const path = this._path(task_id);
    if (!existsSync(path)) {
      throw new Error(`Task ${task_id} not found`);
    }
    return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  }

  _save(task: Record<string, unknown>): void {
    writeFileSync(this._path(Number(task.id)), `${JSON.stringify(task, null, 2)}\n`, "utf8");
  }

  create(subject: string, description = ""): string {
    const task: Record<string, unknown> = {
      id: this._next_id,
      subject,
      description,
      status: "pending",
      owner: "",
      worktree: "",
      blockedBy: [],
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
    };
    this._save(task);
    this._next_id += 1;
    return JSON.stringify(task, null, 2);
  }

  get(task_id: number): string {
    return JSON.stringify(this._load(task_id), null, 2);
  }

  exists(task_id: number): boolean {
    return existsSync(this._path(task_id));
  }

  update(task_id: number, status?: string, owner?: string): string {
    const task = this._load(task_id);
    if (status) {
      if (!["pending", "in_progress", "completed"].includes(status)) {
        throw new Error(`Invalid status: ${status}`);
      }
      task.status = status;
    }
    if (owner !== undefined) {
      task.owner = owner;
    }
    task.updated_at = Date.now() / 1000;
    this._save(task);
    return JSON.stringify(task, null, 2);
  }

  bind_worktree(task_id: number, worktree: string, owner = ""): string {
    const task = this._load(task_id);
    task.worktree = worktree;
    if (owner) {
      task.owner = owner;
    }
    if (task.status === "pending") {
      task.status = "in_progress";
    }
    task.updated_at = Date.now() / 1000;
    this._save(task);
    return JSON.stringify(task, null, 2);
  }

  unbind_worktree(task_id: number): string {
    const task = this._load(task_id);
    task.worktree = "";
    task.updated_at = Date.now() / 1000;
    this._save(task);
    return JSON.stringify(task, null, 2);
  }

  list_all(): string {
    const tasks: Array<Record<string, unknown>> = [];
    for (const file of readdirSync(this.dir).filter((entry) => /^task_\d+\.json$/.test(entry)).sort()) {
      tasks.push(JSON.parse(readFileSync(join(this.dir, file), "utf8")) as Record<string, unknown>);
    }
    if (!tasks.length) {
      return "No tasks.";
    }
    const lines = [];
    for (const task of tasks) {
      const marker =
        ({ pending: "[ ]", in_progress: "[>]", completed: "[x]" }[String(task.status)] ?? "[?]");
      const owner = task.owner ? ` owner=${String(task.owner)}` : "";
      const wt = task.worktree ? ` wt=${String(task.worktree)}` : "";
      lines.push(`${marker} #${task.id}: ${String(task.subject ?? "")}${owner}${wt}`);
    }
    return lines.join("\n");
  }
}

const TASKS = new TaskManager(join(REPO_ROOT, ".tasks"));
const EVENTS = new EventBus(join(REPO_ROOT, ".worktrees", "events.jsonl"));

// -- WorktreeManager: create/list/run/remove git worktrees + lifecycle index --
class WorktreeManager {
  repo_root: string;
  tasks: TaskManager;
  events: EventBus;
  dir: string;
  index_path: string;
  git_available: boolean;

  constructor(repo_root: string, tasks: TaskManager, events: EventBus) {
    this.repo_root = repo_root;
    this.tasks = tasks;
    this.events = events;
    this.dir = join(repo_root, ".worktrees");
    mkdirSync(this.dir, { recursive: true });
    this.index_path = join(this.dir, "index.json");
    if (!existsSync(this.index_path)) {
      writeFileSync(this.index_path, `${JSON.stringify({ worktrees: [] }, null, 2)}\n`, "utf8");
    }
    this.git_available = this._is_git_repo();
  }

  _is_git_repo(): boolean {
    try {
      const result = spawnSync("git", ["rev-parse", "--is-inside-work-tree"], {
        cwd: this.repo_root,
        encoding: "utf8",
        timeout: 10_000,
      });
      return result.status === 0;
    } catch {
      return false;
    }
  }

  _run_git(args: string[]): string {
    if (!this.git_available) {
      throw new Error("Not in a git repository. worktree tools require git.");
    }
    const result = spawnSync("git", args, {
      cwd: this.repo_root,
      encoding: "utf8",
      timeout: 120_000,
    });
    if (result.status !== 0) {
      const msg = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
      throw new Error(msg || `git ${args.join(" ")} failed`);
    }
    return `${result.stdout ?? ""}${result.stderr ?? ""}`.trim() || "(no output)";
  }

  _load_index(): { worktrees: Array<Record<string, unknown>> } {
    return JSON.parse(readFileSync(this.index_path, "utf8")) as {
      worktrees: Array<Record<string, unknown>>;
    };
  }

  _save_index(data: { worktrees: Array<Record<string, unknown>> }): void {
    writeFileSync(this.index_path, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  }

  _find(name: string): Record<string, unknown> | null {
    const idx = this._load_index();
    for (const wt of idx.worktrees) {
      if (wt.name === name) {
        return wt;
      }
    }
    return null;
  }

  _validate_name(name: string): void {
    if (!/^[A-Za-z0-9._-]{1,40}$/.test(name || "")) {
      throw new Error("Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -");
    }
  }

  create(name: string, task_id?: number, base_ref = "HEAD"): string {
    this._validate_name(name);
    if (this._find(name)) {
      throw new Error(`Worktree '${name}' already exists in index`);
    }
    if (task_id !== undefined && !this.tasks.exists(task_id)) {
      throw new Error(`Task ${task_id} not found`);
    }

    const path = join(this.dir, name);
    const branch = `wt/${name}`;
    this.events.emit(
      "worktree.create.before",
      task_id !== undefined ? { id: task_id } : {},
      { name, base_ref }
    );
    try {
      this._run_git(["worktree", "add", "-b", branch, path, base_ref]);

      const entry: Record<string, unknown> = {
        name,
        path,
        branch,
        task_id,
        status: "active",
        created_at: Date.now() / 1000,
      };

      const idx = this._load_index();
      idx.worktrees.push(entry);
      this._save_index(idx);

      if (task_id !== undefined) {
        this.tasks.bind_worktree(task_id, name);
      }

      this.events.emit(
        "worktree.create.after",
        task_id !== undefined ? { id: task_id } : {},
        { name, path, branch, status: "active" }
      );
      return JSON.stringify(entry, null, 2);
    } catch (error) {
      this.events.emit(
        "worktree.create.failed",
        task_id !== undefined ? { id: task_id } : {},
        { name, base_ref },
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  list_all(): string {
    const wts = this._load_index().worktrees;
    if (!wts.length) {
      return "No worktrees in index.";
    }
    const lines = [];
    for (const wt of wts) {
      const suffix = wt.task_id ? ` task=${wt.task_id}` : "";
      lines.push(
        `[${String(wt.status ?? "unknown")}] ${String(wt.name)} -> ` +
          `${String(wt.path)} (${String(wt.branch ?? "-")})${suffix}`
      );
    }
    return lines.join("\n");
  }

  status(name: string): string {
    const wt = this._find(name);
    if (!wt) {
      return `Error: Unknown worktree '${name}'`;
    }
    const path = String(wt.path ?? "");
    if (!existsSync(path)) {
      return `Error: Worktree path missing: ${path}`;
    }
    const result = spawnSync("git", ["status", "--short", "--branch"], {
      cwd: path,
      encoding: "utf8",
      timeout: 60_000,
    });
    const text = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
    return text || "Clean worktree";
  }

  run(name: string, command: string): string {
    const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
    if (dangerous.some((item) => command.includes(item))) {
      return "Error: Dangerous command blocked";
    }

    const wt = this._find(name);
    if (!wt) {
      return `Error: Unknown worktree '${name}'`;
    }
    const path = String(wt.path ?? "");
    if (!existsSync(path)) {
      return `Error: Worktree path missing: ${path}`;
    }

    try {
      const result = spawnSync(command, {
        shell: true,
        cwd: path,
        encoding: "utf8",
        timeout: 300_000,
      });
      if (result.error) {
        if ((result.error as NodeJS.ErrnoException).code === "ETIMEDOUT") {
          return "Error: Timeout (300s)";
        }
        return `Error: ${result.error instanceof Error ? result.error.message : String(result.error)}`;
      }
      const out = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
      return out ? out.slice(0, 50_000) : "(no output)";
    } catch (error) {
      return `Error: ${error instanceof Error ? error.message : String(error)}`;
    }
  }

  remove(name: string, force = false, complete_task = false): string {
    const wt = this._find(name);
    if (!wt) {
      return `Error: Unknown worktree '${name}'`;
    }

    this.events.emit(
      "worktree.remove.before",
      wt.task_id !== undefined ? { id: wt.task_id } : {},
      { name, path: wt.path }
    );
    try {
      const args = ["worktree", "remove"];
      if (force) {
        args.push("--force");
      }
      args.push(String(wt.path));
      this._run_git(args);

      if (complete_task && wt.task_id !== undefined) {
        const task_id = Number(wt.task_id);
        const before = JSON.parse(this.tasks.get(task_id)) as Record<string, unknown>;
        this.tasks.update(task_id, "completed");
        this.tasks.unbind_worktree(task_id);
        this.events.emit(
          "task.completed",
          { id: task_id, subject: String(before.subject ?? ""), status: "completed" },
          { name }
        );
      }

      const idx = this._load_index();
      for (const item of idx.worktrees) {
        if (item.name === name) {
          item.status = "removed";
          item.removed_at = Date.now() / 1000;
        }
      }
      this._save_index(idx);

      this.events.emit(
        "worktree.remove.after",
        wt.task_id !== undefined ? { id: wt.task_id } : {},
        { name, path: wt.path, status: "removed" }
      );
      return `Removed worktree '${name}'`;
    } catch (error) {
      this.events.emit(
        "worktree.remove.failed",
        wt.task_id !== undefined ? { id: wt.task_id } : {},
        { name, path: wt.path },
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  keep(name: string): string {
    const wt = this._find(name);
    if (!wt) {
      return `Error: Unknown worktree '${name}'`;
    }

    const idx = this._load_index();
    let kept: Record<string, unknown> | null = null;
    for (const item of idx.worktrees) {
      if (item.name === name) {
        item.status = "kept";
        item.kept_at = Date.now() / 1000;
        kept = item;
      }
    }
    this._save_index(idx);

    this.events.emit(
      "worktree.keep",
      wt.task_id !== undefined ? { id: wt.task_id } : {},
      { name, path: wt.path, status: "kept" }
    );
    return kept ? JSON.stringify(kept, null, 2) : `Error: Unknown worktree '${name}'`;
  }
}

const WORKTREES = new WorktreeManager(REPO_ROOT, TASKS, EVENTS);

// -- Base tools (kept minimal, same style as previous sessions) --
function safe_path(p: string): string {
  const path = resolve(WORKDIR, p);
  const rel = relative(WORKDIR, path);
  if (rel.startsWith("..")) {
    throw new Error(`Path escapes workspace: ${p}`);
  }
  return path;
}

function run_bash(command: string): string {
  const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
  if (dangerous.some((item) => command.includes(item))) {
    return "Error: Dangerous command blocked";
  }
  try {
    const result = spawnSync(command, {
      shell: true,
      cwd: WORKDIR,
      encoding: "utf8",
      timeout: 120_000,
    });
    if (result.error) {
      if ((result.error as NodeJS.ErrnoException).code === "ETIMEDOUT") {
        return "Error: Timeout (120s)";
      }
      return `Error: ${result.error instanceof Error ? result.error.message : String(result.error)}`;
    }
    const out = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
    return out ? out.slice(0, 50_000) : "(no output)";
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function run_read(path: string, limit?: number): string {
  try {
    let lines = readFileSync(safe_path(path), "utf8").split(/\r?\n/);
    if (limit && limit < lines.length) {
      lines = lines.slice(0, limit).concat(`... (${lines.length - limit} more)`);
    }
    return lines.join("\n").slice(0, 50_000);
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function run_write(path: string, content: string): string {
  try {
    const fp = safe_path(path);
    mkdirSync(dirname(fp), { recursive: true });
    writeFileSync(fp, content, "utf8");
    return `Wrote ${content.length} bytes`;
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function run_edit(path: string, old_text: string, new_text: string): string {
  try {
    const fp = safe_path(path);
    const content = readFileSync(fp, "utf8");
    if (!content.includes(old_text)) {
      return `Error: Text not found in ${path}`;
    }
    writeFileSync(fp, content.replace(old_text, new_text), "utf8");
    return `Edited ${path}`;
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

const TOOL_HANDLERS = {
  bash: (args: Record<string, unknown>) => run_bash(String(args.command ?? "")),
  read_file: (args: Record<string, unknown>) =>
    run_read(String(args.path ?? ""), Number(args.limit || 0) || undefined),
  write_file: (args: Record<string, unknown>) =>
    run_write(String(args.path ?? ""), String(args.content ?? "")),
  edit_file: (args: Record<string, unknown>) =>
    run_edit(
      String(args.path ?? ""),
      String(args.old_text ?? ""),
      String(args.new_text ?? "")
    ),
  task_create: (args: Record<string, unknown>) =>
    TASKS.create(String(args.subject ?? ""), String(args.description ?? "")),
  task_list: (_args: Record<string, unknown>) => TASKS.list_all(),
  task_get: (args: Record<string, unknown>) => TASKS.get(Number(args.task_id)),
  task_update: (args: Record<string, unknown>) =>
    TASKS.update(
      Number(args.task_id),
      args.status ? String(args.status) : undefined,
      args.owner !== undefined ? String(args.owner) : undefined
    ),
  task_bind_worktree: (args: Record<string, unknown>) =>
    TASKS.bind_worktree(
      Number(args.task_id),
      String(args.worktree ?? ""),
      String(args.owner ?? "")
    ),
  worktree_create: (args: Record<string, unknown>) =>
    WORKTREES.create(
      String(args.name ?? ""),
      args.task_id !== undefined ? Number(args.task_id) : undefined,
      String(args.base_ref ?? "HEAD")
    ),
  worktree_list: (_args: Record<string, unknown>) => WORKTREES.list_all(),
  worktree_status: (args: Record<string, unknown>) =>
    WORKTREES.status(String(args.name ?? "")),
  worktree_run: (args: Record<string, unknown>) =>
    WORKTREES.run(String(args.name ?? ""), String(args.command ?? "")),
  worktree_keep: (args: Record<string, unknown>) =>
    WORKTREES.keep(String(args.name ?? "")),
  worktree_remove: (args: Record<string, unknown>) =>
    WORKTREES.remove(
      String(args.name ?? ""),
      Boolean(args.force),
      Boolean(args.complete_task)
    ),
  worktree_events: (args: Record<string, unknown>) =>
    EVENTS.list_recent(Number(args.limit || 20)),
};

const TOOLS = [
  {
    name: "bash",
    description: "Run a shell command in the current workspace (blocking).",
    input_schema: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
  {
    name: "read_file",
    description: "Read file contents.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        limit: { type: "integer" },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to file.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "edit_file",
    description: "Replace exact text in file.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        old_text: { type: "string" },
        new_text: { type: "string" },
      },
      required: ["path", "old_text", "new_text"],
    },
  },
  {
    name: "task_create",
    description: "Create a new task on the shared task board.",
    input_schema: {
      type: "object",
      properties: {
        subject: { type: "string" },
        description: { type: "string" },
      },
      required: ["subject"],
    },
  },
  {
    name: "task_list",
    description: "List all tasks with status, owner, and worktree binding.",
    input_schema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "task_get",
    description: "Get task details by ID.",
    input_schema: {
      type: "object",
      properties: {
        task_id: { type: "integer" },
      },
      required: ["task_id"],
    },
  },
  {
    name: "task_update",
    description: "Update task status or owner.",
    input_schema: {
      type: "object",
      properties: {
        task_id: { type: "integer" },
        status: {
          type: "string",
          enum: ["pending", "in_progress", "completed"],
        },
        owner: { type: "string" },
      },
      required: ["task_id"],
    },
  },
  {
    name: "task_bind_worktree",
    description: "Bind a task to a worktree name.",
    input_schema: {
      type: "object",
      properties: {
        task_id: { type: "integer" },
        worktree: { type: "string" },
        owner: { type: "string" },
      },
      required: ["task_id", "worktree"],
    },
  },
  {
    name: "worktree_create",
    description: "Create a git worktree and optionally bind it to a task.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
        task_id: { type: "integer" },
        base_ref: { type: "string" },
      },
      required: ["name"],
    },
  },
  {
    name: "worktree_list",
    description: "List worktrees tracked in .worktrees/index.json.",
    input_schema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "worktree_status",
    description: "Show git status for one worktree.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
      },
      required: ["name"],
    },
  },
  {
    name: "worktree_run",
    description: "Run a shell command in a named worktree directory.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
        command: { type: "string" },
      },
      required: ["name", "command"],
    },
  },
  {
    name: "worktree_remove",
    description: "Remove a worktree and optionally mark its bound task completed.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
        force: { type: "boolean" },
        complete_task: { type: "boolean" },
      },
      required: ["name"],
    },
  },
  {
    name: "worktree_keep",
    description: "Mark a worktree as kept in lifecycle state without removing it.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
      },
      required: ["name"],
    },
  },
  {
    name: "worktree_events",
    description: "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
    input_schema: {
      type: "object",
      properties: {
        limit: { type: "integer" },
      },
    },
  },
];

async function agent_loop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  while (true) {
    const response = await client.messages.create({
      model: MODEL,
      system: SYSTEM,
      messages: messages as never,
      tools: TOOLS as never,
      max_tokens: 8000,
    });

    messages.push({ role: "assistant", content: response.content });
    if (response.stop_reason !== "tool_use") {
      return;
    }

    const results: Array<Record<string, unknown>> = [];
    for (const block of response.content as unknown[]) {
      if (
        !block ||
        typeof block !== "object" ||
        (block as { type?: string }).type !== "tool_use"
      ) {
        continue;
      }

      const tool_use = block as {
        id: string;
        name: keyof typeof TOOL_HANDLERS | string;
        input: Record<string, unknown>;
      };
      const handler = TOOL_HANDLERS[tool_use.name as keyof typeof TOOL_HANDLERS];

      let output = "";
      try {
        output = handler ? handler(tool_use.input) : `Unknown tool: ${tool_use.name}`;
      } catch (error) {
        output = `Error: ${error instanceof Error ? error.message : String(error)}`;
      }

      console.log(`> ${tool_use.name}: ${String(output).slice(0, 200)}`);
      results.push({
        type: "tool_result",
        tool_use_id: tool_use.id,
        content: String(output),
      });
    }
    messages.push({ role: "user", content: results });
  }
}

console.log(`Repo root for s12: ${REPO_ROOT}`);
if (!WORKTREES.git_available) {
  console.log("Note: Not in a git repo. worktree_* tools will return errors.");
}

const history: Array<{ role: "user" | "assistant"; content: unknown }> = [];
const rl = createInterface({ input, output });

try {
  while (true) {
    const query = await rl.question("\u001b[36ms12 >> \u001b[0m");
    const trimmed = query.trim();
    if (!trimmed || trimmed.toLowerCase() === "q" || trimmed.toLowerCase() === "exit") {
      break;
    }

    history.push({ role: "user", content: query });
    await agent_loop(history);

    const response_content = history[history.length - 1]?.content;
    if (Array.isArray(response_content)) {
      for (const block of response_content) {
        if (
          block &&
          typeof block === "object" &&
          "text" in block &&
          typeof (block as { text?: string }).text === "string"
        ) {
          console.log((block as { text: string }).text);
        }
      }
    }
    console.log();
  }
} finally {
  rl.close();
}
