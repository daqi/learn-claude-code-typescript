// Harness: all mechanisms combined -- the complete cockpit for the model.
/*
s_full.ts- Full Reference Agent

Capstone implementation combining every mechanism from s01-s11.
Session s12 (task-aware worktree isolation) is taught separately.
NOT a teaching session -- this is the "put it all together" reference.

    +------------------------------------------------------------------+
    |                        FULL AGENT                                 |
    |                                                                   |
    |  System prompt (s05 skills, task-first + optional todo nag)      |
    |                                                                   |
    |  Before each LLM call:                                            |
    |  +--------------------+  +------------------+  +--------------+  |
    |  | Microcompact (s06) |  | Drain bg (s08)   |  | Check inbox  |  |
    |  | Auto-compact (s06) |  | notifications    |  | (s09)        |  |
    |  +--------------------+  +------------------+  +--------------+  |
    |                                                                   |
    |  Tool dispatch (s02 pattern):                                     |
    |  +--------+----------+----------+---------+-----------+          |
    |  | bash   | read     | write    | edit    | TodoWrite |          |
    |  | task   | load_sk  | compress | bg_run  | bg_check  |          |
    |  | t_crt  | t_get    | t_upd    | t_list  | spawn_tm  |          |
    |  | list_tm| send_msg | rd_inbox | bcast   | shutdown  |          |
    |  | plan   | idle     | claim    |         |           |          |
    |  +--------+----------+----------+---------+-----------+          |
    |                                                                   |
    |  Subagent (s04):  spawn -> work -> return summary                 |
    |  Teammate (s09):  spawn -> work -> idle -> auto-claim (s11)      |
    |  Shutdown (s10):  request_id handshake                            |
    |  Plan gate (s10): submit -> approve/reject                        |
    +------------------------------------------------------------------+

    REPL commands: /compact /tasks /team /inbox
 */

import Anthropic from "@anthropic-ai/sdk";
import { config as loadEnv } from "dotenv";
import {
  appendFileSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
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

const TEAM_DIR = join(WORKDIR, ".team");
const INBOX_DIR = join(TEAM_DIR, "inbox");
const TASKS_DIR = join(WORKDIR, ".tasks");
const SKILLS_DIR = join(WORKDIR, "skills");
const TRANSCRIPT_DIR = join(WORKDIR, ".transcripts");
const TOKEN_THRESHOLD = 100_000;
const POLL_INTERVAL = 5_000;
const IDLE_TIMEOUT = 60_000;

const VALID_MSG_TYPES = [
  "message",
  "broadcast",
  "shutdown_request",
  "shutdown_response",
  "plan_approval_response",
] as const;
type MessageType = (typeof VALID_MSG_TYPES)[number];

// === SECTION: base_tools ===
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
    return `Wrote ${content.length} bytes to ${path}`;
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

// === SECTION: todos (s03) ===
class TodoManager {
  items: Array<{ content: string; status: string; activeForm: string }> = [];

  update(items: Array<Record<string, unknown>>): string {
    const validated: Array<{ content: string; status: string; activeForm: string }> = [];
    let ip = 0;
    for (const [i, item] of items.entries()) {
      const content = String(item.content ?? "").trim();
      const status = String(item.status ?? "pending").toLowerCase();
      const activeForm = String(item.activeForm ?? "").trim();
      if (!content) {
        throw new Error(`Item ${i}: content required`);
      }
      if (!["pending", "in_progress", "completed"].includes(status)) {
        throw new Error(`Item ${i}: invalid status '${status}'`);
      }
      if (!activeForm) {
        throw new Error(`Item ${i}: activeForm required`);
      }
      if (status === "in_progress") {
        ip += 1;
      }
      validated.push({ content, status, activeForm });
    }
    if (validated.length > 20) {
      throw new Error("Max 20 todos");
    }
    if (ip > 1) {
      throw new Error("Only one in_progress allowed");
    }
    this.items = validated;
    return this.render();
  }

  render(): string {
    if (!this.items.length) {
      return "No todos.";
    }
    const lines = [];
    for (const item of this.items) {
      const marker = (
        { completed: "[x]", in_progress: "[>]", pending: "[ ]" } as Record<string, string>
      )[item.status] ?? "[?]";
      const suffix = item.status === "in_progress" ? ` <- ${item.activeForm}` : "";
      lines.push(`${marker} ${item.content}${suffix}`);
    }
    const done = this.items.filter((item) => item.status === "completed").length;
    lines.push(`\n(${done}/${this.items.length} completed)`);
    return lines.join("\n");
  }

  has_open_items(): boolean {
    return this.items.some((item) => item.status !== "completed");
  }
}

// === SECTION: subagent (s04) ===
async function run_subagent(prompt: string, agent_type = "Explore"): Promise<string> {
  const sub_tools: Array<Record<string, unknown>> = [
    {
      name: "bash",
      description: "Run command.",
      input_schema: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
    },
    {
      name: "read_file",
      description: "Read file.",
      input_schema: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
    },
  ];
  if (agent_type !== "Explore") {
    sub_tools.push(
      {
        name: "write_file",
        description: "Write file.",
        input_schema: {
          type: "object",
          properties: { path: { type: "string" }, content: { type: "string" } },
          required: ["path", "content"],
        },
      },
      {
        name: "edit_file",
        description: "Edit file.",
        input_schema: {
          type: "object",
          properties: {
            path: { type: "string" },
            old_text: { type: "string" },
            new_text: { type: "string" },
          },
          required: ["path", "old_text", "new_text"],
        },
      }
    );
  }

  const sub_handlers = {
    bash: (args: Record<string, unknown>) => run_bash(String(args.command ?? "")),
    read_file: (args: Record<string, unknown>) => run_read(String(args.path ?? "")),
    write_file: (args: Record<string, unknown>) =>
      run_write(String(args.path ?? ""), String(args.content ?? "")),
    edit_file: (args: Record<string, unknown>) =>
      run_edit(
        String(args.path ?? ""),
        String(args.old_text ?? ""),
        String(args.new_text ?? "")
      ),
  };

  const sub_msgs: Array<{ role: "user" | "assistant"; content: unknown }> = [
    { role: "user", content: prompt },
  ];
  let final_content: unknown[] = [];

  for (let i = 0; i < 30; i += 1) {
    const resp = await client.messages.create({
      model: MODEL,
      messages: sub_msgs as never,
      tools: sub_tools as never,
      max_tokens: 8000,
    });
    final_content = resp.content as unknown[];
    sub_msgs.push({ role: "assistant", content: resp.content });
    if (resp.stop_reason !== "tool_use") {
      break;
    }
    const results = [];
    for (const block of resp.content as unknown[]) {
      if ((block as { type?: string }).type !== "tool_use") {
        continue;
      }
      const toolBlock = block as {
        id: string;
        name: keyof typeof sub_handlers | string;
        input: Record<string, unknown>;
      };
      const handler = sub_handlers[toolBlock.name as keyof typeof sub_handlers];
      results.push({
        type: "tool_result",
        tool_use_id: toolBlock.id,
        content: String(handler ? handler(toolBlock.input) : "Unknown tool").slice(0, 50_000),
      });
    }
    sub_msgs.push({ role: "user", content: results });
  }

  return (
    final_content.map((block) => (block as { text?: string }).text ?? "").join("") ||
    "(no summary)"
  );
}

// === SECTION: skills (s05) ===
class SkillLoader {
  skills: Record<string, { meta: Record<string, string>; body: string }> = {};

  constructor(skills_dir: string) {
    this.walk(skills_dir);
  }

  walk(dir: string): void {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          this.walk(full);
          continue;
        }
        if (!entry.isFile() || entry.name !== "SKILL.md") {
          continue;
        }
        const text = readFileSync(full, "utf8");
        const match = text.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
        const meta: Record<string, string> = {};
        let body = text;
        if (match) {
          for (const line of match[1].trim().split(/\r?\n/)) {
            const index = line.indexOf(":");
            if (index !== -1) {
              meta[line.slice(0, index).trim()] = line.slice(index + 1).trim();
            }
          }
          body = match[2].trim();
        }
        const parts = full.split("/");
        const name = meta.name ?? parts[parts.length - 2];
        this.skills[name] = { meta, body };
      }
    } catch {
      return;
    }
  }

  descriptions(): string {
    if (!Object.keys(this.skills).length) {
      return "(no skills)";
    }
    return Object.entries(this.skills)
      .map(([name, skill]) => `  - ${name}: ${skill.meta.description ?? "-"}`)
      .join("\n");
  }

  load(name: string): string {
    const skill = this.skills[name];
    if (!skill) {
      return `Error: Unknown skill '${name}'. Available: ${Object.keys(this.skills).join(", ")}`;
    }
    return `<skill name="${name}">\n${skill.body}\n</skill>`;
  }
}

// === SECTION: compression (s06) ===
function estimate_tokens(messages: Array<{ role: "user" | "assistant"; content: unknown }>): number {
  return JSON.stringify(messages).length / 4;
}

function microcompact(messages: Array<{ role: "user" | "assistant"; content: unknown }>): void {
  const indices: Array<Record<string, unknown>> = [];
  for (const msg of messages) {
    if (msg.role !== "user" || !Array.isArray(msg.content)) {
      continue;
    }
    for (const part of msg.content) {
      if (
        part &&
        typeof part === "object" &&
        (part as { type?: string }).type === "tool_result"
      ) {
        indices.push(part as Record<string, unknown>);
      }
    }
  }
  if (indices.length <= 3) {
    return;
  }
  for (const part of indices.slice(0, -3)) {
    if (typeof part.content === "string" && part.content.length > 100) {
      part.content = "[cleared]";
    }
  }
}

async function auto_compact(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<Array<{ role: "user" | "assistant"; content: unknown }>> {
  mkdirSync(TRANSCRIPT_DIR, { recursive: true });
  const path = join(TRANSCRIPT_DIR, `transcript_${Date.now()}.jsonl`);
  writeFileSync(path, messages.map((msg) => JSON.stringify(msg)).join("\n"), "utf8");
  const resp = await client.messages.create({
    model: MODEL,
    messages: [
      {
        role: "user",
        content: `Summarize for continuity:\n${JSON.stringify(messages).slice(0, 80_000)}`,
      },
    ] as never,
    max_tokens: 2000,
  });
  const summary = (resp.content as unknown[])
    .map((block) => (block as { text?: string }).text ?? "")
    .join("");
  return [
    { role: "user", content: `[Compressed. Transcript: ${path}]\n${summary}` },
    { role: "assistant", content: "Understood. Continuing with summary context." },
  ];
}

// === SECTION: file_tasks (s07) ===
type TaskRecord = {
  id: number;
  subject: string;
  description: string;
  status: "pending" | "in_progress" | "completed" | "deleted";
  owner: string | null;
  blockedBy: number[];
  blocks: number[];
};

class TaskManager {
  constructor() {
    mkdirSync(TASKS_DIR, { recursive: true });
  }

  _next_id(): number {
    const ids = readdirSync(TASKS_DIR)
      .filter((entry) => /^task_\d+\.json$/.test(entry))
      .map((entry) => Number(entry.match(/\d+/)?.[0] ?? 0));
    return (ids.length ? Math.max(...ids) : 0) + 1;
  }

  _load(tid: number): TaskRecord {
    const path = join(TASKS_DIR, `task_${tid}.json`);
    if (!readdirSync(TASKS_DIR).includes(`task_${tid}.json`)) {
      throw new Error(`Task ${tid} not found`);
    }
    return JSON.parse(readFileSync(path, "utf8")) as TaskRecord;
  }

  _save(task: TaskRecord): void {
    writeFileSync(join(TASKS_DIR, `task_${task.id}.json`), JSON.stringify(task, null, 2), "utf8");
  }

  create(subject: string, description = ""): string {
    const task: TaskRecord = {
      id: this._next_id(),
      subject,
      description,
      status: "pending",
      owner: null,
      blockedBy: [],
      blocks: [],
    };
    this._save(task);
    return JSON.stringify(task, null, 2);
  }

  get(tid: number): string {
    return JSON.stringify(this._load(tid), null, 2);
  }

  update(
    tid: number,
    status?: string,
    add_blocked_by: number[] = [],
    add_blocks: number[] = [],
    owner?: string
  ): string {
    const task = this._load(tid);
    if (status) {
      task.status = status as TaskRecord["status"];
      if (status === "completed") {
        for (const file of readdirSync(TASKS_DIR).filter((entry) => /^task_\d+\.json$/.test(entry))) {
          const other = JSON.parse(readFileSync(join(TASKS_DIR, file), "utf8")) as TaskRecord;
          if (other.blockedBy.includes(tid)) {
            other.blockedBy = other.blockedBy.filter((value) => value !== tid);
            this._save(other);
          }
        }
      }
      if (status === "deleted") {
        writeFileSync(join(TASKS_DIR, `task_${tid}.json`), "", "utf8");
        return `Task ${tid} deleted`;
      }
    }
    if (owner !== undefined) {
      task.owner = owner;
    }
    if (add_blocked_by.length) {
      task.blockedBy = [...new Set([...task.blockedBy, ...add_blocked_by])];
    }
    if (add_blocks.length) {
      task.blocks = [...new Set([...task.blocks, ...add_blocks])];
    }
    this._save(task);
    return JSON.stringify(task, null, 2);
  }

  list_all(): string {
    const tasks = readdirSync(TASKS_DIR)
      .filter((entry) => /^task_\d+\.json$/.test(entry))
      .sort()
      .map((entry) => JSON.parse(readFileSync(join(TASKS_DIR, entry), "utf8")) as TaskRecord);
    if (!tasks.length) {
      return "No tasks.";
    }
    const lines = [];
    for (const task of tasks) {
      const marker =
        ({ pending: "[ ]", in_progress: "[>]", completed: "[x]" }[
          task.status
        ] ?? "[?]");
      const owner = task.owner ? ` @${task.owner}` : "";
      const blocked = task.blockedBy.length ? ` (blocked by: ${JSON.stringify(task.blockedBy)})` : "";
      lines.push(`${marker} #${task.id}: ${task.subject}${owner}${blocked}`);
    }
    return lines.join("\n");
  }

  claim(tid: number, owner: string): string {
    const task = this._load(tid);
    task.owner = owner;
    task.status = "in_progress";
    this._save(task);
    return `Claimed task #${tid} for ${owner}`;
  }
}

// === SECTION: background (s08) ===
class BackgroundManager {
  tasks: Record<string, { status: string; command: string; result: string | null }> = {};
  notifications: Array<{ task_id: string; status: string; result: string }> = [];

  run(command: string, timeout = 120): string {
    const tid = randomUUID().slice(0, 8);
    this.tasks[tid] = { status: "running", command, result: null };
    setTimeout(() => this._exec(tid, command, timeout), 0);
    return `Background task ${tid} started: ${command.slice(0, 80)}`;
  }

  _exec(tid: string, command: string, timeout: number): void {
    try {
      const result = spawnSync(command, {
        shell: true,
        cwd: WORKDIR,
        encoding: "utf8",
        timeout: timeout * 1000,
      });
      const output = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim().slice(0, 50_000);
      this.tasks[tid] = {
        status: "completed",
        command,
        result: output || "(no output)",
      };
    } catch (error) {
      this.tasks[tid] = {
        status: "error",
        command,
        result: error instanceof Error ? error.message : String(error),
      };
    }
    this.notifications.push({
      task_id: tid,
      status: this.tasks[tid].status,
      result: String(this.tasks[tid].result ?? "").slice(0, 500),
    });
  }

  check(tid?: string): string {
    if (tid) {
      const task = this.tasks[tid];
      return task ? `[${task.status}] ${task.result ?? "(running)"}` : `Unknown: ${tid}`;
    }
    const lines = Object.entries(this.tasks).map(
      ([id, task]) => `${id}: [${task.status}] ${task.command.slice(0, 60)}`
    );
    return lines.join("\n") || "No bg tasks.";
  }

  drain(): Array<{ task_id: string; status: string; result: string }> {
    return this.notifications.splice(0, this.notifications.length);
  }
}

// === SECTION: messaging (s09) ===
class MessageBus {
  constructor() {
    mkdirSync(INBOX_DIR, { recursive: true });
  }

  send(
    sender: string,
    to: string,
    content: string,
    msg_type: MessageType = "message",
    extra: Record<string, unknown> | null = null
  ): string {
    const msg: Record<string, unknown> = {
      type: msg_type,
      from: sender,
      content,
      timestamp: Date.now() / 1000,
    };
    if (extra) {
      Object.assign(msg, extra);
    }
    appendFileSync(join(INBOX_DIR, `${to}.jsonl`), `${JSON.stringify(msg)}\n`, "utf8");
    return `Sent ${msg_type} to ${to}`;
  }

  read_inbox(name: string): Array<Record<string, unknown>> {
    const path = join(INBOX_DIR, `${name}.jsonl`);
    try {
      const msgs = readFileSync(path, "utf8")
        .trim()
        .split(/\r?\n/)
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      writeFileSync(path, "", "utf8");
      return msgs;
    } catch {
      return [];
    }
  }

  broadcast(sender: string, content: string, names: string[]): string {
    let count = 0;
    for (const name of names) {
      if (name !== sender) {
        this.send(sender, name, content, "broadcast");
        count += 1;
      }
    }
    return `Broadcast to ${count} teammates`;
  }
}

// === SECTION: shutdown + plan tracking (s10) ===
const shutdown_requests: Record<string, { target: string; status: string }> = {};
const plan_requests: Record<string, { from: string; plan: string; status: string }> = {};

// === SECTION: team (s09/s11) ===
function scan_unclaimed_tasks(dir: string): TaskRecord[] {
  return readdirSync(dir)
    .filter((entry) => /^task_\d+\.json$/.test(entry))
    .sort()
    .map((entry) => JSON.parse(readFileSync(join(dir, entry), "utf8")) as TaskRecord)
    .filter((task) => task.status === "pending" && !task.owner && !task.blockedBy.length);
}

function claim_task(dir: string, tid: number, owner: string): string {
  const path = join(dir, `task_${tid}.json`);
  const task = JSON.parse(readFileSync(path, "utf8")) as TaskRecord;
  task.owner = owner;
  task.status = "in_progress";
  writeFileSync(path, JSON.stringify(task, null, 2), "utf8");
  return `Claimed task #${tid} for ${owner}`;
}

class TeammateManager {
  bus: MessageBus;
  task_mgr: TaskManager;
  config_path: string;
  config: { team_name: string; members: Array<{ name: string; role: string; status: string }> };
  threads: Record<string, Promise<void>> = {};

  constructor(bus: MessageBus, task_mgr: TaskManager) {
    mkdirSync(TEAM_DIR, { recursive: true });
    this.bus = bus;
    this.task_mgr = task_mgr;
    this.config_path = join(TEAM_DIR, "config.json");
    this.config = this._load();
  }

  _load(): { team_name: string; members: Array<{ name: string; role: string; status: string }> } {
    try {
      return JSON.parse(readFileSync(this.config_path, "utf8"));
    } catch {
      return { team_name: "default", members: [] };
    }
  }

  _save(): void {
    writeFileSync(this.config_path, JSON.stringify(this.config, null, 2), "utf8");
  }

  _find(name: string): { name: string; role: string; status: string } | null {
    for (const member of this.config.members) {
      if (member.name === name) {
        return member;
      }
    }
    return null;
  }

  spawn(name: string, role: string, prompt: string): string {
    let member = this._find(name);
    if (member) {
      if (!["idle", "shutdown"].includes(member.status)) {
        return `Error: '${name}' is currently ${member.status}`;
      }
      member.status = "working";
      member.role = role;
    } else {
      member = { name, role, status: "working" };
      this.config.members.push(member);
    }
    this._save();
    const thread = this._loop(name, role, prompt);
    this.threads[name] = thread;
    void thread;
    return `Spawned '${name}' (role: ${role})`;
  }

  _set_status(name: string, status: string): void {
    const member = this._find(name);
    if (member) {
      member.status = status;
      this._save();
    }
  }

  async _loop(name: string, role: string, prompt: string): Promise<void> {
    const team_name = this.config.team_name;
    const sys_prompt =
      `You are '${name}', role: ${role}, team: ${team_name}, at ${WORKDIR}. ` +
      "Use idle when done with current work. You may auto-claim tasks.";
    const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
      { role: "user", content: prompt },
    ];
    const tools: Array<Record<string, unknown>> = [
      {
        name: "bash",
        description: "Run command.",
        input_schema: {
          type: "object",
          properties: { command: { type: "string" } },
          required: ["command"],
        },
      },
      {
        name: "read_file",
        description: "Read file.",
        input_schema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
      {
        name: "write_file",
        description: "Write file.",
        input_schema: {
          type: "object",
          properties: { path: { type: "string" }, content: { type: "string" } },
          required: ["path", "content"],
        },
      },
      {
        name: "edit_file",
        description: "Edit file.",
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
        name: "send_message",
        description: "Send message.",
        input_schema: {
          type: "object",
          properties: { to: { type: "string" }, content: { type: "string" } },
          required: ["to", "content"],
        },
      },
      {
        name: "idle",
        description: "Signal no more work.",
        input_schema: { type: "object", properties: {} },
      },
      {
        name: "claim_task",
        description: "Claim task by ID.",
        input_schema: {
          type: "object",
          properties: { task_id: { type: "integer" } },
          required: ["task_id"],
        },
      },
    ];

    while (true) {
      // -- WORK PHASE --
      for (let i = 0; i < 50; i += 1) {
        const inbox = this.bus.read_inbox(name);
        for (const msg of inbox) {
          if (msg.type === "shutdown_request") {
            this._set_status(name, "shutdown");
            return;
          }
          messages.push({ role: "user", content: JSON.stringify(msg) });
        }
        let response;
        try {
          response = await client.messages.create({
            model: MODEL,
            system: sys_prompt,
            messages: messages as never,
            tools: tools as never,
            max_tokens: 8000,
          });
        } catch {
          this._set_status(name, "shutdown");
          return;
        }
        messages.push({ role: "assistant", content: response.content });
        if (response.stop_reason !== "tool_use") {
          break;
        }
        const results = [];
        let idle_requested = false;
        for (const block of response.content as unknown[]) {
          if ((block as { type?: string }).type !== "tool_use") {
            continue;
          }
          const toolBlock = block as { id: string; name: string; input: Record<string, unknown> };
          let output = "";
          if (toolBlock.name === "idle") {
            idle_requested = true;
            output = "Entering idle phase.";
          } else if (toolBlock.name === "claim_task") {
            output = this.task_mgr.claim(Number(toolBlock.input.task_id), name);
          } else if (toolBlock.name === "send_message") {
            output = this.bus.send(name, String(toolBlock.input.to ?? ""), String(toolBlock.input.content ?? ""));
          } else {
            const dispatch = {
              bash: (args: Record<string, unknown>) => run_bash(String(args.command ?? "")),
              read_file: (args: Record<string, unknown>) => run_read(String(args.path ?? "")),
              write_file: (args: Record<string, unknown>) =>
                run_write(String(args.path ?? ""), String(args.content ?? "")),
              edit_file: (args: Record<string, unknown>) =>
                run_edit(
                  String(args.path ?? ""),
                  String(args.old_text ?? ""),
                  String(args.new_text ?? "")
                ),
            };
            const handler = dispatch[toolBlock.name as keyof typeof dispatch];
            output = handler ? handler(toolBlock.input) : "Unknown";
          }
          console.log(`  [${name}] ${toolBlock.name}: ${String(output).slice(0, 120)}`);
          results.push({ type: "tool_result", tool_use_id: toolBlock.id, content: String(output) });
        }
        messages.push({ role: "user", content: results });
        if (idle_requested) {
          break;
        }
      }

      // -- IDLE PHASE: poll for messages and unclaimed tasks --
      this._set_status(name, "idle");
      let resume = false;
      const polls = Math.floor(IDLE_TIMEOUT / Math.max(POLL_INTERVAL, 1));
      for (let i = 0; i < polls; i += 1) {
        await new Promise((resolvePromise) => setTimeout(resolvePromise, POLL_INTERVAL));
        const inbox = this.bus.read_inbox(name);
        if (inbox.length) {
          for (const msg of inbox) {
            if (msg.type === "shutdown_request") {
              this._set_status(name, "shutdown");
              return;
            }
            messages.push({ role: "user", content: JSON.stringify(msg) });
          }
          resume = true;
          break;
        }
        const unclaimed = scan_unclaimed_tasks(TASKS_DIR);
        if (unclaimed.length) {
          const task = unclaimed[0];
          this.task_mgr.claim(task.id, name);
          if (messages.length <= 3) {
            messages.unshift({
              role: "user",
              content: `<identity>You are '${name}', role: ${role}, team: ${team_name}.</identity>`,
            });
            messages.splice(1, 0, { role: "assistant", content: `I am ${name}. Continuing.` });
          }
          messages.push({
            role: "user",
            content: `<auto-claimed>Task #${task.id}: ${task.subject}\n${task.description ?? ""}</auto-claimed>`,
          });
          messages.push({
            role: "assistant",
            content: `Claimed task #${task.id}. Working on it.`,
          });
          resume = true;
          break;
        }
      }
      if (!resume) {
        this._set_status(name, "shutdown");
        return;
      }
      this._set_status(name, "working");
    }
  }

  list_all(): string {
    if (!this.config.members.length) {
      return "No teammates.";
    }
    const lines = [`Team: ${this.config.team_name}`];
    for (const member of this.config.members) {
      lines.push(`  ${member.name} (${member.role}): ${member.status}`);
    }
    return lines.join("\n");
  }

  member_names(): string[] {
    return this.config.members.map((member) => member.name);
  }
}

// === SECTION: global_instances ===
const TODO = new TodoManager();
const SKILLS = new SkillLoader(SKILLS_DIR);
const TASK_MGR = new TaskManager();
const BG = new BackgroundManager();
const BUS = new MessageBus();
const TEAM = new TeammateManager(BUS, TASK_MGR);

// === SECTION: system_prompt ===
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use tools to solve tasks.
Prefer task_create/task_update/task_list for multi-step work. Use TodoWrite for short checklists.
Use task for subagent delegation. Use load_skill for specialized knowledge.
Skills: ${SKILLS.descriptions()}`;

// === SECTION: shutdown_protocol (s10) ===
function handle_shutdown_request(teammate: string): string {
  const req_id = randomUUID().slice(0, 8);
  shutdown_requests[req_id] = { target: teammate, status: "pending" };
  BUS.send("lead", teammate, "Please shut down.", "shutdown_request", { request_id: req_id });
  return `Shutdown request ${req_id} sent to '${teammate}'`;
}

// === SECTION: plan_approval (s10) ===
function handle_plan_review(request_id: string, approve: boolean, feedback = ""): string {
  const req = plan_requests[request_id];
  if (!req) {
    return `Error: Unknown plan request_id '${request_id}'`;
  }
  req.status = approve ? "approved" : "rejected";
  BUS.send("lead", req.from, feedback, "plan_approval_response", {
    request_id,
    approve,
    feedback,
  });
  return `Plan ${req.status} for '${req.from}'`;
}

// === SECTION: tool_dispatch (s02) ===
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
  TodoWrite: (args: Record<string, unknown>) =>
    TODO.update((args.items as Array<Record<string, unknown>>) ?? []),
  task: (args: Record<string, unknown>) =>
    run_subagent(String(args.prompt ?? ""), String(args.agent_type ?? "Explore")),
  load_skill: (args: Record<string, unknown>) => SKILLS.load(String(args.name ?? "")),
  compress: (_args: Record<string, unknown>) => "Compressing...",
  background_run: (args: Record<string, unknown>) =>
    BG.run(String(args.command ?? ""), Number(args.timeout || 120)),
  check_background: (args: Record<string, unknown>) =>
    BG.check(args.task_id ? String(args.task_id) : undefined),
  task_create: (args: Record<string, unknown>) =>
    TASK_MGR.create(String(args.subject ?? ""), String(args.description ?? "")),
  task_get: (args: Record<string, unknown>) => TASK_MGR.get(Number(args.task_id)),
  task_update: (args: Record<string, unknown>) =>
    TASK_MGR.update(
      Number(args.task_id),
      args.status ? String(args.status) : undefined,
      Array.isArray(args.add_blocked_by) ? (args.add_blocked_by as number[]) : [],
      Array.isArray(args.add_blocks) ? (args.add_blocks as number[]) : [],
      args.owner ? String(args.owner) : undefined
    ),
  task_list: (_args: Record<string, unknown>) => TASK_MGR.list_all(),
  spawn_teammate: (args: Record<string, unknown>) =>
    TEAM.spawn(String(args.name ?? ""), String(args.role ?? ""), String(args.prompt ?? "")),
  list_teammates: (_args: Record<string, unknown>) => TEAM.list_all(),
  send_message: (args: Record<string, unknown>) =>
    BUS.send(
      "lead",
      String(args.to ?? ""),
      String(args.content ?? ""),
      (String(args.msg_type ?? "message") as MessageType)
    ),
  read_inbox: (_args: Record<string, unknown>) =>
    JSON.stringify(BUS.read_inbox("lead"), null, 2),
  broadcast: (args: Record<string, unknown>) =>
    BUS.broadcast("lead", String(args.content ?? ""), TEAM.member_names()),
  shutdown_request: (args: Record<string, unknown>) =>
    handle_shutdown_request(String(args.teammate ?? "")),
  plan_approval: (args: Record<string, unknown>) =>
    handle_plan_review(
      String(args.request_id ?? ""),
      Boolean(args.approve),
      String(args.feedback ?? "")
    ),
  idle: (_args: Record<string, unknown>) => "Lead does not idle.",
  claim_task: (args: Record<string, unknown>) =>
    TASK_MGR.claim(Number(args.task_id), "lead"),
};

const TOOLS = [
  {
    name: "bash",
    description: "Run a shell command.",
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
      properties: { path: { type: "string" }, limit: { type: "integer" } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to file.",
    input_schema: {
      type: "object",
      properties: { path: { type: "string" }, content: { type: "string" } },
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
    name: "TodoWrite",
    description: "Update task tracking list.",
    input_schema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          items: {
            type: "object",
            properties: {
              content: { type: "string" },
              status: { type: "string", enum: ["pending", "in_progress", "completed"] },
              activeForm: { type: "string" },
            },
            required: ["content", "status", "activeForm"],
          },
        },
      },
      required: ["items"],
    },
  },
  {
    name: "task",
    description: "Spawn a subagent for isolated exploration or work.",
    input_schema: {
      type: "object",
      properties: {
        prompt: { type: "string" },
        agent_type: { type: "string", enum: ["Explore", "general-purpose"] },
      },
      required: ["prompt"],
    },
  },
  {
    name: "load_skill",
    description: "Load specialized knowledge by name.",
    input_schema: {
      type: "object",
      properties: { name: { type: "string" } },
      required: ["name"],
    },
  },
  {
    name: "compress",
    description: "Manually compress conversation context.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "background_run",
    description: "Run command in background thread.",
    input_schema: {
      type: "object",
      properties: { command: { type: "string" }, timeout: { type: "integer" } },
      required: ["command"],
    },
  },
  {
    name: "check_background",
    description: "Check background task status.",
    input_schema: {
      type: "object",
      properties: { task_id: { type: "string" } },
    },
  },
  {
    name: "task_create",
    description: "Create a persistent file task.",
    input_schema: {
      type: "object",
      properties: { subject: { type: "string" }, description: { type: "string" } },
      required: ["subject"],
    },
  },
  {
    name: "task_get",
    description: "Get task details by ID.",
    input_schema: {
      type: "object",
      properties: { task_id: { type: "integer" } },
      required: ["task_id"],
    },
  },
  {
    name: "task_update",
    description: "Update task status or dependencies.",
    input_schema: {
      type: "object",
      properties: {
        task_id: { type: "integer" },
        status: { type: "string", enum: ["pending", "in_progress", "completed", "deleted"] },
        add_blocked_by: { type: "array", items: { type: "integer" } },
        add_blocks: { type: "array", items: { type: "integer" } },
      },
      required: ["task_id"],
    },
  },
  {
    name: "task_list",
    description: "List all tasks.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "spawn_teammate",
    description: "Spawn a persistent autonomous teammate.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
        role: { type: "string" },
        prompt: { type: "string" },
      },
      required: ["name", "role", "prompt"],
    },
  },
  {
    name: "list_teammates",
    description: "List all teammates.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "send_message",
    description: "Send a message to a teammate.",
    input_schema: {
      type: "object",
      properties: {
        to: { type: "string" },
        content: { type: "string" },
        msg_type: { type: "string", enum: [...VALID_MSG_TYPES] },
      },
      required: ["to", "content"],
    },
  },
  {
    name: "read_inbox",
    description: "Read and drain the lead's inbox.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "broadcast",
    description: "Send message to all teammates.",
    input_schema: {
      type: "object",
      properties: { content: { type: "string" } },
      required: ["content"],
    },
  },
  {
    name: "shutdown_request",
    description: "Request a teammate to shut down.",
    input_schema: {
      type: "object",
      properties: { teammate: { type: "string" } },
      required: ["teammate"],
    },
  },
  {
    name: "plan_approval",
    description: "Approve or reject a teammate's plan.",
    input_schema: {
      type: "object",
      properties: {
        request_id: { type: "string" },
        approve: { type: "boolean" },
        feedback: { type: "string" },
      },
      required: ["request_id", "approve"],
    },
  },
  {
    name: "idle",
    description: "Enter idle state.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "claim_task",
    description: "Claim a task from the board.",
    input_schema: {
      type: "object",
      properties: { task_id: { type: "integer" } },
      required: ["task_id"],
    },
  },
];

// === SECTION: agent_loop ===
async function agent_loop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  let rounds_without_todo = 0;
  while (true) {
    // s06: compression pipeline
    microcompact(messages);
    if (estimate_tokens(messages) > TOKEN_THRESHOLD) {
      console.log("[auto-compact triggered]");
      messages.splice(0, messages.length, ...(await auto_compact(messages)));
    }

    // s08: drain background notifications
    const notifs = BG.drain();
    if (notifs.length) {
      const txt = notifs
        .map((notif) => `[bg:${notif.task_id}] ${notif.status}: ${notif.result}`)
        .join("\n");
      messages.push({
        role: "user",
        content: `<background-results>\n${txt}\n</background-results>`,
      });
      messages.push({ role: "assistant", content: "Noted background results." });
    }

    // s10: check lead inbox
    const inbox = BUS.read_inbox("lead");
    if (inbox.length) {
      messages.push({
        role: "user",
        content: `<inbox>${JSON.stringify(inbox, null, 2)}</inbox>`,
      });
      messages.push({ role: "assistant", content: "Noted inbox messages." });
    }

    // LLM call
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

    // Tool execution
    const results = [];
    let used_todo = false;
    let manual_compress = false;
    for (const block of response.content as unknown[]) {
      if ((block as { type?: string }).type !== "tool_use") {
        continue;
      }
      const toolBlock = block as {
        id: string;
        name: keyof typeof TOOL_HANDLERS | string;
        input: Record<string, unknown>;
      };
      if (toolBlock.name === "compress") {
        manual_compress = true;
      }
      const handler = TOOL_HANDLERS[toolBlock.name as keyof typeof TOOL_HANDLERS];
      let output = "";
      try {
        output = handler ? String(await handler(toolBlock.input)) : `Unknown tool: ${toolBlock.name}`;
      } catch (error) {
        output = `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
      console.log(`> ${toolBlock.name}: ${String(output).slice(0, 200)}`);
      results.push({
        type: "tool_result",
        tool_use_id: toolBlock.id,
        content: String(output),
      });
      if (toolBlock.name === "TodoWrite") {
        used_todo = true;
      }
    }

    // s03: nag reminder (only when todo workflow is active)
    rounds_without_todo = used_todo ? 0 : rounds_without_todo + 1;
    if (TODO.has_open_items() && rounds_without_todo >= 3) {
      results.unshift({ type: "text", text: "<reminder>Update your todos.</reminder>" });
    }
    messages.push({ role: "user", content: results });

    // s06: manual compress
    if (manual_compress) {
      console.log("[manual compact]");
      messages.splice(0, messages.length, ...(await auto_compact(messages)));
    }
  }
}

// === SECTION: repl ===
const history: Array<{ role: "user" | "assistant"; content: unknown }> = [];
const rl = createInterface({ input, output });

try {
  while (true) {
    const query = await rl.question("\u001b[36ms_full >> \u001b[0m");
    const trimmed = query.trim();
    if (!trimmed || trimmed.toLowerCase() === "q" || trimmed.toLowerCase() === "exit") {
      break;
    }
    if (trimmed === "/compact") {
      if (history.length) {
        console.log("[manual compact via /compact]");
        history.splice(0, history.length, ...(await auto_compact(history)));
      }
      continue;
    }
    if (trimmed === "/tasks") {
      console.log(TASK_MGR.list_all());
      continue;
    }
    if (trimmed === "/team") {
      console.log(TEAM.list_all());
      continue;
    }
    if (trimmed === "/inbox") {
      console.log(JSON.stringify(BUS.read_inbox("lead"), null, 2));
      continue;
    }

    history.push({ role: "user", content: query });
    await agent_loop(history);

    const response_content = history[history.length - 1]?.content;
    if (Array.isArray(response_content)) {
      console.log(response_content.map((block) => (block as { text?: string }).text ?? "").join(""));
    }
    console.log();
  }
} finally {
  rl.close();
}
