// Harness: autonomy -- models that find work without being told.
/*
s11_autonomous_agents.ts- Autonomous Agents

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on s10's protocols.

    Teammate lifecycle:
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown

    Identity re-injection after compression:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

Key insight: "The agent finds work itself."
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

const POLL_INTERVAL = 5_000;
const IDLE_TIMEOUT = 60_000;

const SYSTEM = `You are a team lead at ${WORKDIR}. Teammates are autonomous -- they find work themselves.`;

const VALID_MSG_TYPES = [
  "message",
  "broadcast",
  "shutdown_request",
  "shutdown_response",
  "plan_approval_response",
] as const;

type MessageType = (typeof VALID_MSG_TYPES)[number];

// -- Request trackers --
const shutdown_requests: Record<string, { target: string; status: string }> = {};
const plan_requests: Record<string, { from: string; plan: string; status: string }> = {};

// -- MessageBus: JSONL inbox per teammate --
class MessageBus {
  dir: string;

  constructor(inbox_dir: string) {
    this.dir = inbox_dir;
    mkdirSync(this.dir, { recursive: true });
  }

  send(
    sender: string,
    to: string,
    content: string,
    msg_type: MessageType | string = "message",
    extra: Record<string, unknown> | null = null
  ): string {
    if (!VALID_MSG_TYPES.includes(msg_type as MessageType)) {
      return `Error: Invalid type '${msg_type}'. Valid: ${VALID_MSG_TYPES.join(", ")}`;
    }
    const msg: Record<string, unknown> = {
      type: msg_type,
      from: sender,
      content,
      timestamp: Date.now() / 1000,
    };
    if (extra) {
      Object.assign(msg, extra);
    }
    const inbox_path = join(this.dir, `${to}.jsonl`);
    appendFileSync(inbox_path, `${JSON.stringify(msg)}\n`, "utf8");
    return `Sent ${msg_type} to ${to}`;
  }

  read_inbox(name: string): Array<Record<string, unknown>> {
    const inbox_path = join(this.dir, `${name}.jsonl`);
    if (!existsSync(inbox_path)) {
      return [];
    }
    const messages: Array<Record<string, unknown>> = [];
    const content = readFileSync(inbox_path, "utf8").trim();
    for (const line of content.split(/\r?\n/)) {
      if (line) {
        messages.push(JSON.parse(line) as Record<string, unknown>);
      }
    }
    writeFileSync(inbox_path, "", "utf8");
    return messages;
  }

  broadcast(sender: string, content: string, teammates: string[]): string {
    let count = 0;
    for (const name of teammates) {
      if (name !== sender) {
        this.send(sender, name, content, "broadcast");
        count += 1;
      }
    }
    return `Broadcast to ${count} teammates`;
  }
}

const BUS = new MessageBus(INBOX_DIR);

// -- Task board scanning --
function scan_unclaimed_tasks(): Array<Record<string, unknown>> {
  mkdirSync(TASKS_DIR, { recursive: true });
  const unclaimed: Array<Record<string, unknown>> = [];
  for (const file of readdirSync(TASKS_DIR).filter((entry) => /^task_\d+\.json$/.test(entry)).sort()) {
    const task = JSON.parse(readFileSync(join(TASKS_DIR, file), "utf8")) as Record<string, unknown>;
    if (
      task.status === "pending" &&
      !task.owner &&
      !task.blockedBy
    ) {
      unclaimed.push(task);
    }
  }
  return unclaimed;
}

function claim_task(task_id: number, owner: string): string {
  const path = join(TASKS_DIR, `task_${task_id}.json`);
  if (!existsSync(path)) {
    return `Error: Task ${task_id} not found`;
  }
  const task = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  task.owner = owner;
  task.status = "in_progress";
  writeFileSync(path, `${JSON.stringify(task, null, 2)}\n`, "utf8");
  return `Claimed task #${task_id} for ${owner}`;
}

// -- Identity re-injection after compression --
function make_identity_block(name: string, role: string, team_name: string): {
  role: "user";
  content: string;
} {
  return {
    role: "user",
    content: `<identity>You are '${name}', role: ${role}, team: ${team_name}. Continue your work.</identity>`,
  };
}

// -- Autonomous TeammateManager --
class TeammateManager {
  dir: string;
  config_path: string;
  config: {
    team_name: string;
    members: Array<{ name: string; role: string; status: string }>;
  };
  threads: Record<string, Promise<void>>;

  constructor(team_dir: string) {
    this.dir = team_dir;
    mkdirSync(this.dir, { recursive: true });
    this.config_path = join(this.dir, "config.json");
    this.config = this._load_config();
    this.threads = {};
  }

  _load_config(): {
    team_name: string;
    members: Array<{ name: string; role: string; status: string }>;
  } {
    if (existsSync(this.config_path)) {
      return JSON.parse(readFileSync(this.config_path, "utf8")) as {
        team_name: string;
        members: Array<{ name: string; role: string; status: string }>;
      };
    }
    return { team_name: "default", members: [] };
  }

  _save_config(): void {
    writeFileSync(this.config_path, `${JSON.stringify(this.config, null, 2)}\n`, "utf8");
  }

  _find_member(name: string): { name: string; role: string; status: string } | null {
    for (const member of this.config.members) {
      if (member.name === name) {
        return member;
      }
    }
    return null;
  }

  _set_status(name: string, status: string): void {
    const member = this._find_member(name);
    if (member) {
      member.status = status;
      this._save_config();
    }
  }

  spawn(name: string, role: string, prompt: string): string {
    let member = this._find_member(name);
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
    this._save_config();
    const thread = this._loop(name, role, prompt);
    this.threads[name] = thread;
    void thread;
    return `Spawned '${name}' (role: ${role})`;
  }

  async _loop(name: string, role: string, prompt: string): Promise<void> {
    const team_name = this.config.team_name;
    const sys_prompt = `You are '${name}', role: ${role}, team: ${team_name}, at ${WORKDIR}. Use idle tool when you have no more work. You will auto-claim new tasks.`;
    const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
      { role: "user", content: prompt },
    ];
    const tools = this._teammate_tools();

    while (true) {
      // -- WORK PHASE: standard agent loop --
      for (let i = 0; i < 50; i += 1) {
        const inbox = BUS.read_inbox(name);
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
          this._set_status(name, "idle");
          return;
        }

        messages.push({ role: "assistant", content: response.content });
        if (response.stop_reason !== "tool_use") {
          break;
        }

        const results: Array<Record<string, unknown>> = [];
        let idle_requested = false;
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
            name: string;
            input: Record<string, unknown>;
          };
          let output = "";
          if (tool_use.name === "idle") {
            idle_requested = true;
            output = "Entering idle phase. Will poll for new tasks.";
          } else {
            output = this._exec(name, tool_use.name, tool_use.input);
          }
          console.log(`  [${name}] ${tool_use.name}: ${String(output).slice(0, 120)}`);
          results.push({
            type: "tool_result",
            tool_use_id: tool_use.id,
            content: String(output),
          });
        }
        messages.push({ role: "user", content: results });
        if (idle_requested) {
          break;
        }
      }

      // -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
      this._set_status(name, "idle");
      let resume = false;
      const polls = Math.floor(IDLE_TIMEOUT / Math.max(POLL_INTERVAL, 1));
      for (let i = 0; i < polls; i += 1) {
        await new Promise((resolvePromise) => setTimeout(resolvePromise, POLL_INTERVAL));

        const inbox = BUS.read_inbox(name);
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

        const unclaimed = scan_unclaimed_tasks();
        if (unclaimed.length) {
          const task = unclaimed[0];
          claim_task(Number(task.id), name);
          const task_prompt = `<auto-claimed>Task #${task.id}: ${String(task.subject ?? "")}\n${String(task.description ?? "")}</auto-claimed>`;
          if (messages.length <= 3) {
            messages.unshift(make_identity_block(name, role, team_name));
            messages.splice(1, 0, {
              role: "assistant",
              content: `I am ${name}. Continuing.`,
            });
          }
          messages.push({ role: "user", content: task_prompt });
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

  _exec(sender: string, tool_name: string, args: Record<string, unknown>): string {
    // these base tools are unchanged from s02
    if (tool_name === "bash") {
      return _run_bash(String(args.command ?? ""));
    }
    if (tool_name === "read_file") {
      return _run_read(String(args.path ?? ""));
    }
    if (tool_name === "write_file") {
      return _run_write(String(args.path ?? ""), String(args.content ?? ""));
    }
    if (tool_name === "edit_file") {
      return _run_edit(
        String(args.path ?? ""),
        String(args.old_text ?? ""),
        String(args.new_text ?? "")
      );
    }
    if (tool_name === "send_message") {
      return BUS.send(
        sender,
        String(args.to ?? ""),
        String(args.content ?? ""),
        String(args.msg_type ?? "message")
      );
    }
    if (tool_name === "read_inbox") {
      return JSON.stringify(BUS.read_inbox(sender), null, 2);
    }
    if (tool_name === "shutdown_response") {
      const req_id = String(args.request_id ?? "");
      if (shutdown_requests[req_id]) {
        shutdown_requests[req_id].status = Boolean(args.approve) ? "approved" : "rejected";
      }
      BUS.send(sender, "lead", String(args.reason ?? ""), "shutdown_response", {
        request_id: req_id,
        approve: Boolean(args.approve),
      });
      return `Shutdown ${Boolean(args.approve) ? "approved" : "rejected"}`;
    }
    if (tool_name === "plan_approval") {
      const plan_text = String(args.plan ?? "");
      const req_id = randomUUID().slice(0, 8);
      plan_requests[req_id] = { from: sender, plan: plan_text, status: "pending" };
      BUS.send(sender, "lead", plan_text, "plan_approval_response", {
        request_id: req_id,
        plan: plan_text,
      });
      return `Plan submitted (request_id=${req_id}). Waiting for approval.`;
    }
    if (tool_name === "claim_task") {
      return claim_task(Number(args.task_id), sender);
    }
    return `Unknown tool: ${tool_name}`;
  }

  _teammate_tools(): Array<Record<string, unknown>> {
    // these base tools are unchanged from s02
    return [
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
          properties: { path: { type: "string" } },
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
        name: "send_message",
        description: "Send message to a teammate.",
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
        description: "Read and drain your inbox.",
        input_schema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "shutdown_response",
        description: "Respond to a shutdown request.",
        input_schema: {
          type: "object",
          properties: {
            request_id: { type: "string" },
            approve: { type: "boolean" },
            reason: { type: "string" },
          },
          required: ["request_id", "approve"],
        },
      },
      {
        name: "plan_approval",
        description: "Submit a plan for lead approval.",
        input_schema: {
          type: "object",
          properties: {
            plan: { type: "string" },
          },
          required: ["plan"],
        },
      },
      {
        name: "idle",
        description: "Signal that you have no more work. Enters idle polling phase.",
        input_schema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "claim_task",
        description: "Claim a task from the task board by ID.",
        input_schema: {
          type: "object",
          properties: {
            task_id: { type: "integer" },
          },
          required: ["task_id"],
        },
      },
    ];
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

const TEAM = new TeammateManager(TEAM_DIR);

// -- Base tool implementations (these base tools are unchanged from s02) --
function _safe_path(p: string): string {
  const path = resolve(WORKDIR, p);
  const rel = relative(WORKDIR, path);
  if (rel.startsWith("..")) {
    throw new Error(`Path escapes workspace: ${p}`);
  }
  return path;
}

function _run_bash(command: string): string {
  const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"];
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

function _run_read(path: string, limit?: number): string {
  try {
    let lines = readFileSync(_safe_path(path), "utf8").split(/\r?\n/);
    if (limit && limit < lines.length) {
      lines = lines.slice(0, limit).concat(`... (${lines.length - limit} more)`);
    }
    return lines.join("\n").slice(0, 50_000);
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function _run_write(path: string, content: string): string {
  try {
    const fp = _safe_path(path);
    mkdirSync(dirname(fp), { recursive: true });
    writeFileSync(fp, content, "utf8");
    return `Wrote ${content.length} bytes`;
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function _run_edit(path: string, old_text: string, new_text: string): string {
  try {
    const fp = _safe_path(path);
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

// -- Lead-specific protocol handlers --
function handle_shutdown_request(teammate: string): string {
  const req_id = randomUUID().slice(0, 8);
  shutdown_requests[req_id] = { target: teammate, status: "pending" };
  BUS.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {
    request_id: req_id,
  });
  return `Shutdown request ${req_id} sent to '${teammate}'`;
}

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

function _check_shutdown_status(request_id: string): string {
  return JSON.stringify(shutdown_requests[request_id] ?? { error: "not found" });
}

// -- Lead tool dispatch (14 tools) --
const TOOL_HANDLERS = {
  bash: (args: Record<string, unknown>) => _run_bash(String(args.command ?? "")),
  read_file: (args: Record<string, unknown>) =>
    _run_read(String(args.path ?? ""), Number(args.limit || 0) || undefined),
  write_file: (args: Record<string, unknown>) =>
    _run_write(String(args.path ?? ""), String(args.content ?? "")),
  edit_file: (args: Record<string, unknown>) =>
    _run_edit(
      String(args.path ?? ""),
      String(args.old_text ?? ""),
      String(args.new_text ?? "")
    ),
  spawn_teammate: (args: Record<string, unknown>) =>
    TEAM.spawn(String(args.name ?? ""), String(args.role ?? ""), String(args.prompt ?? "")),
  list_teammates: (_args: Record<string, unknown>) => TEAM.list_all(),
  send_message: (args: Record<string, unknown>) =>
    BUS.send(
      "lead",
      String(args.to ?? ""),
      String(args.content ?? ""),
      String(args.msg_type ?? "message")
    ),
  read_inbox: (_args: Record<string, unknown>) =>
    JSON.stringify(BUS.read_inbox("lead"), null, 2),
  broadcast: (args: Record<string, unknown>) =>
    BUS.broadcast("lead", String(args.content ?? ""), TEAM.member_names()),
  shutdown_request: (args: Record<string, unknown>) =>
    handle_shutdown_request(String(args.teammate ?? "")),
  shutdown_response: (args: Record<string, unknown>) =>
    _check_shutdown_status(String(args.request_id ?? "")),
  plan_approval: (args: Record<string, unknown>) =>
    handle_plan_review(
      String(args.request_id ?? ""),
      Boolean(args.approve),
      String(args.feedback ?? "")
    ),
  idle: (_args: Record<string, unknown>) => "Lead does not idle.",
  claim_task: (args: Record<string, unknown>) => claim_task(Number(args.task_id), "lead"),
};

// these base tools are unchanged from s02
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
    name: "spawn_teammate",
    description: "Spawn an autonomous teammate.",
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
    input_schema: {
      type: "object",
      properties: {},
    },
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
    input_schema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "broadcast",
    description: "Send a message to all teammates.",
    input_schema: {
      type: "object",
      properties: {
        content: { type: "string" },
      },
      required: ["content"],
    },
  },
  {
    name: "shutdown_request",
    description: "Request a teammate to shut down.",
    input_schema: {
      type: "object",
      properties: {
        teammate: { type: "string" },
      },
      required: ["teammate"],
    },
  },
  {
    name: "shutdown_response",
    description: "Check shutdown request status.",
    input_schema: {
      type: "object",
      properties: {
        request_id: { type: "string" },
      },
      required: ["request_id"],
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
    description: "Enter idle state (for lead -- rarely used).",
    input_schema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "claim_task",
    description: "Claim a task from the board by ID.",
    input_schema: {
      type: "object",
      properties: {
        task_id: { type: "integer" },
      },
      required: ["task_id"],
    },
  },
];

async function agent_loop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  while (true) {
    const inbox = BUS.read_inbox("lead");
    if (inbox.length) {
      messages.push({
        role: "user",
        content: `<inbox>${JSON.stringify(inbox, null, 2)}</inbox>`,
      });
      messages.push({
        role: "assistant",
        content: "Noted inbox messages.",
      });
    }

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

const history: Array<{ role: "user" | "assistant"; content: unknown }> = [];
const rl = createInterface({ input, output });

try {
  while (true) {
    const query = await rl.question("\u001b[36ms11 >> \u001b[0m");
    const trimmed = query.trim();
    if (!trimmed || trimmed.toLowerCase() === "q" || trimmed.toLowerCase() === "exit") {
      break;
    }
    if (trimmed === "/team") {
      console.log(TEAM.list_all());
      continue;
    }
    if (trimmed === "/inbox") {
      console.log(JSON.stringify(BUS.read_inbox("lead"), null, 2));
      continue;
    }
    if (trimmed === "/tasks") {
      mkdirSync(TASKS_DIR, { recursive: true });
      for (const file of readdirSync(TASKS_DIR).filter((entry) => /^task_\d+\.json$/.test(entry)).sort()) {
        const task = JSON.parse(readFileSync(join(TASKS_DIR, file), "utf8")) as Record<string, unknown>;
        const marker =
          ({ pending: "[ ]", in_progress: "[>]", completed: "[x]" }[
            String(task.status)
          ] ?? "[?]");
        const owner = task.owner ? ` @${String(task.owner)}` : "";
        console.log(`  ${marker} #${task.id}: ${String(task.subject ?? "")}${owner}`);
      }
      continue;
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
