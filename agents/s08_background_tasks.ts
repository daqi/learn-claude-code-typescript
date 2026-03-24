// Harness: background execution -- the model thinks while the harness waits.
/*
s08_background_tasks.ts- Background Tasks

Run commands in background threads. A notification queue is drained
before each LLM call to deliver results.

    Main thread                Background thread
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call]  <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+

    Timeline:
    Agent ----[spawn A]----[spawn B]----[other work]----
                  |              |
                  v              v
              [A runs]      [B runs]        (parallel)
                  |              |
                  +-- notification queue --> [results injected]

Key insight: "Fire and forget -- the agent doesn't block while the command runs."
 */

import Anthropic from "@anthropic-ai/sdk";
import { config as loadEnv } from "dotenv";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { resolve, dirname, relative } from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

loadEnv({ override: true, quiet: true });

if (process.env.ANTHROPIC_BASE_URL) {
  delete process.env.ANTHROPIC_AUTH_TOKEN;
}

const WORKDIR = resolve(process.cwd());
const MODEL = process.env.MODEL_ID ?? "claude-sonnet-4-20250514";
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use background_run for long-running commands.`;
const DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || undefined,
});

// -- BackgroundManager: threaded execution + notification queue --
class BackgroundManager {
  tasks: Record<string, { status: string; result: string | null; command: string }>;
  notificationQueue: Array<{ task_id: string; status: string; command: string; result: string }>;

  constructor() {
    this.tasks = {};
    this.notificationQueue = [];
  }

  run(command: string): string {
    const taskId = randomUUID().slice(0, 8);
    this.tasks[taskId] = { status: "running", result: null, command };
    setTimeout(() => this.execute(taskId, command), 0);
    return `Background task ${taskId} started: ${command.slice(0, 80)}`;
  }

  execute(taskId: string, command: string): void {
    let output = "";
    let status = "completed";
    try {
      const result = spawnSync(command, {
        shell: true,
        cwd: WORKDIR,
        encoding: "utf8",
        timeout: 300_000,
      });
      output = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim().slice(0, 50_000);
      if (result.error && (result.error as NodeJS.ErrnoException).code === "ETIMEDOUT") {
        output = "Error: Timeout (300s)";
        status = "timeout";
      } else if (result.error) {
        output = `Error: ${result.error instanceof Error ? result.error.message : String(result.error)}`;
        status = "error";
      }
    } catch (error) {
      output = `Error: ${error instanceof Error ? error.message : String(error)}`;
      status = "error";
    }

    this.tasks[taskId].status = status;
    this.tasks[taskId].result = output || "(no output)";
    this.notificationQueue.push({
      task_id: taskId,
      status,
      command: command.slice(0, 80),
      result: (output || "(no output)").slice(0, 500),
    });
  }

  check(taskId?: string): string {
    if (taskId) {
      const task = this.tasks[taskId];
      if (!task) {
        return `Error: Unknown task ${taskId}`;
      }
      return `[${task.status}] ${task.command.slice(0, 60)}\n${task.result ?? "(running)"}`;
    }
    const lines = Object.entries(this.tasks).map(([id, task]) => `${id}: [${task.status}] ${task.command.slice(0, 60)}`);
    return lines.length ? lines.join("\n") : "No background tasks.";
  }

  drainNotifications(): Array<{ task_id: string; status: string; command: string; result: string }> {
    const notifications = [...this.notificationQueue];
    this.notificationQueue = [];
    return notifications;
  }
}

const BG = new BackgroundManager();

// -- Tool implementations --
function safePath(filePath: string): string {
  const absolute = resolve(WORKDIR, filePath);
  const rel = relative(WORKDIR, absolute);
  if (rel.startsWith("..")) {
    throw new Error(`Path escapes workspace: ${filePath}`);
  }
  return absolute;
}

function runBash(command: string): string {
  if (DANGEROUS_COMMANDS.some((token) => command.includes(token))) {
    return "Error: Dangerous command blocked";
  }
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
  const text = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
  return text.slice(0, 50_000) || "(no output)";
}

function runRead(filePath: string, limit?: number): string {
  try {
    const lines = readFileSync(safePath(filePath), "utf8").split(/\r?\n/);
    if (limit && limit < lines.length) {
      const remaining = lines.length - limit;
      return [...lines.slice(0, limit), `... (${remaining} more)`].join("\n").slice(0, 50_000);
    }
    return lines.join("\n").slice(0, 50_000);
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function runWrite(filePath: string, content: string): string {
  try {
    const absolute = safePath(filePath);
    mkdirSync(dirname(absolute), { recursive: true });
    writeFileSync(absolute, content, "utf8");
    return `Wrote ${content.length} bytes`;
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function runEdit(filePath: string, oldText: string, newText: string): string {
  try {
    const absolute = safePath(filePath);
    const content = readFileSync(absolute, "utf8");
    if (!content.includes(oldText)) {
      return `Error: Text not found in ${filePath}`;
    }
    writeFileSync(absolute, content.replace(oldText, newText), "utf8");
    return `Edited ${filePath}`;
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

const TOOL_HANDLERS = {
  bash: (input: Record<string, unknown>) => runBash(String(input.command ?? "")),
  read_file: (input: Record<string, unknown>) =>
    runRead(String(input.path ?? ""), Number(input.limit || 0) || undefined),
  write_file: (input: Record<string, unknown>) =>
    runWrite(String(input.path ?? ""), String(input.content ?? "")),
  edit_file: (input: Record<string, unknown>) =>
    runEdit(
      String(input.path ?? ""),
      String(input.old_text ?? ""),
      String(input.new_text ?? "")
    ),
  background_run: (input: Record<string, unknown>) => BG.run(String(input.command ?? "")),
  check_background: (input: Record<string, unknown>) =>
    BG.check(input.task_id ? String(input.task_id) : undefined),
};

const TOOLS = [
  {
    name: "bash",
    description: "Run a shell command (blocking).",
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
    name: "background_run",
    description: "Run command in background thread. Returns task_id immediately.",
    input_schema: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
  {
    name: "check_background",
    description: "Check background task status. Omit task_id to list all.",
    input_schema: {
      type: "object",
      properties: { task_id: { type: "string" } },
    },
  },
];

async function agentLoop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  while (true) {
    // Drain background notifications and inject as system message before LLM call
    const notifications = BG.drainNotifications();
    if (notifications.length && messages.length) {
      const notificationText = notifications
        .map((item) => `[bg:${item.task_id}] ${item.status}: ${item.result}`)
        .join("\n");
      messages.push({
        role: "user",
        content: `<background-results>\n${notificationText}\n</background-results>`,
      });
      messages.push({ role: "assistant", content: "Noted background results." });
    }

    const response = await client.messages.create({
      model: MODEL,
      system: SYSTEM,
      messages: messages as never,
      tools: TOOLS as never,
      max_tokens: 8000,
    });

    messages.push({
      role: "assistant",
      content: response.content,
    });

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

      const toolBlock = block as {
        id: string;
        name: keyof typeof TOOL_HANDLERS | string;
        input: Record<string, unknown>;
      };

      let output = "";
      try {
        const handler = TOOL_HANDLERS[toolBlock.name as keyof typeof TOOL_HANDLERS];
        output = handler ? handler(toolBlock.input) : `Unknown tool: ${toolBlock.name}`;
      } catch (error) {
        output = `Error: ${error instanceof Error ? error.message : String(error)}`;
      }

      console.log(`> ${toolBlock.name}: ${String(output).slice(0, 200)}`);
      results.push({
        type: "tool_result",
        tool_use_id: toolBlock.id,
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
    const query = await rl.question("\u001b[36ms08 >> \u001b[0m");
    const trimmed = query.trim();
    if (!trimmed || trimmed.toLowerCase() === "q" || trimmed.toLowerCase() === "exit") {
      break;
    }

    history.push({ role: "user", content: query });
    await agentLoop(history);

    const responseContent = history[history.length - 1]?.content;
    if (Array.isArray(responseContent)) {
      console.log(responseContent.map((block) => (block as { text?: string }).text ?? "").join(""));
    }
    console.log();
  }
} finally {
  rl.close();
}
