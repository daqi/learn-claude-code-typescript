// Harness: planning -- keeping the model on course without scripting the route.
/*
s03_todo_write.ts- TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
 */

import Anthropic from "@anthropic-ai/sdk";
import { config as loadEnv } from "dotenv";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve, dirname, relative } from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

loadEnv({ override: true, quiet: true });

if (process.env.ANTHROPIC_BASE_URL) {
  delete process.env.ANTHROPIC_AUTH_TOKEN;
}

const WORKDIR = resolve(process.cwd());
const MODEL = process.env.MODEL_ID ?? "claude-sonnet-4-20250514";
const SYSTEM = `You are a coding agent at ${WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose.`;
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || undefined,
});

const DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];

// -- TodoManager: structured state the LLM writes to --
class TodoManager {
  items: Array<{ id: string; text: string; status: string }> = [];

  update(items: Array<Record<string, unknown>>): string {
    if (items.length > 20) {
      throw new Error("Max 20 todos allowed");
    }

    const validated = [];
    let inProgressCount = 0;
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      const text = String(item.text ?? "").trim();
      const status = String(item.status ?? "pending").toLowerCase();
      const itemId = String(item.id ?? i + 1);

      if (!text) {
        throw new Error(`Item ${itemId}: text required`);
      }
      if (!["pending", "in_progress", "completed"].includes(status)) {
        throw new Error(`Item ${itemId}: invalid status '${status}'`);
      }
      if (status === "in_progress") {
        inProgressCount += 1;
      }

      validated.push({ id: itemId, text, status });
    }

    if (inProgressCount > 1) {
      throw new Error("Only one task can be in_progress at a time");
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
      const marker =
        { pending: "[ ]", in_progress: "[>]", completed: "[x]" }[
        item.status as "pending" | "in_progress" | "completed"
        ];
      lines.push(`${marker} #${item.id}: ${item.text}`);
    }
    const done = this.items.filter((item) => item.status === "completed").length;
    lines.push(`\n(${done}/${this.items.length} completed)`);
    return lines.join("\n");
  }
}

const TODO = new TodoManager();

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
  todo: (input: Record<string, unknown>) =>
    TODO.update(Array.isArray(input.items) ? (input.items as Array<Record<string, unknown>>) : []),
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
    name: "todo",
    description: "Update task list. Track progress on multi-step tasks.",
    input_schema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          items: {
            type: "object",
            properties: {
              id: { type: "string" },
              text: { type: "string" },
              status: { type: "string", enum: ["pending", "in_progress", "completed"], },
            },
            required: ["id", "text", "status"],
          },
        },
      },
      required: ["items"],
    },
  },
];

// -- Agent loop with nag reminder injection --
async function agentLoop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  let roundsSinceTodo = 0;

  while (true) {
    // Nag reminder is injected below, alongside tool results
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
    let usedTodo = false;
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
      if (toolBlock.name === "todo") {
        usedTodo = true;
      }
    }

    roundsSinceTodo = usedTodo ? 0 : roundsSinceTodo + 1;
    if (roundsSinceTodo >= 3) {
      results.unshift({ type: "text", text: "<reminder>Update your todos.</reminder>" });
    }

    messages.push({ role: "user", content: results });
  }
}

const history: Array<{ role: "user" | "assistant"; content: unknown }> = [];
const rl = createInterface({ input, output });

try {
  while (true) {
    const query = await rl.question("\u001b[36ms03 >> \u001b[0m");
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
