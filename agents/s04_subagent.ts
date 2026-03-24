// Harness: context isolation -- protecting the model's clarity of thought.
/*
s04_subagent.ts- Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ----------> | while tool_use:  |
    |   prompt="..."   |             |   call tools     |
    |   description="" |             |   append results |
    |                  |  summary    |                  |
    |   result = "..." | <---------  | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
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
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use the task tool to delegate exploration or subtasks.`;
const SUBAGENT_SYSTEM = `You are a coding subagent at ${WORKDIR}. Complete the given task, then summarize your findings.`;
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || undefined,
});

const DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];

// -- Tool implementations shared by parent and child --
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
};

// Child gets all base tools except task (no recursive spawning)
const CHILD_TOOLS = [
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
];

// -- Subagent: fresh context, filtered tools, summary-only return --
async function runSubagent(prompt: string): Promise<string> {
  const subMessages: Array<{ role: "user" | "assistant"; content: unknown }> = [
    { role: "user", content: prompt },
  ];

  let finalContent: unknown[] = [];
  for (let index = 0; index < 30; index += 1) {
    const response = await client.messages.create({
      model: MODEL,
      system: SUBAGENT_SYSTEM,
      messages: subMessages as never,
      tools: CHILD_TOOLS as never,
      max_tokens: 8000,
    });
    finalContent = response.content as unknown[];

    subMessages.push({
      role: "assistant",
      content: response.content,
    });

    if (response.stop_reason !== "tool_use") {
      break;
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
      const handler = TOOL_HANDLERS[toolBlock.name as keyof typeof TOOL_HANDLERS];
      const output = handler ? handler(toolBlock.input) : `Unknown tool: ${toolBlock.name}`;
      results.push({
        type: "tool_result",
        tool_use_id: toolBlock.id,
        content: String(output).slice(0, 50_000),
      });
    }

    subMessages.push({ role: "user", content: results });
  }

  // Only the final text returns to the parent -- child context is discarded
  return (
    finalContent
      .map((block) => {
        if (
          block &&
          typeof block === "object" &&
          "text" in block &&
          typeof (block as { text?: string }).text === "string"
        ) {
          return (block as { text: string }).text;
        }
        return "";
      })
      .join("") || "(no summary)"
  );
}

// -- Parent tools: base tools + task dispatcher --
const PARENT_TOOLS = [
  ...CHILD_TOOLS,
  {
    name: "task",
    description: "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
    input_schema: {
      type: "object",
      properties: {
        prompt: { type: "string" },
        description: { type: "string", description: "Short description of the task" },
      },
      required: ["prompt"],
    },
  },
];

async function agentLoop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  while (true) {
    const response = await client.messages.create({
      model: MODEL,
      system: SYSTEM,
      messages: messages as never,
      tools: PARENT_TOOLS as never,
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
        name: keyof typeof TOOL_HANDLERS | "task" | string;
        input: Record<string, unknown>;
      };

      let output = "";
      if (toolBlock.name === "task") {
        const description = String(toolBlock.input.description ?? "subtask");
        console.log(`> task (${description}): ${String(toolBlock.input.prompt ?? "").slice(0, 80)}`);
        output = await runSubagent(String(toolBlock.input.prompt ?? ""));
      } else {
        const handler = TOOL_HANDLERS[toolBlock.name as keyof typeof TOOL_HANDLERS];
        output = handler ? handler(toolBlock.input) : `Unknown tool: ${toolBlock.name}`;
      }

      console.log(`  ${String(output).slice(0, 200)}`);
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
    const query = await rl.question("\u001b[36ms04 >> \u001b[0m");
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
