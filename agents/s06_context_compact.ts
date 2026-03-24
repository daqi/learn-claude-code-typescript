// Harness: compression -- clean memory for infinite sessions.
/*
s06_context_compact.ts- Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
        |               |
        no              yes
        |               |
        v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
 */

import Anthropic from "@anthropic-ai/sdk";
import { config as loadEnv } from "dotenv";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve, dirname, relative, join } from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

loadEnv({ override: true, quiet: true });

if (process.env.ANTHROPIC_BASE_URL) {
  delete process.env.ANTHROPIC_AUTH_TOKEN;
}

const WORKDIR = resolve(process.cwd());
const MODEL = process.env.MODEL_ID ?? "claude-sonnet-4-20250514";
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use tools to solve tasks.`;
const THRESHOLD = 50_000;
const TRANSCRIPT_DIR = join(WORKDIR, ".transcripts");
const KEEP_RECENT = 3;
const DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || undefined,
});

function estimateTokens(messages: Array<{ role: "user" | "assistant"; content: unknown }>): number {
  return JSON.stringify(messages).length / 4;
}

// -- Layer 1: micro_compact - replace old tool results with placeholders --
function microCompact(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Array<{ role: "user" | "assistant"; content: unknown }> {
  const toolResults: Array<[number, number, Record<string, unknown>]> = [];
  for (let msgIndex = 0; msgIndex < messages.length; msgIndex += 1) {
    const message = messages[msgIndex];
    if (message.role === "user" && Array.isArray(message.content)) {
      for (let partIndex = 0; partIndex < message.content.length; partIndex += 1) {
        const part = message.content[partIndex];
        if (
          part &&
          typeof part === "object" &&
          (part as { type?: string }).type === "tool_result"
        ) {
          toolResults.push([msgIndex, partIndex, part as Record<string, unknown>]);
        }
      }
    }
  }

  if (toolResults.length <= KEEP_RECENT) {
    return messages;
  }

  const toolNameMap: Record<string, string> = {};
  for (const message of messages) {
    if (message.role !== "assistant" || !Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (
        block &&
        typeof block === "object" &&
        (block as { type?: string }).type === "tool_use" &&
        typeof (block as { id?: string }).id === "string" &&
        typeof (block as { name?: string }).name === "string"
      ) {
        toolNameMap[(block as { id: string }).id] = (block as { name: string }).name;
      }
    }
  }

  for (const [, , result] of toolResults.slice(0, Math.max(0, toolResults.length - KEEP_RECENT))) {
    if (typeof result.content === "string" && result.content.length > 100) {
      const toolId = String(result.tool_use_id ?? "");
      const toolName = toolNameMap[toolId] ?? "unknown";
      result.content = `[Previous: used ${toolName}]`;
    }
  }
  return messages;
}

// -- Layer 2: auto_compact - save transcript, summarize, replace messages --
async function autoCompact(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<Array<{ role: "user" | "assistant"; content: unknown }>> {
  mkdirSync(TRANSCRIPT_DIR, { recursive: true });
  const transcriptPath = join(TRANSCRIPT_DIR, `transcript_${Math.floor(Date.now() / 1000)}.jsonl`);
  writeFileSync(
    transcriptPath,
    messages.map((message) => JSON.stringify(message)).join("\n"),
    "utf8"
  );
  console.log(`[transcript saved: ${transcriptPath}]`);

  const conversationText = JSON.stringify(messages).slice(0, 80_000);
  const response = await client.messages.create({
    model: MODEL,
    messages: [
      {
        role: "user",
        content:
          "Summarize this conversation for continuity. Include: 1) What was accomplished, 2) Current state, 3) Key decisions made. Be concise but preserve critical details.\n\n" +
          conversationText,
      },
    ] as never,
    max_tokens: 2000,
  });

  const summary = response.content
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
    .join("");

  return [
    {
      role: "user",
      content: `[Conversation compressed. Transcript: ${transcriptPath}]\n\n${summary}`,
    },
    {
      role: "assistant",
      content: "Understood. I have the context from the summary. Continuing.",
    },
  ];
}

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
  compact: () => "Manual compression requested.",
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
    name: "compact",
    description: "Trigger manual conversation compression.",
    input_schema: {
      type: "object",
      properties: {
        focus: { type: "string", description: "What to preserve in the summary" },
      },
    },
  },
];

async function agentLoop(
  messages: Array<{ role: "user" | "assistant"; content: unknown }>
): Promise<void> {
  while (true) {
    // Layer 1: micro_compact before each LLM call
    microCompact(messages);

    // Layer 2: auto_compact if token estimate exceeds threshold
    if (estimateTokens(messages) > THRESHOLD) {
      console.log("[auto_compact triggered]");
      messages.splice(0, messages.length, ...(await autoCompact(messages)));
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
    let manualCompact = false;
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
      if (toolBlock.name === "compact") {
        manualCompact = true;
        output = "Compressing...";
      } else {
        try {
          const handler = TOOL_HANDLERS[toolBlock.name as keyof typeof TOOL_HANDLERS];
          output = handler ? handler(toolBlock.input) : `Unknown tool: ${toolBlock.name}`;
        } catch (error) {
          output = `Error: ${error instanceof Error ? error.message : String(error)}`;
        }
      }

      console.log(`> ${toolBlock.name}: ${String(output).slice(0, 200)}`);
      results.push({
        type: "tool_result",
        tool_use_id: toolBlock.id,
        content: String(output),
      });
    }

    messages.push({ role: "user", content: results });

    // Layer 3: manual compact triggered by the compact tool
    if (manualCompact) {
      console.log("[manual compact]");
      messages.splice(0, messages.length, ...(await autoCompact(messages)));
    }
  }
}

const history: Array<{ role: "user" | "assistant"; content: unknown }> = [];
const rl = createInterface({ input, output });

try {
  while (true) {
    const query = await rl.question("\u001b[36ms06 >> \u001b[0m");
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
