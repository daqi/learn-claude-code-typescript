// Harness: the loop -- the model's first connection to the real world.
/*
s01_agent_loop.ts- The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
 */

import Anthropic from "@anthropic-ai/sdk";
import { config as loadEnv } from "dotenv";
import { spawnSync } from "node:child_process";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

loadEnv({ override: true, quiet: true });

if (process.env.ANTHROPIC_BASE_URL) {
  delete process.env.ANTHROPIC_AUTH_TOKEN;
}

const MODEL = process.env.MODEL_ID ?? "claude-sonnet-4-20250514";
const SYSTEM = `You are a coding agent at ${process.cwd()}. Use bash to solve tasks. Act, don't explain.`;
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || undefined,
});

const DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];

function runBash(command: string): string {
  if (DANGEROUS_COMMANDS.some((token) => command.includes(token))) {
    return "Error: Dangerous command blocked";
  }
  const result = spawnSync(command, {
    shell: true,
    cwd: process.cwd(),
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

const TOOLS = [
  {
    name: "bash",
    description: "Run a shell command.",
    input_schema: {
      type: "object",
      properties: { command: { type: "string" }, },
      required: ["command"],
    },
  },
];

// -- The core pattern: a while loop that calls tools until the model stops --
async function agentLoop(
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

    // Append assistant turn
    messages.push({
      role: "assistant",
      content: response.content,
    });

    // If the model didn't call a tool, we're done
    if (response.stop_reason !== "tool_use") {
      return;
    }

    // Execute each tool call, collect results
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
        name: string;
        input: Record<string, unknown>;
      };
      let output = "";
      try {
        output =
          toolBlock.name === "bash"
            ? runBash(String(toolBlock.input.command ?? ""))
            : `Unknown tool: ${toolBlock.name}`;
      } catch (error) {
        output = `Error: ${error instanceof Error ? error.message : String(error)}`;
      }

      console.log(`\u001b[33m$ ${String(toolBlock.input.command ?? "")}\u001b[0m`);
      console.log(output.slice(0, 200));
      results.push({
        type: "tool_result",
        tool_use_id: toolBlock.id,
        content: output,
      });
    }

    messages.push({ role: "user", content: results });
  }
}

const history: Array<{ role: "user" | "assistant"; content: unknown }> = [];
const rl = createInterface({ input, output });

try {
  while (true) {
    const query = await rl.question("\u001b[36ms01 >> \u001b[0m");
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
