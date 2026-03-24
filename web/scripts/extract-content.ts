import * as fs from "fs";
import * as path from "path";
import ts from "typescript";
import type {
  AgentVersion,
  VersionDiff,
  DocContent,
  VersionIndex,
} from "../src/types/agent-data";
import { VERSION_META, VERSION_ORDER, LEARNING_PATH } from "../src/lib/constants";

// Resolve paths relative to this script's location (web/scripts/)
const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "..");
const AGENTS_DIR = path.join(REPO_ROOT, "agents");
const DOCS_DIR = path.join(REPO_ROOT, "docs");
const OUT_DIR = path.join(WEB_DIR, "src", "data", "generated");
const BASE_TOOL_NAMES = ["bash", "read_file", "write_file", "edit_file"] as const;
const TOOL_NAME_BY_IDENTIFIER: Record<string, string> = {
  bashTool: "bash",
  readFileTool: "read_file",
  writeFileTool: "write_file",
  editFileTool: "edit_file",
};

// Map TypeScript filenames to version IDs
// s01_agent_loop.ts -> s01
// s02_tool_use.ts -> s02
// s_full.ts -> s_full (reference agent, typically skipped)
function filenameToVersionId(filename: string): string | null {
  const base = path.basename(filename, ".ts");
  if (base === "s_full") return null;

  const match = base.match(/^(s\d+[a-c]?)_/);
  if (!match) return null;
  return match[1];
}

function createSourceFile(source: string, fileName: string): ts.SourceFile {
  return ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS
  );
}

function getNodeLine(sourceFile: ts.SourceFile, pos: number): number {
  return sourceFile.getLineAndCharacterOfPosition(pos).line + 1;
}

function getFirstLineText(source: string, node: ts.Node): string {
  return source
    .slice(node.getStart(), node.getEnd())
    .split(/\r?\n/, 1)[0]
    .trim();
}

function getPropertyName(
  name: ts.PropertyName | ts.BindingName | undefined
): string | null {
  if (!name) return null;
  if (ts.isIdentifier(name) || ts.isStringLiteral(name)) {
    return name.text;
  }
  return null;
}

function getObjectProperty(
  node: ts.ObjectLiteralExpression,
  propertyName: string
): ts.ObjectLiteralElementLike | undefined {
  return node.properties.find((property) => {
    if (
      ts.isPropertyAssignment(property) ||
      ts.isShorthandPropertyAssignment(property)
    ) {
      return getPropertyName(property.name) === propertyName;
    }
    return false;
  });
}

function getPropertyValue(
  property: ts.ObjectLiteralElementLike | undefined
): ts.Expression | undefined {
  if (!property) return undefined;
  if (ts.isPropertyAssignment(property)) {
    return property.initializer;
  }
  if (ts.isShorthandPropertyAssignment(property)) {
    return property.name;
  }
  return undefined;
}

function unwrapExpression(expression: ts.Expression): ts.Expression {
  let current = expression;
  while (
    ts.isParenthesizedExpression(current) ||
    ts.isAsExpression(current) ||
    ts.isSatisfiesExpression(current) ||
    ts.isNonNullExpression(current) ||
    ts.isTypeAssertionExpression(current)
  ) {
    current = current.expression;
  }
  return current;
}

function getToolNameFromObjectLiteral(
  node: ts.ObjectLiteralExpression
): string | null {
  const nameProp = getObjectProperty(node, "name");
  const schemaProp = getObjectProperty(node, "input_schema");
  if (
    nameProp &&
    schemaProp &&
    (() => {
      const nameValue = getPropertyValue(nameProp);
      const schemaValue = getPropertyValue(schemaProp);
      return (
        !!nameValue &&
        !!schemaValue &&
        ts.isStringLiteral(nameValue) &&
        ts.isObjectLiteralExpression(unwrapExpression(schemaValue))
      );
    })()
  ) {
    return (getPropertyValue(nameProp) as ts.StringLiteral).text;
  }
  return null;
}

// Extract classes from TypeScript source
function extractClasses(
  sourceFile: ts.SourceFile
): { name: string; startLine: number; endLine: number }[] {
  const classes: { name: string; startLine: number; endLine: number }[] = [];

  for (const statement of sourceFile.statements) {
    if (ts.isClassDeclaration(statement) && statement.name) {
      classes.push({
        name: statement.name.text,
        startLine: getNodeLine(sourceFile, statement.getStart(sourceFile)),
        endLine: getNodeLine(sourceFile, statement.getEnd()),
      });
    }
  }

  return classes;
}

// Extract top-level functions from TypeScript source
function extractFunctions(
  sourceFile: ts.SourceFile,
  source: string
): { name: string; signature: string; startLine: number }[] {
  const functions: { name: string; signature: string; startLine: number }[] = [];

  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name) {
      functions.push({
        name: statement.name.text,
        signature: getFirstLineText(source, statement),
        startLine: getNodeLine(sourceFile, statement.getStart(sourceFile)),
      });
      continue;
    }

    if (!ts.isVariableStatement(statement)) {
      continue;
    }

    for (const declaration of statement.declarationList.declarations) {
      const name = getPropertyName(declaration.name);
      const initializer = declaration.initializer;
      if (
        !name ||
        !initializer ||
        (!ts.isArrowFunction(initializer) &&
          !ts.isFunctionExpression(initializer))
      ) {
        continue;
      }

      functions.push({
        name,
        signature: getFirstLineText(source, declaration),
        startLine: getNodeLine(sourceFile, declaration.getStart(sourceFile)),
      });
    }
  }

  return functions;
}

// Extract tool names from the `tools` expressions passed into agentLoop.
function extractTools(sourceFile: ts.SourceFile): string[] {
  const toolNames: string[] = [];
  const seen = new Set<string>();
  const bindings = new Map<string, ts.Expression>();

  function addTool(name: string): void {
    if (!seen.has(name)) {
      seen.add(name);
      toolNames.push(name);
    }
  }

  function resolveExpression(expression: ts.Expression | undefined): void {
    if (!expression) return;
    const current = unwrapExpression(expression);

    if (ts.isIdentifier(current)) {
      const directName = TOOL_NAME_BY_IDENTIFIER[current.text];
      if (directName) {
        addTool(directName);
        return;
      }
      const bound = bindings.get(current.text);
      if (bound && bound !== current) {
        resolveExpression(bound);
      }
      return;
    }

    if (ts.isArrayLiteralExpression(current)) {
      for (const element of current.elements) {
        if (ts.isSpreadElement(element)) {
          resolveExpression(element.expression);
        } else {
          resolveExpression(element);
        }
      }
      return;
    }

    if (ts.isCallExpression(current) && ts.isIdentifier(current.expression)) {
      if (current.expression.text === "createBaseTools") {
        for (const name of BASE_TOOL_NAMES) {
          addTool(name);
        }
      }
      return;
    }

    if (ts.isObjectLiteralExpression(current)) {
      const toolName = getToolNameFromObjectLiteral(current);
      if (toolName) {
        addTool(toolName);
      }
    }
  }

  for (const statement of sourceFile.statements) {
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (!ts.isIdentifier(declaration.name) || !declaration.initializer) {
        continue;
      }
      bindings.set(declaration.name.text, declaration.initializer);
    }
  }

  function visit(node: ts.Node): void {
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "agentLoop"
    ) {
      const [options] = node.arguments;
      const current = options && unwrapExpression(options);
      if (current && ts.isObjectLiteralExpression(current)) {
        const toolsProp = getObjectProperty(current, "tools");
        resolveExpression(getPropertyValue(toolsProp));
      }
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  if (!toolNames.length) {
    resolveExpression(
      bindings.get("tools") ??
        bindings.get("TOOLS") ??
        bindings.get("PARENT_TOOLS") ??
        bindings.get("CHILD_TOOLS")
    );
  }
  return toolNames;
}

// Count lines containing real TypeScript tokens, excluding comments/blank lines
function countLoc(sourceFile: ts.SourceFile, source: string): number {
  const occupiedLines = new Set<number>();
  const scanner = ts.createScanner(
    ts.ScriptTarget.Latest,
    true,
    ts.LanguageVariant.Standard,
    source
  );

  let token = scanner.scan();
  while (token !== ts.SyntaxKind.EndOfFileToken) {
    occupiedLines.add(getNodeLine(sourceFile, scanner.getTokenPos()));
    token = scanner.scan();
  }

  return occupiedLines.size;
}

// Detect locale from subdirectory path
// docs/en/s01-the-agent-loop.md -> "en"
// docs/zh/s01-the-agent-loop.md -> "zh"
// docs/ja/s01-the-agent-loop.md -> "ja"
function detectLocale(relPath: string): "en" | "zh" | "ja" {
  if (relPath.startsWith("zh/") || relPath.startsWith("zh\\")) return "zh";
  if (relPath.startsWith("ja/") || relPath.startsWith("ja\\")) return "ja";
  return "en";
}

// Extract version from doc filename (e.g., "s01-the-agent-loop.md" -> "s01")
function extractDocVersion(filename: string): string | null {
  const m = filename.match(/^(s\d+[a-c]?)-/);
  return m ? m[1] : null;
}

// Main extraction
function main() {
  console.log("Extracting content from agents and docs...");
  console.log(`  Repo root: ${REPO_ROOT}`);
  console.log(`  Agents dir: ${AGENTS_DIR}`);
  console.log(`  Docs dir: ${DOCS_DIR}`);

  // Skip extraction if source directories don't exist (e.g. Vercel build).
  // Pre-committed generated data will be used instead.
  if (!fs.existsSync(AGENTS_DIR)) {
    console.log("  Agents directory not found, skipping extraction.");
    console.log("  Using pre-committed generated data.");
    return;
  }

  // 1. Read all agent files
  const agentFiles = fs
    .readdirSync(AGENTS_DIR)
    .filter((f) => f.startsWith("s") && f.endsWith(".ts"));

  console.log(`  Found ${agentFiles.length} agent files`);

  const versions: AgentVersion[] = [];

  for (const filename of agentFiles) {
    const versionId = filenameToVersionId(filename);
    if (!versionId) {
      console.warn(`  Skipping ${filename}: could not determine version ID`);
      continue;
    }

    const filePath = path.join(AGENTS_DIR, filename);
    const source = fs.readFileSync(filePath, "utf-8");
    const sourceFile = createSourceFile(source, filename);

    const meta = VERSION_META[versionId];
    const classes = extractClasses(sourceFile);
    const functions = extractFunctions(sourceFile, source);
    const tools = extractTools(sourceFile);
    const loc = countLoc(sourceFile, source);

    versions.push({
      id: versionId,
      filename,
      title: meta?.title ?? versionId,
      subtitle: meta?.subtitle ?? "",
      loc,
      tools,
      newTools: [], // computed after all versions are loaded
      coreAddition: meta?.coreAddition ?? "",
      keyInsight: meta?.keyInsight ?? "",
      classes,
      functions,
      layer: meta?.layer ?? "tools",
      source,
    });
  }

  // Sort versions according to VERSION_ORDER
  const orderMap = new Map(VERSION_ORDER.map((v, i) => [v, i]));
  versions.sort(
    (a, b) => (orderMap.get(a.id as any) ?? 99) - (orderMap.get(b.id as any) ?? 99)
  );

  // 2. Compute newTools for each version
  for (let i = 0; i < versions.length; i++) {
    const prev = i > 0 ? new Set(versions[i - 1].tools) : new Set<string>();
    versions[i].newTools = versions[i].tools.filter((t) => !prev.has(t));
  }

  // 3. Compute diffs between adjacent versions in LEARNING_PATH
  const diffs: VersionDiff[] = [];
  const versionMap = new Map(versions.map((v) => [v.id, v]));

  for (let i = 1; i < LEARNING_PATH.length; i++) {
    const fromId = LEARNING_PATH[i - 1];
    const toId = LEARNING_PATH[i];
    const fromVer = versionMap.get(fromId);
    const toVer = versionMap.get(toId);

    if (!fromVer || !toVer) continue;

    const fromClassNames = new Set(fromVer.classes.map((c) => c.name));
    const fromFuncNames = new Set(fromVer.functions.map((f) => f.name));
    const fromToolNames = new Set(fromVer.tools);

    diffs.push({
      from: fromId,
      to: toId,
      newClasses: toVer.classes
        .map((c) => c.name)
        .filter((n) => !fromClassNames.has(n)),
      newFunctions: toVer.functions
        .map((f) => f.name)
        .filter((n) => !fromFuncNames.has(n)),
      newTools: toVer.tools.filter((t) => !fromToolNames.has(t)),
      locDelta: toVer.loc - fromVer.loc,
    });
  }

  // 4. Read doc files from locale subdirectories (en/, zh/, ja/)
  const docs: DocContent[] = [];

  if (fs.existsSync(DOCS_DIR)) {
    const localeDirs = ["en", "zh", "ja"];
    let totalDocFiles = 0;

    for (const locale of localeDirs) {
      const localeDir = path.join(DOCS_DIR, locale);
      if (!fs.existsSync(localeDir)) continue;

      const docFiles = fs
        .readdirSync(localeDir)
        .filter((f) => f.endsWith(".md"));

      totalDocFiles += docFiles.length;

      for (const filename of docFiles) {
        const version = extractDocVersion(filename);
        if (!version) {
          console.warn(`  Skipping doc ${locale}/${filename}: could not determine version`);
          continue;
        }

        const filePath = path.join(localeDir, filename);
        const content = fs.readFileSync(filePath, "utf-8");

        const titleMatch = content.match(/^#\s+(.+)$/m);
        const title = titleMatch ? titleMatch[1] : filename;

        docs.push({ version, locale: locale as "en" | "zh" | "ja", title, content });
      }
    }

    console.log(`  Found ${totalDocFiles} doc files across ${localeDirs.length} locales`);
  } else {
    console.warn(`  Docs directory not found: ${DOCS_DIR}`);
  }

  // 5. Write output
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const index: VersionIndex = { versions, diffs };
  const indexPath = path.join(OUT_DIR, "versions.json");
  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
  console.log(`  Wrote ${indexPath}`);

  const docsPath = path.join(OUT_DIR, "docs.json");
  fs.writeFileSync(docsPath, JSON.stringify(docs, null, 2));
  console.log(`  Wrote ${docsPath}`);

  // Summary
  console.log("\nExtraction complete:");
  console.log(`  ${versions.length} versions`);
  console.log(`  ${diffs.length} diffs`);
  console.log(`  ${docs.length} docs`);
  for (const v of versions) {
    console.log(
      `    ${v.id}: ${v.loc} LOC, ${v.tools.length} tools, ${v.classes.length} classes, ${v.functions.length} functions`
    );
  }
}

main();
