# s03: TodoWrite (待办写入)

`s01 > s02 > [ s03 ] s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"没有计划的 agent 走哪算哪"* -- 先列步骤再动手, 完成率翻倍。
>
> **Harness 层**: 规划 -- 让模型不偏航, 但不替它画航线。

## 问题

多步任务中, 模型会丢失进度 -- 重复做过的事、跳步、跑偏。对话越长越严重: 工具结果不断填满上下文, 系统提示的影响力逐渐被稀释。一个 10 步重构可能做完 1-3 步就开始即兴发挥, 因为 4-10 步已经被挤出注意力了。

## 解决方案

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> | Tools   |
| prompt |      |       |      | + todo  |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                          |
              +-----------+-----------+
              | TodoManager state     |
              | [ ] task A            |
              | [>] task B  <- doing  |
              | [x] task C            |
              +-----------------------+
                          |
              if roundsSinceTodo >= 3:
                inject <reminder> into tool_result
```

## 工作原理

1. TodoManager 存储带状态的项目。同一时间只允许一个 `in_progress`。

```typescript
class TodoManager {
  items = [];

  update(items) {
    const validated = [];
    let inProgressCount = 0;
    for (const item of items) {
      const status = item.status ?? "pending";
      if (status === "in_progress") {
        inProgressCount += 1;
      }
      validated.push({
        id: item.id,
        text: item.text,
        status,
      });
    }
    if (inProgressCount > 1) {
      throw new Error("Only one task can be in_progress");
    }
    this.items = validated;
    return this.render();
  }
}
```

2. `todo` 工具和其他工具一样加入 dispatch map。

```typescript
const TOOL_HANDLERS = {
  // ...base tools...
  todo: (input) => TODO.update(input.items),
};
```

3. nag reminder: 模型连续 3 轮以上不调用 `todo` 时注入提醒。

```typescript
if (roundsSinceTodo >= 3 && messages.length) {
  const last = messages[messages.length - 1];
  if (last.role === "user" && Array.isArray(last.content)) {
    last.content.unshift({
      type: "text",
      text: "<reminder>Update your todos.</reminder>",
    });
  }
}
```

"同时只能有一个 in_progress" 强制顺序聚焦。nag reminder 制造问责压力 -- 你不更新计划, 系统就追着你问。

## 相对 s02 的变更

| 组件           | 之前 (s02)       | 之后 (s03)                     |
|----------------|------------------|--------------------------------|
| Tools          | 4                | 5 (+todo)                      |
| 规划           | 无               | 带状态的 TodoManager           |
| Nag 注入       | 无               | 3 轮后注入 `<reminder>`        |
| Agent loop     | 简单分发         | + roundsSinceTodo 计数器     |

## 试一试

```sh
cd learn-claude-code-typescript
npx tsx agents/s03_todo_write.ts
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Refactor the file hello.ts: add type hints, docstrings, and a main guard`
2. `Create a TypeScript package with index.ts, utils.ts, and tests/utils.test.ts`
3. `Review all TypeScript files and fix any style issues`
