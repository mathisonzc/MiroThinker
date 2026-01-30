# MiroThinker Agent 调用与 Tool 调用流程

## 一、总体流程概览

```
main.py
  -> create_pipeline_components()    # 初始化 ToolManager、OutputFormatter
  -> execute_task_pipeline()          # 创建 LLM Client + Orchestrator
    -> orchestrator.run_main_agent()  # 主 Agent 循环
      -> [循环] handle_llm_call() -> 解析工具调用 -> 执行工具 -> 更新历史
      -> generate_and_finalize_answer()  # 最终答案生成
```

---

## 二、Agent 调用流程（详细）

### 阶段 1：初始化

`pipeline.py:178-215` 的 `create_pipeline_components()` 负责：

```
1. 读取 agent 配置（如 mirothinker_v1.5_keep5_max200.yaml）
2. 根据 tools 列表创建 MCP Server 参数
3. 构建 ToolManager（主 Agent + 子 Agent 各一个）
4. 返回 (main_agent_tool_manager, sub_agent_tool_managers, output_formatter)
```

然后 `execute_task_pipeline()` 创建关键对象：

```python
# pipeline.py:96-111
llm_client = ClientFactory(task_id=unique_id, cfg=cfg, task_log=task_log)
orchestrator = Orchestrator(
    main_agent_tool_manager,
    sub_agent_tool_managers,
    llm_client, output_formatter, cfg, task_log, stream_queue
)
```

### 阶段 2：主 Agent 循环

`orchestrator.py:736-1197` 的 `run_main_agent()` 是核心循环：

```
+------------------------------------------------------------------+
|  初始化                                                            |
|  |- 获取 tool_definitions（从 ToolManager 异步获取）                 |
|  |- 生成 system_prompt = MCP系统指令 + Agent角色指令                 |
|  |- message_history = [{"role":"user", "content": task}]           |
|  +- 设置 max_turns, turn_count=0, consecutive_rollbacks=0          |
+------------------------------------------------------------------+
                            |
                            v
+--------------- while turn < max_turns ---------------------------+
|                                                                   |
|  (1) LLM 调用                                                     |
|  |- answer_generator.handle_llm_call(system_prompt,               |
|  |     message_history, tool_definitions)                         |
|  |- 内部调用 llm_client.create_message()                          |
|  |   +- _remove_tool_result_from_messages(keep=5)                 |
|  |       保留所有 assistant 消息，只保留最近5个 tool result           |
|  |       旧的 tool result -> "Tool result is omitted"              |
|  |- 返回 (response_text, should_break, tool_calls, history)       |
|  +- 提取中间 \boxed{} 答案 -> intermediate_boxed_answers           |
|                                                                   |
|  (2) 判断是否有工具调用                                              |
|  |- 如果 tool_calls 为空:                                          |
|  |   |- 检查是否有 MCP 格式错误（XML 标签残留）-> 回滚重试            |
|  |   |- 检查是否有拒绝关键词（"I'm sorry"）   -> 回滚重试            |
|  |   +- 都没有 -> break 退出循环（模型主动停止）                      |
|  |                                                                |
|  |- 如果有 tool_calls:                                             |
|  |   +- 进入工具执行流程（见第三节）                                  |
|  |                                                                |
|  (3) 更新消息历史                                                   |
|  |- llm_client.update_message_history(history, tool_results)      |
|  |   +- 将工具结果合并为一条 user 消息追加到 history                  |
|  |                                                                |
|  (4) 上下文长度检查                                                 |
|  |- ensure_summary_context(history, summary_prompt)               |
|  |   +- 估算 tokens: 上次prompt + 上次completion + 最后user        |
|  |       + summary_prompt + max_tokens + 1000 buffer              |
|  |   +- 如果超过 max_context_length -> 删除最后一对消息 -> break     |
|  +- 保存日志                                                      |
|                                                                   |
+-------------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------------+
|  最终答案生成                                                       |
|  generate_and_finalize_answer()                                    |
|  |- 如果 context_management ON 且超时 -> 跳过答案,直接生成失败摘要     |
|  |- 否则 -> 注入 summary_prompt -> LLM 生成最终答案                  |
|  |- 提取 \boxed{} 内容作为 final_boxed_answer                      |
|  +- 如果失败 -> 生成 failure_experience_summary 供重试               |
+-------------------------------------------------------------------+
```

---

## 三、Tool 调用流程（详细）

当 LLM 返回包含工具调用的响应时，进入 `orchestrator.py:900-1098` 的工具执行流程。

### 3.1 工具调用解析

LLM 的响应文本中包含 XML 格式的工具调用：

```xml
<think>
我需要搜索相关信息...
</think>

<use_mcp_tool>
<server_name>search_and_scrape_webpage</server_name>
<tool_name>google_search</tool_name>
<arguments>
{"q": "GAIA benchmark latest results", "num": 10}
</arguments>
</use_mcp_tool>
```

由 `parsing_utils.py` 的 `parse_llm_response_for_tool_calls()` 解析为：

```python
[{
    "server_name": "search_and_scrape_webpage",
    "tool_name": "google_search",
    "arguments": {"q": "GAIA benchmark latest results", "num": 10},
    "id": "uuid-xxx"
}]
```

### 3.2 工具执行判断分支

```
遍历每个 tool_call:
|
|- (1) 参数自动修正
|   tool_executor.fix_tool_call_arguments(tool_name, arguments)
|   例: "description" -> "info_to_extract" (LLM 常犯错误)
|
|- (2) 判断是否为子 Agent 调用
|   server_name.startswith("agent-") ?
|
|--- YES: 子 Agent 调用 ----------------------------------------+
|   |                                                            |
|   |- 重复查询检测（cache_name = "main_" + tool_name）           |
|   |- orchestrator.run_sub_agent(server_name, subtask)          |
|   |   |- 创建独立 message_history                              |
|   |   |- 生成子 Agent 专属 system_prompt                       |
|   |   |- 进入子 Agent 自己的 while 循环                         |
|   |   |   +- (与主 Agent 循环结构相同)                          |
|   |   |- 子 Agent summary 生成                                 |
|   |   +- 返回 final_answer_text                                |
|   +- 将子 Agent 结果作为 tool_result 返回给主 Agent              |
|                                                                |
|--- NO: 普通工具调用 -------------------------------------------+
|   |                                                            |
|   |- 重复查询检测                                               |
|   |   get_query_str_from_tool_call() -> 生成查询签名             |
|   |   +- google_search -> "google_search_{q}"                  |
|   |   +- scrape_and_extract_info -> "scrape_{url}_{info}"      |
|   |   检查 used_queries[cache_name][query_str] > 0?            |
|   |   |- 是重复 + 未达回滚上限 -> 回滚(pop history, turn--)      |
|   |   +- 是重复 + 已达上限 -> 允许执行（避免死锁）                 |
|   |                                                            |
|   |- 执行工具                                                   |
|   |   main_agent_tool_manager.execute_tool_call(               |
|   |       server_name, tool_name, arguments)                   |
|   |   +- 通过 MCP Stdio 协议调用对应的 MCP Server               |
|   |                                                            |
|   |- 后处理                                                     |
|   |   post_process_tool_call_result()                          |
|   |   +- Demo 模式下截断 scrape 结果至 20K 字符                  |
|   |                                                            |
|   |- 错误检测 -> 是否需要回滚                                    |
|   |   should_rollback_result() 检查:                           |
|   |   |- "Unknown tool:" 错误                                  |
|   |   |- "Error executing tool" 错误                           |
|   |   +- Google 搜索返回空结果                                  |
|   |   如果需要回滚 -> pop history, turn--, continue              |
|   |                                                            |
|   +- 记录查询到 used_queries 供后续去重                          |
|                                                                |
+- 格式化工具结果                                                  |
   output_formatter.format_tool_result_for_user(result)           |
   -> (call_id, formatted_result)                                 |
```

### 3.3 工具结果写回消息历史

```python
# openai_client.py:338-358
def update_message_history(self, message_history, all_tool_results_content_with_id):
    # 将所有工具结果合并为一条 user 消息
    merged_text = "\n".join([item[1]["text"] for item in results])
    message_history.append({"role": "user", "content": merged_text})
```

**重要设计选择**：MiroThinker 使用的是 **OpenAI text completion 格式**（不是 function calling），工具结果作为普通 `user` 消息回传给模型。这是因为基座模型是 Qwen3，通过 XML 标签做工具调用比 function calling 更灵活。

消息历史的结构如下：

```
message_history = [
    {"role": "user",      "content": "原始任务描述"},              # 始终保留
    {"role": "assistant", "content": "<think>...</think>\n<use_mcp_tool>..."},  # 保留
    {"role": "user",      "content": "Tool result is omitted..."},  # 旧结果被替换
    {"role": "assistant", "content": "<think>...</think>\n<use_mcp_tool>..."},  # 保留
    {"role": "user",      "content": "Tool result is omitted..."},  # 旧结果被替换
    ...
    {"role": "assistant", "content": "<think>...</think>\n<use_mcp_tool>..."},  # 保留
    {"role": "user",      "content": "{"result": "实际的搜索结果..."}"},  # 最近5个保留
    ...
]
```

---

## 四、MCP 工具执行底层

工具执行通过 `ToolManager.execute_tool_call()` 完成，底层走 MCP Stdio 协议：

```
Orchestrator
  -> ToolManager.execute_tool_call(server_name, tool_name, arguments)
    -> 找到对应的 MCP Server 进程（Stdio 通信）
    -> 发送 JSON-RPC 请求给 MCP Server
    -> MCP Server 执行实际操作:
        |- tool-python: E2B 沙箱运行 Python 代码
        |- search_and_scrape_webpage: Serper API 调用 Google 搜索
        +- jina_scrape_llm_summary: Jina 抓取网页 + LLM 摘要
    -> 返回 JSON 结果
```

### MCP Server 配置方式

定义在 `src/config/settings.py` 中：

```python
def create_mcp_server_parameters(cfg, agent_cfg):
    configs = []
    if "tool-python" in agent_cfg["tools"]:
        configs.append({
            "name": "tool-python",
            "params": StdioServerParameters(
                command=sys.executable,
                args=["-m", "miroflow_tools.mcp_servers.python_mcp_server"],
                env={"E2B_API_KEY": E2B_API_KEY}
            )
        })
    # ... 其他工具类似
    return configs, blacklist
```

每个工具作为独立的 Python 进程运行，通过 stdin/stdout 与主进程通信。

### 可用的 MCP Server 列表

| Server 名称 | 功能 | 依赖 |
|-------------|------|------|
| `tool-python` | E2B 沙箱代码执行 | E2B_API_KEY |
| `search_and_scrape_webpage` | Google 搜索 | SERPER_API_KEY |
| `jina_scrape_llm_summary` | 网页抓取 + LLM 摘要 | JINA_API_KEY |
| `tool-vqa` / `tool-vqa-os` | 视觉问答 | 视觉模型 |
| `tool-transcribe` | 音频转录 | 转录模型 |
| `tool-reading` | 文档转换 (PDF/DOC/PPT) | 本地解析库 |
| `tool-reasoning` | 复杂推理 | 推理模型 |
| `tool-google-search` | Google 搜索 + 网页抓取 | Google API |
| `tool-sogou-search` | 搜狗搜索（中文） | Sogou API |

---

## 五、回滚机制的完整流程图

```
LLM 返回响应
    |
    |- 无工具调用 + 检测到 MCP 标签残留 --+
    |- 无工具调用 + 检测到拒绝关键词 -----+
    |- 重复查询检测命中 -----------------+
    |- 工具返回 "Unknown tool" 错误 -----+
    |- Google 搜索返回空结果 ------------+
    |                                    |
    |                                    v
    |                       consecutive_rollbacks < 5 ?
    |                       |- YES: message_history.pop()
    |                       |       turn_count -= 1
    |                       |       consecutive_rollbacks += 1
    |                       |       continue (重新发送给 LLM)
    |                       |
    |                       +- NO: 允许通过 / 强制 break
    |                              (避免无限循环)
    |
    |- 工具执行成功
    |   +- consecutive_rollbacks 重置为 0
    |
    +- (继续下一轮循环)
```

### 回滚的具体实现

```python
# orchestrator.py:206-252  _handle_response_format_issues()
# 检测 MCP 格式错误
if any(mcp_tag in assistant_response_text for mcp_tag in mcp_tags):
    if consecutive_rollbacks < MAX_CONSECUTIVE_ROLLBACKS - 1:
        turn_count -= 1
        consecutive_rollbacks += 1
        if message_history[-1]["role"] == "assistant":
            message_history.pop()  # 删除错误的 assistant 消息
        return True, False, ...    # should_continue=True

# 检测拒绝关键词
if any(keyword in assistant_response_text for keyword in refusal_keywords):
    # 同样的回滚逻辑...
```

```python
# orchestrator.py:257-316  _check_duplicate_query()
query_str = get_query_str_from_tool_call(tool_name, arguments)
count = used_queries[cache_name][query_str]
if count > 0:
    if consecutive_rollbacks < MAX_CONSECUTIVE_ROLLBACKS - 1:
        message_history.pop()      # 删除包含重复调用的 assistant 消息
        turn_count -= 1
        consecutive_rollbacks += 1
        return True, True, ...     # is_duplicate=True, should_rollback=True
```

---

## 六、上下文长度管理流程

```
每轮工具执行后:
    |
    v
ensure_summary_context(message_history, summary_prompt)
    |
    |- 计算 estimated_total =
    |     last_prompt_tokens           # 上次 LLM 调用的输入 token 数
    |   + last_completion_tokens       # 上次 LLM 调用的输出 token 数
    |   + last_user_tokens * 1.5       # 最后一条 user 消息的 token（含 buffer）
    |   + summary_tokens * 1.5         # summary prompt 的 token（含 buffer）
    |   + max_tokens                   # 预留的最大输出空间
    |   + 1000                         # 额外 buffer
    |
    |- estimated_total >= max_context_length ?
    |   |
    |   |- YES:
    |   |   |- 删除最后的 user 消息（工具结果）
    |   |   |- 删除最后的 assistant 消息（工具调用）
    |   |   |- 设置 turn_count = max_turns -> 触发 break
    |   |   +- 进入最终答案生成
    |   |
    |   +- NO: 继续下一轮循环
```

---

## 七、最终答案生成流程

```
generate_and_finalize_answer()
    |
    |- context_management ON 且 reached_max_turns?
    |   |
    |   |- YES: 跳过答案生成
    |   |   |- 直接调用 generate_failure_summary()
    |   |   |   |- 构建 failure_summary_history
    |   |   |   |- 追加 FAILURE_SUMMARY_PROMPT
    |   |   |   |- 追加 FAILURE_SUMMARY_ASSISTANT_PREFIX (引导结构化输出)
    |   |   |   |- LLM 生成失败摘要
    |   |   |   +- 返回 failure_experience_summary
    |   |   +- 返回 (FORMAT_ERROR, failure_summary)
    |   |
    |   +- NO: 正常生成答案
    |       |
    |       v
    |   generate_final_answer_with_retries()
    |       |- 注入 summary_prompt 到 message_history
    |       |- for retry_idx in range(max_retries):   # 默认最多3次
    |       |   |- handle_llm_call() 生成最终答案
    |       |   |- _extract_boxed_content() 提取 \boxed{}
    |       |   |- 找到 -> break
    |       |   +- 未找到 -> pop assistant, retry
    |       |
    |       +- 返回 (final_answer_text, final_summary, final_boxed_answer)
    |
    |- context_management OFF?
    |   |- 如果无有效答案, 尝试用 intermediate_boxed_answers[-1] 作为 fallback
    |   +- 返回
    |
    |- context_management ON + 正常完成?
    |   |- 如果无有效答案, 不使用 fallback（避免猜测降低准确率）
    |   |- 生成 failure_experience_summary
    |   +- 返回
```

---

## 八、子 Agent 调用流程

当主 Agent 的工具调用中 `server_name.startswith("agent-")` 时，触发子 Agent：

```
主 Agent 循环中:
    |
    |- server_name = "agent-browsing"
    |- arguments = {"subtask": "搜索某个具体问题的答案"}
    |
    v
run_sub_agent("agent-browsing", subtask)
    |
    |- task_description += "\n\nPlease provide the answer and detailed
    |   supporting information of the subtask given to you."
    |
    |- 创建独立的 message_history = [{"role":"user", "content": subtask}]
    |
    |- 获取子 Agent 专属的 tool_definitions
    |   +- 子 Agent 可以有不同的工具集
    |
    |- 生成子 Agent 专属 system_prompt
    |   = MCP系统指令 + generate_agent_specific_system_prompt("agent-browsing")
    |
    |- 设置子 Agent 的 max_turns（从配置读取）
    |
    |- while turn < sub_agent_max_turns:
    |   |- handle_llm_call() (复用同一个 LLM client)
    |   |- 工具执行（使用子 Agent 自己的 ToolManager）
    |   |- 重复检测（使用子 Agent 自己的 cache_name = sub_agent_id + "_" + tool_name）
    |   |- 上下文长度检查
    |   +- 保存日志到独立的 sub_agent_session
    |
    |- 子 Agent 循环结束
    |- 注入子 Agent 专属的 summary_prompt
    |- LLM 生成子 Agent 最终答案
    |- 去除 <think>...</think> 内容
    |
    +- 返回 final_answer_text 给主 Agent（作为工具结果）
```

子 Agent 的设计特点：
- 完全独立的消息历史（不共享主 Agent 的上下文）
- 独立的查询去重缓存
- 独立的 ToolManager 和工具集
- 共享同一个 LLM Client（节省连接开销）
- 结果通过 `tool_result` 回传给主 Agent

---

## 九、LLM 调用的重试与容错

`openai_client.py:80-277` 的 `_create_message()` 包含多层重试逻辑：

```
最大重试次数: 10
基础等待时间: 30 秒

for attempt in range(10):
    |
    |- 构造请求参数
    |   |- model, temperature, messages, top_p, max_tokens
    |   |- GPT-5 使用 max_completion_tokens 而非 max_tokens
    |   |- DeepSeek-V3 添加 thinking.type="enabled"
    |   |- 如果最后一条是 assistant 消息 -> continue_final_message=True
    |
    |- 发送请求
    |
    |- 成功后检查:
    |   |
    |   |- finish_reason == "length"?
    |   |   |- 非最后一次重试 -> max_tokens *= 1.1, retry
    |   |   +- 最后一次 -> 返回截断响应（让 ReAct 循环继续）
    |   |
    |   |- 重复内容检测:
    |   |   |- 取最后 50 字符, 检查在全文中出现次数 > 5
    |   |   |- 非最后一次重试 -> retry
    |   |   +- 最后一次 -> 返回
    |   |
    |   +- 正常返回
    |
    |- 异常处理:
    |   |- TimeoutError -> retry (等待 30 秒)
    |   |- CancelledError -> 直接 raise
    |   |- "longer than the model" -> 直接 raise (上下文超限)
    |   +- 其他 -> retry (等待 30 秒)
```

---

## 十、完整的一次任务执行时序图

```
时间轴 -->

[main.py]
    | Hydra 配置加载
    v
[create_pipeline_components]
    | 创建 ToolManager x N (主 Agent + 子 Agents)
    | 创建 OutputFormatter
    v
[execute_task_pipeline]
    | 创建 TaskLog
    | 创建 LLM Client (OpenAI/Anthropic)
    | 创建 Orchestrator
    v
[run_main_agent]
    | 获取 tool_definitions
    | 生成 system_prompt
    | message_history = [user: task]
    |
    |  +-- Turn 1 --+
    |  | LLM Call    |  system_prompt + message_history -> LLM
    |  | 解析响应     |  XML 解析工具调用
    |  | 参数修正     |  fix_tool_call_arguments
    |  | 重复检测     |  _check_duplicate_query
    |  | 执行工具     |  ToolManager.execute_tool_call (MCP Stdio)
    |  | 错误检测     |  should_rollback_result
    |  | 更新历史     |  update_message_history (合并为 user 消息)
    |  | 上下文检查   |  ensure_summary_context
    |  +-------------+
    |
    |  +-- Turn 2 --+
    |  | (同上)      |  旧 tool result 被替换为 "omitted"
    |  +-------------+
    |
    |  ... (最多 200-600 轮)
    |
    |  +-- Turn N (子Agent调用) --+
    |  | 检测到 server="agent-browsing"
    |  | run_sub_agent()
    |  |   +-- Sub Turn 1 --+
    |  |   | (独立循环)       |
    |  |   +-- Sub Turn M --+
    |  |   | 子Agent summary |
    |  |   +--> 返回结果给主Agent
    |  +---------------------------+
    |
    |  +-- 最终答案生成 --+
    |  | 注入 summary_prompt
    |  | LLM 生成最终答案
    |  | 提取 \boxed{} 内容
    |  | (如果失败) 生成 failure_experience_summary
    |  +------------------+
    |
    v
[pipeline 收尾]
    | 保存 TaskLog (JSON)
    | 关闭 LLM Client
    | 返回 (final_summary, boxed_answer, log_path, failure_summary)
```

---

## 十一、关键代码文件索引

| 流程环节 | 文件 | 核心函数/类 |
|---------|------|------------|
| 入口 | `main.py` | `amain()` |
| 管线创建 | `core/pipeline.py` | `create_pipeline_components()`, `execute_task_pipeline()` |
| Agent 循环 | `core/orchestrator.py` | `Orchestrator.run_main_agent()`, `run_sub_agent()` |
| 回滚处理 | `core/orchestrator.py` | `_handle_response_format_issues()`, `_check_duplicate_query()` |
| LLM 调用 | `core/answer_generator.py` | `AnswerGenerator.handle_llm_call()` |
| 最终答案 | `core/answer_generator.py` | `generate_and_finalize_answer()`, `generate_failure_summary()` |
| 工具执行 | `core/tool_executor.py` | `ToolExecutor.fix_tool_call_arguments()`, `should_rollback_result()` |
| Prompt 模板 | `utils/prompt_utils.py` | `generate_mcp_system_prompt()`, `generate_agent_summarize_prompt()` |
| XML 解析 | `utils/parsing_utils.py` | `parse_llm_response_for_tool_calls()` |
| LLM 基类 | `llm/base_client.py` | `BaseClient.create_message()`, `_remove_tool_result_from_messages()` |
| OpenAI 客户端 | `llm/providers/openai_client.py` | `OpenAIClient._create_message()`, `ensure_summary_context()` |
| 输出格式化 | `io/output_formatter.py` | `OutputFormatter._extract_boxed_content()` |
| 流式输出 | `core/stream_handler.py` | `StreamHandler.update()` |
| MCP 配置 | `config/settings.py` | `create_mcp_server_parameters()` |
| 任务日志 | `logging/task_logger.py` | `TaskLog` |
