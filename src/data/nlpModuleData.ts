import { ModuleContent } from "./moduleContent";

export const nlpLLMModule: ModuleContent = {
  id: "nlp-llm",
  title: "Agentic AI",
  subtitle: "From LLM foundations to autonomous agents — understand tools, memory, multi-agent systems, MCP, and the full agentic AI stack.",
  color: "200 80% 55%",
  theory: [
    {
      title: "Intuition",
      content:
        "Language models learn to predict the next word. This simple objective, scaled to billions of parameters and trillions of tokens, produces emergent abilities: reasoning, code generation, and instruction following. But an LLM alone is stateless and passive — it cannot act in the real world. This limitation drives the evolution from LLMs → Tools → Agents → Multi-Agent Systems → Autonomous Organizations. Understanding this full stack is essential for building modern AI systems.",
    },
    {
      title: "Foundation: What is an LLM?",
      content:
        "At the core of agentic AI is a Large Language Model — a probabilistic function that maps input text to output text. LLMs are capable of natural language understanding, approximate reasoning, code generation, and limited planning. However, they are fundamentally stateless (no memory between calls), passive (cannot take actions), and prone to hallucination. This critical limitation — LLM alone = no real-world action — is what motivates the entire agentic AI stack: tools, agents, memory, and orchestration.",
    },
    {
      title: "Tokenization & BPE",
      content:
        "Tokenization converts raw text into a sequence of integer IDs from a fixed vocabulary. Byte Pair Encoding (BPE) starts with individual characters and iteratively merges the most frequent adjacent pair into a new token. SentencePiece uses a unigram language model that maximizes corpus likelihood. The vocabulary size controls the trade-off between sequence length and vocabulary coverage. GPT-2 uses ~50K tokens; modern models use 32K–128K.",
      equations: [
        {
          label: "BPE Merge Rule",
          tex: "(a^*, b^*) = \\arg\\max_{(a,b)} \\text{freq}(a, b)",
          variables: [
            { symbol: "a, b", meaning: "adjacent token pair in the corpus" },
            { symbol: "freq", meaning: "co-occurrence count across all words" },
          ],
        },
        {
          label: "SentencePiece Objective",
          tex: "V^* = \\arg\\max_V \\sum_{s \\in D} \\log P(s | V)",
        },
      ],
    },
    {
      title: "Word Embeddings & Representations",
      content:
        "Embeddings map discrete tokens to dense vectors in a continuous space where semantic relationships are preserved. Word2Vec (Skip-Gram with Negative Sampling) learns static embeddings by predicting context words. The famous analogy v(king) - v(man) + v(woman) ≈ v(queen) emerges from the parallelogram structure of relational vectors. Contextual embeddings (BERT, GPT) produce different vectors for the same word depending on its context — 'bank' near 'river' vs 'bank' near 'money'.",
      equations: [
        {
          label: "Skip-Gram Objective (SGNS)",
          tex: "\\mathcal{L} = \\sum_{(c,w) \\in D} \\log \\sigma(\\mathbf{v}_c \\cdot \\mathbf{v}_w) + k \\cdot \\mathbb{E}_{w' \\sim P_n} [\\log \\sigma(-\\mathbf{v}_c \\cdot \\mathbf{v}_{w'})]",
          variables: [
            { symbol: "σ(x)", meaning: "sigmoid function 1/(1+e⁻ˣ)" },
            { symbol: "Pₙ(w)", meaning: "noise distribution ∝ freq(w)^{3/4}" },
            { symbol: "k", meaning: "number of negative samples (typically 5–20)" },
          ],
        },
        {
          label: "Cosine Similarity",
          tex: "\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\cdot \\|\\mathbf{v}\\|_2} \\in [-1, 1]",
        },
      ],
    },
    {
      title: "Self-Attention Mechanism",
      content:
        "Self-attention is the core operation of the Transformer. Each token computes Query, Key, and Value vectors by multiplying its embedding with learned weight matrices. The dot product QKᵀ computes all n² pairwise token similarities in one matrix multiply. The √dₖ scaling prevents dot products from growing with dimension (Var(q·k) = dₖ for unit-variance components), keeping softmax gradients stable.",
      equations: [
        {
          label: "Scaled Dot-Product Attention",
          tex: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V",
          variables: [
            { symbol: "Q = XW^Q", meaning: "query matrix ∈ ℝⁿˣᵈᵏ" },
            { symbol: "K = XW^K", meaning: "key matrix ∈ ℝⁿˣᵈᵏ" },
            { symbol: "V = XW^V", meaning: "value matrix ∈ ℝⁿˣᵈᵥ" },
            { symbol: "√dₖ", meaning: "scaling factor to prevent gradient saturation" },
          ],
        },
      ],
    },
    {
      title: "Multi-Head Attention & Positional Encoding",
      content:
        "Multi-head attention runs h parallel attention heads, each with its own W^Q, W^K, W^V projections at reduced dimension dₖ = d_model/h. Different heads learn different relationship types. Positional encoding injects sequence order information since attention is permutation-invariant. RoPE (Rotary Position Embeddings) encodes relative positions through rotation matrices and is the standard in modern LLMs.",
      equations: [
        {
          label: "Multi-Head Attention",
          tex: "\\text{MHA}(X) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h) W^O",
        },
        {
          label: "Sinusoidal Positional Encoding",
          tex: "PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)",
        },
      ],
    },
    {
      title: "Full Transformer Block",
      content:
        "A Transformer layer applies multi-head self-attention followed by a feed-forward network (FFN), with layer normalization and residual connections. The Pre-LN formulation (used in modern LLMs) applies LayerNorm before each sub-layer for more stable training. The FFN expands the dimension by 4× with a nonlinearity (SwiGLU in LLaMA), then projects back.",
      equations: [
        {
          label: "Transformer Layer (Pre-LN)",
          tex: "x' = x + \\text{MHA}(\\text{LN}(x)), \\quad \\text{output} = x' + \\text{FFN}(\\text{LN}(x'))",
        },
        {
          label: "Feed-Forward Network",
          tex: "\\text{FFN}(x) = \\max(0, xW_1 + b_1) W_2 + b_2",
        },
      ],
    },
    {
      title: "LLM Training Pipeline — SFT, RLHF & DPO",
      content:
        "Modern LLM training has four stages: (1) Pre-training on trillions of tokens with CLM loss, (2) Supervised Fine-Tuning (SFT) on high-quality instruction-response pairs, (3) RLHF — train a reward model on human preferences using Bradley-Terry, then optimize the policy with PPO and KL penalty, (4) DPO — Direct Preference Optimization eliminates the separate reward model by deriving a closed-form solution. DPO is equivalent to RLHF but simpler to implement.",
      equations: [
        {
          label: "Reward Model (Bradley-Terry)",
          tex: "P(y_w \\succ y_l) = \\sigma(r_\\phi(x, y_w) - r_\\phi(x, y_l))",
        },
        {
          label: "DPO Loss",
          tex: "\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]",
        },
      ],
    },
    {
      title: "RAG & LoRA — Retrieval and Efficient Fine-Tuning",
      content:
        "RAG (Retrieval-Augmented Generation) grounds LLM responses in retrieved documents: retrieve top-k relevant chunks via dense embedding similarity, then condition generation on the retrieved context. LoRA (Low-Rank Adaptation) freezes base weights and adds trainable low-rank matrices B·A where rank r ≪ min(d,k), saving 99.6% of parameters. QLoRA combines 4-bit quantized base weights with BF16 LoRA adapters.",
      equations: [
        {
          label: "RAG Retrieval Score",
          tex: "P(d | x) \\propto \\exp\\left(\\frac{E_q(x) \\cdot E_d(d)}{\\tau}\\right)",
        },
        {
          label: "LoRA Weight Update",
          tex: "W' = W_0 + \\Delta W = W_0 + B \\cdot A, \\quad B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times k}",
        },
      ],
    },
    {
      title: "Tools: Giving LLMs Capabilities",
      content:
        "A tool is any function or API the LLM can call — calculator, web search, database query, file system, external APIs. The key insight: LLM decides what tool to call → tool executes → result returns to LLM → LLM continues reasoning. Tools extend LLMs beyond text, enable real-world interaction, and provide accuracy for tasks like math and retrieval where LLMs alone hallucinate. Tool calling is implemented via function definitions in the API schema that the LLM can invoke with structured arguments.",
    },
    {
      title: "Agents: Decision-Making Systems",
      content:
        "An agent is LLM + Tools + Control Loop. The core loop: (1) Observe — receive input and context, (2) Think — LLM reasons about what to do, (3) Act — make a tool call or generate a response, (4) Repeat until the task is done. The key property is autonomy: agents are decision-makers that choose their own actions. The ReAct framework formalizes this by interleaving Thought, Action, and Observation traces. The agent policy π_θ(aₜ|hₜ) is the LLM conditioned on the full interaction history.",
      equations: [
        {
          label: "Agent Policy",
          tex: "\\pi_\\theta(a_t | h_t) = \\text{LLM}_\\theta(a_t | s_0, a_0, o_0, \\ldots, s_t)",
          variables: [
            { symbol: "hₜ", meaning: "full interaction history up to time t" },
            { symbol: "aₜ", meaning: "action (tool call or text response)" },
            { symbol: "oₜ", meaning: "observation from tool execution" },
          ],
        },
        {
          label: "Agent Formal Definition",
          tex: "\\mathcal{M} = (S, A, T, R, \\pi_\\theta)",
        },
      ],
    },
    {
      title: "Agent Architecture — Planner, Executor, Controller",
      content:
        "Agents are structured systems beyond a simple loop. The Planner breaks complex tasks into subtasks. The Executor runs each step (tool calls, sub-agent invocations). The Controller manages flow, stopping conditions, and error recovery. The pattern: Planner → Executor → Feedback → Planner. Advanced architectures use tree-of-thought (branching plans), self-reflection (critiquing own outputs), and hierarchical decomposition (recursive subtask planning).",
    },
    {
      title: "Memory: Adding State to Agents",
      content:
        "Without memory, each agent step is independent. With memory, agents become stateful. Three types: (1) Short-term memory — conversation history within the context window, (2) Long-term memory — persistent storage in vector databases for retrieval across sessions, (3) Working memory — current task state and scratchpad for intermediate results. Memory enables continuity, learning from past interactions, and personalization. Vector databases (FAISS, Pinecone, Chroma) store embeddings for efficient similarity search.",
    },
    {
      title: "Multi-Agent Systems",
      content:
        "Single agents don't scale to complex tasks. Multi-agent systems use specialized agents that collaborate: Researcher (gathers information), Planner (decomposes tasks), Coder (writes code), Reviewer (validates outputs), Executor (runs actions). The flow: User → Planner → task distribution → specialized agents → Aggregator → final output. Frameworks: LangChain provides tool abstraction and basic agents; LangGraph adds graph-based orchestration with explicit state, control flow, nodes (agents), edges (transitions), and shared memory.",
    },
    {
      title: "MCP — Model Context Protocol",
      content:
        "MCP standardizes how agents discover and use tools across systems. Before MCP: every tool required custom integration. After MCP: tools are standardized services. Architecture: LLM Agent → MCP Client → MCP Server(s) → Tools/APIs. MCP Servers host tools and expose them via a JSON-RPC 2.0 protocol. MCP Clients connect to servers and discover tools dynamically. FastMCP (Python) lets you build MCP servers quickly with @tool decorators. LangChain MCP Adapters convert MCP tools into LangChain-compatible tools for use in LangGraph agents. MCP is essentially the 'API layer for AI agents'.",
    },
    {
      title: "OpenClaw & Paperclip — Agent OS and Organizations",
      content:
        "OpenClaw is an operating system for agents, combining: multiple specialized LLMs, persistent memory, local + MCP tools, and communication channels (agent↔agent, agent↔human). It enables autonomous multi-agent execution. Paperclip sits above as an organization layer — a coordination system for many agents, analogous to a company: CEO Agent (planner), Manager Agents (task decomposition), Worker Agents (execution), QA Agents (validation). The full stack: Paperclip (coordination) → OpenClaw agents → LangGraph workflows → MCP tools → External world.",
    },
    {
      title: "Evolution of Intelligence",
      content:
        "The progression from passive language model to autonomous organization follows a clear evolutionary path: LLM (Think) → Tools (Act) → Agent (Decide) → Memory (Remember) → Multi-Agent (Collaborate) → MCP (Integrate) → OpenClaw (Systemize) → Paperclip (Organize). Each layer adds a new capability. Understanding this stack is essential for building, deploying, and reasoning about modern AI systems — from simple chatbots to fully autonomous agent organizations.",
    },
  ],
  algorithms: [
    {
      name: "Transformer Training Pipeline",
      steps: [
        { step: "Tokenize Corpus", detail: "Build BPE vocabulary from training data; encode all text as token sequences" },
        { step: "Embed Tokens", detail: "Map token IDs to dense vectors; add positional encodings" },
        { step: "Self-Attention", detail: "Compute Q, K, V projections; scaled dot-product attention with causal mask" },
        { step: "Feed-Forward & Residuals", detail: "FFN with 4× expansion, LayerNorm, residual connections" },
        { step: "Pre-train with CLM", detail: "Next-token prediction on trillions of tokens with AdamW optimizer" },
        { step: "SFT on Instructions", detail: "Fine-tune on curated instruction-response pairs with low LR" },
        { step: "Align with DPO/RLHF", detail: "Optimize model against human preference data" },
      ],
    },
    {
      name: "ReAct Agent Loop",
      steps: [
        { step: "Receive User Query", detail: "Parse natural language input and initialize agent context" },
        { step: "Thought Generation", detail: "LLM reasons about what information is needed and which tool to use" },
        { step: "Action Selection", detail: "Select tool and generate arguments from available tool definitions" },
        { step: "Tool Execution", detail: "Execute tool call via MCP/function calling; receive observation" },
        { step: "Iterate or Answer", detail: "Continue reasoning loop or generate final answer to user" },
      ],
    },
    {
      name: "Multi-Agent Orchestration",
      steps: [
        { step: "Task Decomposition", detail: "Planner agent breaks user goal into subtasks with dependencies" },
        { step: "Agent Assignment", detail: "Route each subtask to specialized agent (researcher, coder, analyst)" },
        { step: "Parallel Execution", detail: "Independent subtasks run concurrently across agent pool" },
        { step: "Result Aggregation", detail: "Aggregator collects outputs, resolves conflicts, merges results" },
        { step: "Quality Review", detail: "QA agent validates output quality and requests re-execution if needed" },
      ],
    },
  ],
  papers: [
    { year: 2013, title: "Word2Vec", authors: "Mikolov et al.", venue: "NeurIPS", summary: "Skip-gram and CBOW for learning word embeddings via prediction." },
    { year: 2017, title: "Attention Is All You Need", authors: "Vaswani et al.", venue: "NeurIPS", summary: "Introduced the Transformer architecture with self-attention replacing recurrence." },
    { year: 2018, title: "BERT", authors: "Devlin et al.", venue: "NAACL", summary: "Bidirectional pre-training via masked language modeling." },
    { year: 2020, title: "GPT-3", authors: "Brown et al.", venue: "NeurIPS", summary: "175B parameter model demonstrating few-shot learning capabilities." },
    { year: 2021, title: "LoRA", authors: "Hu et al.", venue: "ICLR", summary: "Low-rank adaptation for efficient LLM fine-tuning." },
    { year: 2022, title: "InstructGPT / RLHF", authors: "Ouyang et al.", venue: "NeurIPS", summary: "Alignment via reinforcement learning from human feedback." },
    { year: 2023, title: "DPO", authors: "Rafailov et al.", venue: "NeurIPS", summary: "Direct Preference Optimization — closed-form alternative to RLHF." },
    { year: 2023, title: "ReAct", authors: "Yao et al.", venue: "ICLR", summary: "Interleaving reasoning and acting for LLM agents with tool use." },
    { year: 2024, title: "LLaMA 3", authors: "Meta AI", venue: "arXiv", summary: "Open-weight LLMs up to 405B parameters with 15T+ training tokens." },
    { year: 2024, title: "MCP", authors: "Anthropic", venue: "Open Standard", summary: "Model Context Protocol — standardized JSON-RPC 2.0 interface for agent tool use." },
    { year: 2025, title: "LangGraph", authors: "LangChain", venue: "Framework", summary: "Graph-based multi-agent orchestration with explicit state management." },
  ],
};
