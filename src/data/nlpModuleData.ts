import { ModuleContent } from "./moduleContent";

export const nlpLLMModule: ModuleContent = {
  id: "nlp-llm",
  title: "NLP & Large Language Models",
  subtitle: "From tokenization to agentic AI — understand the full NLP stack powering modern LLMs, transformers, alignment, and intelligent agents.",
  color: "200 80% 55%",
  theory: [
    {
      title: "Intuition",
      content:
        "Language models learn to predict the next word. This simple objective, scaled to billions of parameters and trillions of tokens, produces emergent abilities: reasoning, code generation, and instruction following. Understanding the stack — from tokenization to alignment — is essential for building and using modern AI systems.",
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
        "Self-attention is the core operation of the Transformer. Each token computes Query, Key, and Value vectors by multiplying its embedding with learned weight matrices. The dot product QKᵀ computes all n² pairwise token similarities in one matrix multiply. The √dₖ scaling prevents dot products from growing with dimension (Var(q·k) = dₖ for unit-variance components), keeping softmax gradients stable. Without scaling, softmax becomes near-one-hot with vanishing gradients.",
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
        "Multi-head attention runs h parallel attention heads, each with its own W^Q, W^K, W^V projections at reduced dimension dₖ = d_model/h. Different heads learn different relationship types: Head 1 might capture syntactic dependencies (subject–verb), Head 2 coreference (he→John), Head 3 positional adjacency. Positional encoding injects sequence order information since attention is permutation-invariant. Sinusoidal encodings use sin/cos at different frequencies; RoPE (Rotary Position Embeddings) encodes relative positions through rotation matrices.",
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
        "A Transformer layer applies multi-head self-attention followed by a feed-forward network (FFN), with layer normalization and residual connections at each stage. The Pre-LN formulation (used in modern LLMs) applies LayerNorm before each sub-layer for more stable training. The FFN expands the dimension by 4× with a nonlinearity (ReLU or SwiGLU in LLaMA), then projects back. Most parameters (8·d²_model) live in the FFN, not the attention (4·d²_model).",
      equations: [
        {
          label: "Transformer Layer (Pre-LN)",
          tex: "x' = x + \\text{MHA}(\\text{LN}(x)), \\quad \\text{output} = x' + \\text{FFN}(\\text{LN}(x'))",
        },
        {
          label: "Feed-Forward Network",
          tex: "\\text{FFN}(x) = \\max(0, xW_1 + b_1) W_2 + b_2",
          variables: [
            { symbol: "W₁ ∈ ℝ^{d×4d}", meaning: "expansion projection" },
            { symbol: "W₂ ∈ ℝ^{4d×d}", meaning: "compression projection" },
          ],
        },
        {
          label: "Layer Normalization",
          tex: "\\text{LN}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta",
        },
      ],
    },
    {
      title: "BERT vs GPT — Encoder vs Decoder",
      content:
        "BERT uses bidirectional masked language modeling (MLM): mask 15% of tokens and predict them from full context. GPT uses autoregressive causal language modeling (CLM): predict the next token given all previous tokens. BERT excels at understanding tasks (classification, NER) via fine-tuning the [CLS] token. GPT excels at generation by sequential sampling. Decoding strategies include greedy (argmax), beam search (top-K partial sequences), nucleus sampling (top-p), and temperature scaling.",
      equations: [
        {
          label: "BERT MLM Objective",
          tex: "\\mathcal{L}_{\\text{MLM}} = -\\sum_{i \\in \\mathcal{M}} \\log P(x_i | x_{\\setminus \\mathcal{M}}; \\theta)",
        },
        {
          label: "GPT CLM Objective",
          tex: "\\mathcal{L}_{\\text{CLM}} = -\\sum_t \\log P(x_t | x_1, \\ldots, x_{t-1}; \\theta)",
        },
      ],
    },
    {
      title: "LLM Training Pipeline — SFT, RLHF & DPO",
      content:
        "Modern LLM training has four stages: (1) Pre-training on trillions of tokens with CLM loss, (2) Supervised Fine-Tuning (SFT) on high-quality instruction-response pairs, (3) RLHF — train a reward model on human preferences using Bradley-Terry, then optimize the policy with PPO and KL penalty, (4) DPO — Direct Preference Optimization eliminates the separate reward model by deriving a closed-form solution where the implicit reward is β·log(π_θ/π_ref). DPO is equivalent to RLHF but simpler to implement.",
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
      title: "Neural Scaling Laws",
      content:
        "Chinchilla scaling law: model loss L(N,D) = E + A/Nᵅ + B/Dᵝ, where N = parameters, D = tokens. Optimal allocation: N* ∝ C^0.5, D* ∝ C^0.5 — meaning parameters and data should scale equally. GPT-3 (175B params, 300B tokens) was under-trained by 10×. Chinchilla (70B params, 1.4T tokens) was optimal. Modern models (LLaMA 3) over-train intentionally for better inference efficiency. Emergent abilities appear at specific scales: instruction following ~8B, arithmetic ~13B, chain-of-thought ~100B.",
      equations: [
        {
          label: "Chinchilla Scaling Law",
          tex: "L(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}",
          variables: [
            { symbol: "E = 1.69", meaning: "irreducible entropy" },
            { symbol: "α = 0.34, β = 0.28", meaning: "power-law exponents" },
            { symbol: "FLOPs ≈ 6ND", meaning: "compute budget for forward + backward" },
          ],
        },
      ],
    },
    {
      title: "RAG & LoRA — Retrieval and Efficient Fine-Tuning",
      content:
        "RAG (Retrieval-Augmented Generation) grounds LLM responses in retrieved documents: retrieve top-k relevant chunks via dense embedding similarity (FAISS HNSW in O(log n)), then condition generation on the retrieved context. LoRA (Low-Rank Adaptation) freezes base weights and adds trainable low-rank matrices B·A where rank r ≪ min(d,k). For d=k=4096 and r=16, this saves 99.6% of parameters. QLoRA combines 4-bit quantized base weights with BF16 LoRA adapters, fitting 65B models on a single 48GB GPU. LoRA adapters merge at inference: W' = W₀ + BA with zero additional latency.",
      equations: [
        {
          label: "RAG Retrieval Score",
          tex: "P(d | x) \\propto \\exp\\left(\\frac{E_q(x) \\cdot E_d(d)}{\\tau}\\right)",
        },
        {
          label: "LoRA Weight Update",
          tex: "W' = W_0 + \\Delta W = W_0 + B \\cdot A, \\quad B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times k}",
          variables: [
            { symbol: "r", meaning: "LoRA rank (typically 4–64)" },
            { symbol: "savings", meaning: "1 - r(d+k)/(dk) ≈ 99.6% for r=16" },
          ],
        },
      ],
    },
    {
      title: "Agentic AI & Tool Use",
      content:
        "LLM agents combine reasoning with tool use. The ReAct framework interleaves Thought (reasoning about current state), Action (selecting a tool), and Observation (processing tool output). The agent policy π_θ(aₜ|hₜ) is the LLM conditioned on the full history. MCP (Model Context Protocol) provides a standardized JSON-RPC 2.0 interface for tool definitions and calls. Security requires OAuth 2.1, rate limiting, and sandboxed execution. Temperature T controls exploration vs exploitation in tool selection.",
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
      title: "Real-World Applications",
      content:
        "The NLP/LLM stack powers chatbots, code assistants (Copilot, Cursor), search engines (Perplexity), document analysis, translation, and agentic workflows. Understanding tokenization explains why LLMs struggle with character-level tasks. Understanding attention explains context window limits. Understanding alignment (RLHF/DPO) explains why models refuse certain requests. Understanding LoRA enables efficient domain adaptation. Understanding agents enables building autonomous AI systems.",
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
  ],
  papers: [
    { year: 2013, title: "Word2Vec", authors: "Mikolov et al.", venue: "NeurIPS", summary: "Skip-gram and CBOW for learning word embeddings via prediction." },
    { year: 2017, title: "Attention Is All You Need", authors: "Vaswani et al.", venue: "NeurIPS", summary: "Introduced the Transformer architecture with self-attention replacing recurrence." },
    { year: 2018, title: "BERT", authors: "Devlin et al.", venue: "NAACL", summary: "Bidirectional pre-training via masked language modeling — set new benchmarks on 11 NLP tasks." },
    { year: 2020, title: "GPT-3", authors: "Brown et al.", venue: "NeurIPS", summary: "175B parameter model demonstrating few-shot learning capabilities." },
    { year: 2021, title: "LoRA", authors: "Hu et al.", venue: "ICLR", summary: "Low-rank adaptation for efficient LLM fine-tuning — 99.6% parameter savings." },
    { year: 2022, title: "Chinchilla", authors: "Hoffmann et al.", venue: "NeurIPS", summary: "Compute-optimal scaling laws — parameters and data should scale equally." },
    { year: 2022, title: "InstructGPT / RLHF", authors: "Ouyang et al.", venue: "NeurIPS", summary: "Alignment via reinforcement learning from human feedback." },
    { year: 2023, title: "DPO", authors: "Rafailov et al.", venue: "NeurIPS", summary: "Direct Preference Optimization — closed-form alternative to RLHF without reward model." },
    { year: 2023, title: "ReAct", authors: "Yao et al.", venue: "ICLR", summary: "Interleaving reasoning and acting for LLM agents with tool use." },
    { year: 2024, title: "LLaMA 3", authors: "Meta AI", venue: "arXiv", summary: "Open-weight LLMs up to 405B parameters with 15T+ training tokens." },
  ],
};
