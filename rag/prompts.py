"""
Day 11: Prompt Templates for AI Retail Intelligence Assistant

Three prompt types:
1. BASE PROMPT     — general store assistant
2. ANALYTICS PROMPT — questions about revenue/margins/KPIs
3. INVENTORY PROMPT — questions about stock/restock/dead stock
4. RAG PROMPT       — answers grounded in retrieved context (Day 16)

Why separate prompts per domain:
Each domain needs different instructions. Analytics questions
need the model to reason about numbers. Inventory questions
need the model to prioritize urgency. RAG questions need the
model to stay strictly within retrieved context.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

RETAIL_ASSISTANT_PERSONA = """You are an AI Retail Intelligence Assistant for a Pakistani kiryana (corner store) business.

Your role:
- Analyze sales data, inventory levels, and competitor prices
- Give practical, actionable advice to the store owner
- Always answer in clear, simple language (the owner may not be technical)
- When mentioning prices, always use PKR (Pakistani Rupees)
- Be concise — the owner is busy running a store

You have access to the store's:
- Product catalog (Pakistani brands: Tapal, Shan, National, Surf Excel, etc.)
- 60+ days of sales transaction history
- Current inventory stock levels
- Profit margins per product and category

IMPORTANT RULES:
- Never make up numbers. Only use data provided to you.
- If you don't have enough data to answer, say so clearly.
- Always prioritize CRITICAL stock alerts over general advice.
- Format currency as Rs. X,XXX (e.g., Rs. 1,250)
"""


# ── 2. Base Chat Prompt ───────────────────────────────────────────────────────
def get_base_prompt() -> ChatPromptTemplate:
    """
    Basic conversational prompt.
    Used for general store questions without specific data context.
    
    {question} is the placeholder LangChain fills at runtime.
    
    ChatPromptTemplate uses message roles (system/human/assistant)
    which maps to how chat models like phi3 expect input.
    """
    return ChatPromptTemplate.from_messages([
        ("system", RETAIL_ASSISTANT_PERSONA),
        ("human", "{question}"),
    ])


# ── 3. Analytics Prompt ───────────────────────────────────────────────────────
def get_analytics_prompt() -> ChatPromptTemplate:
    """
    Prompt for revenue/margin/KPI questions.
    Injects live store metrics as context before the question.
    
    {store_context} = JSON dump of get_store_kpis() result
    {question}      = user's question
    
    Why inject context in the prompt:
    phi3 doesn't have access to your DB. You pass the data
    as text in the prompt, and phi3 reasons over it.
    This is "context-stuffing" — simpler than RAG but limited
    to how much data fits in the context window (~4000 tokens).
    """
    template = """You are an AI Retail Intelligence Assistant for a Pakistani kiryana store.

CURRENT STORE METRICS:
{store_context}

Based on the above store data, answer the following question clearly and concisely.
Only use the numbers provided above — do not invent figures.
If the data doesn't contain enough information, say "I need more data to answer this."

QUESTION: {question}

ANSWER:"""

    return ChatPromptTemplate.from_messages([
        ("system", RETAIL_ASSISTANT_PERSONA),
        ("human", template),
    ])


# ── 4. Inventory Prompt ───────────────────────────────────────────────────────
def get_inventory_prompt() -> ChatPromptTemplate:
    """
    Prompt for stock/restock/dead stock questions.
    Injects inventory health data + alert list.
    
    {inventory_context} = JSON of inventory health summary
    {alerts_context}    = list of low stock alerts
    {question}          = user's question
    """
    template = """You are an AI Retail Intelligence Assistant for a Pakistani kiryana store.

CURRENT INVENTORY STATUS:
{inventory_context}

LOW STOCK ALERTS:
{alerts_context}

Using the inventory data above, answer the following question.
Prioritize CRITICAL alerts in your response.
Give specific product names and quantities where available.
Always mention the supplier so the owner knows who to call.

QUESTION: {question}

ANSWER:"""

    return ChatPromptTemplate.from_messages([
        ("system", RETAIL_ASSISTANT_PERSONA),
        ("human", template),
    ])


# ── 5. RAG Prompt (used in Day 16) ───────────────────────────────────────────
def get_rag_prompt() -> ChatPromptTemplate:
    """
    The main RAG prompt — grounds answers in retrieved documents.
    
    {context}  = text chunks retrieved from FAISS vector store
    {question} = user's question
    
    The instruction "ONLY use the context below" is critical.
    Without it, phi3 mixes retrieved facts with its training
    data and hallucinates. This instruction is your
    hallucination prevention mechanism.
    
    This prompt is built today but fully activated Day 16
    when FAISS retrieval is wired in.
    """
    template = """You are a retail analyst for a Pakistani kiryana (grocery) store. Answer the owner's question using ONLY the store data below.

STORE DATA:
{context}

HOW TO ANSWER:
- Answer the EXACT question that is asked, directly and confidently.
- The data may contain "Question:" and "Answer:" lines. If one matches the owner's question, use that answer.
- Use only facts, product names, numbers and PKR amounts found in the data above. Never invent or guess.
- Name the specific product(s) the question is about. Never substitute or mix up a different product.
- Do NOT hedge. Only reply "I don't have that information." when the answer is genuinely absent from the data above — never when the fact is present.
- Be concise: 1 to 3 sentences.

Question: {question}
Answer:"""

    return ChatPromptTemplate.from_messages([
        ("system", RETAIL_ASSISTANT_PERSONA),
        ("human", template),
    ])


# ── 6. Standalone question rewriter (used in Day 16 conversational RAG) ───────
def get_question_rewriter_prompt() -> PromptTemplate:
    """
    Rewrites follow-up questions to be standalone.
    
    Why: In a conversation:
    - "What are my best sellers?" → standalone, fine
    - "What about their margins?" → ambiguous, refers to "best sellers"
    
    This prompt makes the second question standalone:
    "What are the profit margins of my best selling products?"
    
    FAISS can then retrieve the right documents for it.
    Used in Day 16 for conversational RAG memory.
    """
    template = """Given the conversation history and a follow-up question, 
rewrite the follow-up question to be a complete standalone question.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""

    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )