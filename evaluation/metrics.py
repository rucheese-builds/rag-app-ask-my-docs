import re
from langchain_ollama import OllamaLLM

class GeminiEvalLLM:
    def __init__(self):
        import google.generativeai as genai
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in your .env file!")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.fallback_model = OllamaLLM(model="mistral", temperature=0.0)

    def invoke(self, prompt: str) -> str:
        import time
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                time.sleep(1.0) # Avoid hitting Gemini free tier rate limits (15 RPM)
                return response.text.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "service_disabled" in err_str or "blocked" in err_str or "403" in err_str:
                    print(f"[Gemini Eval] API is disabled or blocked. Falling back to local Ollama Mistral...")
                    return self.fallback_model.invoke(prompt)
                print(f"[Gemini Eval] Attempt {attempt+1} failed: {e}")
                time.sleep(2.0)
        print("[Gemini Eval] All attempts failed. Falling back to local Ollama Mistral...")
        return self.fallback_model.invoke(prompt)

def get_eval_llm():
    try:
        print("[Evaluator] Initializing Google Gemini API (gemini-1.5-flash) as judge...")
        return GeminiEvalLLM()
    except Exception as e:
        print(f"[Evaluator] Gemini init failed ({e}). Falling back to local Ollama Mistral.")
        return OllamaLLM(model="mistral", temperature=0.0)

def _parse_score(response):
    match = re.search(r"SCORE:\s*([0-9.]+)", response)
    if match:
        val = match.group(1).rstrip('.')
        try:
            return min(1.0, max(0.0, float(val)))
        except ValueError:
            pass
    return None

def evaluate_faithfulness(llm, question, context, answer):
    """
    Measures if the generated answer is strictly grounded in the retrieved context.
    Score: Number of supported claims / total claims.
    """
    if "I don't have enough information" in answer or "outside the scope" in answer.lower():
        # Fallback answers are considered faithful if indeed no info is there
        return 1.0

    prompt = f"""You are an evaluator assessing the faithfulness of a generated answer.
Your task is to identify key claims in the generated answer and check if each claim is supported by the retrieved context.

Context:
{context}

Generated Answer:
{answer}

Identify the distinct claims in the generated answer. For each claim, check if it is directly supported by the context.
Output format:
Claim 1: [text of claim]
Supported: YES or NO
Claim 2: [text of claim]
Supported: YES or NO

Then, output the final score in the format:
SCORE: [ratio of YES answers to total claims, e.g. 0.75 or 1.00]

Be strict. If a claim cannot be inferred from the context, mark it as Supported: NO.
"""
    try:
        response = llm.invoke(prompt)
        # Parse output for SCORE: X.XX
        score = _parse_score(response)
        if score is not None:
            return score
        
        # Fallback: parse YES/NO
        supported_lines = re.findall(r"Supported:\s*(YES|NO)", response, re.IGNORECASE)
        if not supported_lines:
            return 0.5 # Default fallback
        yes_count = sum(1 for line in supported_lines if "yes" in line.lower())
        return yes_count / len(supported_lines)
    except Exception as e:
        print(f"Error in faithfulness evaluation: {e}")
        return 0.5

def evaluate_answer_relevance(llm, question, answer):
    """
    Measures how directly the answer addresses the question.
    """
    prompt = f"""You are an evaluator assessing answer relevance.
Evaluate how directly and completely the generated answer addresses the question.
Penalize answers that are off-topic, contain redundant information, or fail to answer the question.
Note: If the question is out-of-domain/unanswerable, and the answer correctly identifies that it cannot be answered, score it 1.0.

Question:
{question}

Generated Answer:
{answer}

Output format:
Explanation: [brief explanation of your rating]
SCORE: [rating between 0.0 and 1.0, e.g., 0.90]
"""
    try:
        response = llm.invoke(prompt)
        score = _parse_score(response)
        if score is not None:
            return score
        return 0.5
    except Exception as e:
        print(f"Error in answer relevance evaluation: {e}")
        return 0.5

def evaluate_context_precision(llm, question, retrieved_chunks):
    """
    Measures if the retrieved context chunks (top 3) are relevant to the question.
    Computes Precision @ K based on LLM binary classification.
    """
    if not retrieved_chunks:
        return 0.0

    chunk_texts = ""
    for i, doc in enumerate(retrieved_chunks):
        chunk_texts += f"--- Chunk {i+1} ---\n{doc.page_content}\n\n"

    prompt = f"""You are an evaluator assessing context relevance.
For each of the retrieved chunks below, determine if it contains relevant information to answer the question.

Question:
{question}

Retrieved Chunks:
{chunk_texts}

For each chunk, determine if it is relevant. Output exactly in this format:
Chunk 1 Relevant: YES or NO
Chunk 2 Relevant: YES or NO
Chunk 3 Relevant: YES or NO
"""
    try:
        response = llm.invoke(prompt)
        
        # Parse YES/NO for each chunk
        relevance = []
        lines = response.split("\n")
        for i in range(1, len(retrieved_chunks) + 1):
            chunk_found = False
            for line in lines:
                if f"chunk {i}" in line.lower() or f"chunk{i}" in line.lower():
                    if "yes" in line.lower():
                        relevance.append(True)
                        chunk_found = True
                        break
                    elif "no" in line.lower():
                        relevance.append(False)
                        chunk_found = True
                        break
            if not chunk_found:
                pattern = rf"Chunk\s*{i}.*?(YES|NO)"
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    relevance.append(match.group(1).upper() == "YES")
                else:
                    relevance.append(False)
        
        # Calculate Context Precision:
        # P@k = (relevant chunks up to k) / k
        # Context Precision = sum(P@k * rel_k) / sum(rel_k)
        precisions = []
        relevant_so_far = 0
        for k, rel in enumerate(relevance):
            if rel:
                relevant_so_far += 1
                precisions.append(relevant_so_far / (k + 1))
        
        if not precisions:
            return 0.0
        return sum(precisions) / len(precisions)
    except Exception as e:
        print(f"Error in context precision evaluation: {e}")
        return 0.5

def evaluate_context_recall(llm, ground_truth, context):
    """
    Measures if the retrieved context covers all key facts in the ground truth answer.
    """
    if "outside the scope" in ground_truth.lower() or "not in corpus" in ground_truth.lower():
        # Negative/out-of-domain queries
        return 1.0

    prompt = f"""You are an evaluator assessing context recall.
Your task is to identify key factual statements in the ground truth answer and check if each fact is present in the retrieved context.

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

Identify key factual statements in the ground truth. Check if each statement is found in the context.
Output format:
Fact 1: [text of fact]
Found: YES or NO
Fact 2: [text of fact]
Found: YES or NO

Then, output the final score in the format:
SCORE: [ratio of YES answers to total facts, e.g. 0.80 or 1.00]
"""
    try:
        response = llm.invoke(prompt)
        score = _parse_score(response)
        if score is not None:
            return score
        
        found_lines = re.findall(r"Found:\s*(YES|NO)", response, re.IGNORECASE)
        if not found_lines:
            return 0.5
        yes_count = sum(1 for line in found_lines if "yes" in line.lower())
        return yes_count / len(found_lines)
    except Exception as e:
        print(f"Error in context recall evaluation: {e}")
        return 0.5
