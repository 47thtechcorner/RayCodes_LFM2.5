import os
import json
import ollama

# 1. Load clinical abstracts
def load_abstracts():
    with open("medical_abstracts.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
    abstracts = []
    for entry in content.replace("\r\n", "\n").split("\n\n"):
        lines = entry.strip().split("\n")
        lang, title, text = "Unknown", "Untitled", ""
        for line in lines:
            if line.startswith("[Language:"):
                lang = line.replace("[Language:", "").replace("]", "").strip()
            elif line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Abstract:"):
                text = line.replace("Abstract:", "").strip()
        if text:
            abstracts.append({
                "lang": lang,
                "title": title,
                "abstract": text,
                "full_text": f"Title: {title}\nAbstract: {text}"
            })
    return abstracts

# Helper to compute cosine similarity manually
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = sum(a * a for a in v1) ** 0.5
    norm_b = sum(b * b for b in v2) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

# 2. Main pipeline
def main():
    print("Loading medical abstracts...")
    abstracts = load_abstracts()
    
    # Load or compute cached embeddings via Ollama
    cache_file = "embeddings_cache.json"
    embeddings = []
    
    if os.path.exists(cache_file):
        print("Loading pre-computed document embeddings from local cache...")
        with open(cache_file, "r") as f:
            embeddings = json.load(f)
    else:
        print("Pre-computing document embeddings via Ollama (nomic-embed-text)...")
        for i, doc in enumerate(abstracts):
            print(f"[{i+1}/{len(abstracts)}] Embedding: {doc['title']}...")
            try:
                resp = ollama.embeddings(model="nomic-embed-text", prompt=doc["full_text"])
                embeddings.append(resp["embedding"])
            except Exception as e:
                print(f"Failed to generate embedding for '{doc['title']}': {e}")
                embeddings.append([])
        
        # Save cache
        with open(cache_file, "w") as f:
            json.dump(embeddings, f)
        print("Document embeddings successfully cached.")

    query = input("\nEnter your clinical query: ")
    if not query.strip():
        return
    
    print("Encoding query via Ollama (nomic-embed-text)...")
    try:
        q_resp = ollama.embeddings(model="nomic-embed-text", prompt=query)
        q_emb = q_resp["embedding"]
    except Exception as e:
        print(f"Error encoding query: {e}")
        return
        
    # Calculate similarities
    scored_docs = []
    for doc, doc_emb in zip(abstracts, embeddings):
        if not doc_emb:
            continue
        score = cosine_similarity(q_emb, doc_emb)
        scored_docs.append((score, doc))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- Top Retrieved Sources ---")
    retrieved_contexts = []
    for score, doc in scored_docs[:3]:
        print(f"[{doc['lang']}] {doc['title']} (Similarity Score: {score:.4f})")
        retrieved_contexts.append(doc["full_text"])
        
    # LLM Synthesis
    print("\nSynthesizing explanation with Gemma 4...")
    context_str = "\n\n".join(retrieved_contexts)
    prompt = f"Using the following medical trial abstracts, answer the user query in the query's language.\n\nContext:\n{context_str}\n\nQuery: {query}\n\nAnswer:"
    
    try:
        response = ollama.chat(model='gemma4:e2b', messages=[{"role": "user", "content": prompt}])
        synthesis = response['message']['content']
        print("\n=== Gemma 4 Synthesis ===")
        print(synthesis)
        
        # Save to markdown file
        with open("app_output.md", "a", encoding="utf-8") as out:
            out.write(f"# Clinical Query: {query}\n\n")
            out.write("## Retrieved Sources\n")
            for score, doc in scored_docs[:3]:
                out.write(f"* **[{doc['lang']}] {doc['title']}** (Similarity Score: {score:.4f})\n")
            out.write("\n## Gemma 4 Synthesis\n")
            out.write(f"{synthesis}\n\n---\n\n")
        print("\n[INFO] Response successfully appended to app_output.md")
    except Exception as e:
        print(f"\nOllama synthesis failed: {e}")
        print("(Note: Gemma 4 requires 7.2GB size, check system memory resources.)")

if __name__ == "__main__":
    main()
