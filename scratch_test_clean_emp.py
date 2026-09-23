from app.pipeline import load_env, load_database, load_generator, interpret_query, retrieve
load_env()
vdb, docs = load_database("chroma_db_bilanci")
gen = load_generator("groq")

q = "Quanti dipendenti ha Amazon nel 2023?"
interp = interpret_query(q, docs, gen)
print("Keywords extracted by LLM:", interp.get("keywords_en"))

# Let's test Chroma with pure English keywords
clean_kws = interp.get("keywords_en", [])
query_str = " ".join(clean_kws) if clean_kws else q
hits = vdb.find(query=query_str, k=20, filters={"$and": [{"company": {"$eq": "amazon"}}, {"fiscal_year": {"$in": [2023]}}]})

print(f"Top 5 hits for query '{query_str}':")
for i, (d, s) in enumerate(hits[:5]):
    idx = d.metadata.get("chunk_index")
    print(f"  Rank {i+1}: Chunk #{idx} (score: {s:.4f}) -> {d.page_content[:120].replace(chr(10), ' ')}")
