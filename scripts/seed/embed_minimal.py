from rtfm_agent.ingestion.documents import load_asc_files
from rtfm_agent.ingestion.chunker import load_and_chunk_docs
from fastembed import TextEmbedding

docs = load_asc_files("docs/progit2")
chunks = load_and_chunk_docs(docs)
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# Test embedding a single chunk
vec_gen = model.embed([chunks[0]["chunk_text"]])
vec = list(vec_gen)[0]
print(f'vec type: {type(vec)}')
print(f'has iter: {hasattr(vec, "__iter__")}')
if hasattr(vec, "__iter__"):
    vec_list = [float(v) for v in vec]
    print(f'vec_list len: {len(vec_list)}')
    print(f'first 5: {vec_list[:5]}')