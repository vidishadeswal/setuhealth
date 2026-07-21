import os

# torch and faiss-cpu each bundle their own OpenMP runtime; loading both in one
# process (as this app does — embeddings/reranking via torch, vector search via
# faiss) causes an OpenMP library collision that segfaults on macOS. Must be set
# before either library is imported anywhere, hence: top of the package __init__.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
