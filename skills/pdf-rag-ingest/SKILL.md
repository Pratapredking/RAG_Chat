---
name: pdf-rag-ingest
description: Ingest text-based or scanned PDF documents into a searchable RAG vector store, including extraction or OCR routing, chunking, embeddings, metadata, indexing, and retrieval verification. Use when the user wants PDFs uploaded, scanned, indexed, embedded, or made searchable; do not use for merely reading, summarizing, editing, or exporting a PDF.
---

# PDF RAG Ingest

Turn each requested PDF into searchable, source-attributed chunks while preserving the project's existing embedding and database conventions.

## Workflow

1. Resolve the exact PDF paths and target RAG project. Inspect existing ingestion, embedding, vector-index, and search code before adding another pipeline.
2. Validate each input as a readable PDF. Detect encrypted files and pages with little or no extracted text. Treat those pages as scanned content and use an available OCR path; if OCR is unavailable, report the affected pages instead of storing empty or invented text.
3. Reuse the project's embedding model and dimensionality. Never mix vectors from incompatible embedding models in one index. If a model change is required, explain that all documents in that index must be re-embedded.
4. Compute a stable document ID from the source bytes. Split extracted text into retrieval-sized chunks, retaining filename, page number, document ID, chunk index, and ingestion time.
5. Batch embedding requests. Upsert by `(document_id, chunk_index)` or replace only records scoped to the same document ID. Never clear an entire shared collection to ingest one PDF.
6. Ensure the vector index matches the stored vector field, dimensions, and similarity metric. Add a filter field for `document_id` when searches may be scoped to one document.
7. Verify ingestion with document/chunk counts and at least one semantic query based on known PDF content. Confirm results include usable text plus filename and page metadata.
8. Report what was ingested, whether OCR was used, the collection/index, chunk count, and the verification result. Clearly report skipped or failed pages.

## Safety and quality constraints

- Do not print API keys, database URIs, or `.env` contents. It is safe to report whether required variable names are present.
- Do not upload, commit, or retain private PDFs beyond the locations the user placed in scope.
- Prefer a dedicated PDF chunk collection over mixing document chunks with unrelated product records.
- Reject path traversal for uploaded filenames, validate file size/type, and avoid trusting the filename as the document identity.
- Keep page-level provenance so downstream answers can cite the original file and page.
- Database writes are authorized when the user explicitly asks to ingest, index, embed, store, or make the PDF searchable. Otherwise prepare the implementation and stop before writing external data.

## Project-specific routing

For the `ecommerce-chat-helper` repository, read [references/ecommerce-chat-helper.md](references/ecommerce-chat-helper.md) before implementing or running ingestion.
