def get_top_service_codes(query: str, pre_clarification: bool, tool_context: ToolContext) -> Dict[str, Any]:

    client = genai.Client(vertexai=True, project="anbc-dev-prv-ps-ds", location="us-east4")

    # Load FAISS and metadata
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(BASE_DIR, "scm_index.faiss")
    metadata_path = os.path.join(BASE_DIR, "scm_metadata.pkl")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Embed query
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=3072
        )
    )

    query_vector = np.array(response.embedding.values, dtype="float32")

    # Query FAISS
    k = 7
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    top_matches = metadata.iloc[indices[0]].copy()

    # Similarity score
    similarity_scores = 1 / (1 + distances[0])
    top_matches["similarity_score"] = similarity_scores

    # Weighted score
    top_matches["weighted_score"] = (
        0.8 * top_matches["similarity_score"] +
        0.2 * top_matches["total_claim_count"]
    )

    # Sort
    top_matches = top_matches.sort_values(by="weighted_score", ascending=False)

    # Save to context
    tool_context.state["query"] = query
    tool_context.state["top_service_codes"] = top_matches.to_dict(orient="records")

    # Prepare summary string for LLM
    subset = top_matches.head(3)
    summary = "\n".join([
        f"{row['Consumer-Friendly Title']}: {row['Consumer-Friendly Description']} "
        f"(Claims: {row['total_claim_count']})"
        for _, row in subset.iterrows()
    ])
    tool_context.state["top_service_codes_str"] = summary

    # RETURN → this goes to the ADK UI
    return {
        "top_service_codes": top_matches.to_dict(orient="records")
    }
