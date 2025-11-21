def after_tool_callback(tool: BaseTool, args: Dict[str, Any], 
                        tool_context: ToolContext, tool_response: Dict) -> str:
    """
    Formats the tool response into a clean message for the user.
    """

    # Extract results from tool output
    if isinstance(tool_response, dict) and "result" in tool_response:
        results = tool_response["result"]
    else:
        return "The tool finished but returned no usable data."

    # Build user-friendly summary
    message_lines = ["Here are the top matching service codes:"]
    for item in results[:3]:   # Top 3 for readability
        code = item.get("primary_svc_cd", "")
        title = item.get("Consumer-Friendly Title", "")
        desc = item.get("Consumer-Friendly Description", "")
        claims = item.get("total_claim_count", "")

        message_lines.append(
            f"- *{code}* — {title}\n  ({desc}) | Claims: {claims}"
        )

    return "\n".join(message_lines)
                            
