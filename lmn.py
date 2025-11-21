def after_tool_callback(tool, args, tool_context, tool_response):
    return f"Here are the top service codes:\n{tool_response['result']}"
