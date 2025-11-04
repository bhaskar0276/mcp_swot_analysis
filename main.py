from mcp.server.fastmcp import FastMCP
from test_swot import test_api_endpoint
import requests

mcp = FastMCP("swot analysis mcp ")


@mcp.tool()
async def swot_analysis():
    return test_api_endpoint()

@mcp.tool()
async def call_Bhaskar():
    return "Hello Man"


if __name__ == "__main__":
    mcp.run()
