"""Intercept the actual HTTP requests each framework sends to Ollama.
Patches the httpx client to log request bodies."""

import sys
sys.path.insert(0, ".")

import json
import time
import httpx

# Monkey-patch httpx to capture requests
_original_send = httpx.AsyncClient.send
_original_sync_send = httpx.Client.send
captured_requests = []

async def _patched_async_send(self, request, **kwargs):
    if "/v1/chat/completions" in str(request.url):
        body = json.loads(request.content)
        captured_requests.append({
            "url": str(request.url),
            "messages": body.get("messages", []),
            "response_format": body.get("response_format"),
            "temperature": body.get("temperature"),
            "model": body.get("model"),
            "tools": body.get("tools"),
            "extra_keys": [k for k in body.keys() if k not in ("messages", "response_format", "temperature", "model", "tools", "stream", "max_tokens")],
        })
    return await _original_send(self, request, **kwargs)

def _patched_sync_send(self, request, **kwargs):
    if "/v1/chat/completions" in str(request.url) or "/chat/completions" in str(request.url):
        try:
            body = json.loads(request.content)
            captured_requests.append({
                "url": str(request.url),
                "messages": body.get("messages", []),
                "response_format": body.get("response_format"),
                "temperature": body.get("temperature"),
                "model": body.get("model"),
                "tools": body.get("tools"),
                "extra_keys": [k for k in body.keys() if k not in ("messages", "response_format", "temperature", "model", "tools", "stream", "max_tokens")],
            })
        except:
            pass
    return _original_sync_send(self, request, **kwargs)

httpx.AsyncClient.send = _patched_async_send
httpx.Client.send = _patched_sync_send

# Also patch requests library (used by litellm/crewai)
try:
    import requests
    _original_requests_send = requests.Session.send
    def _patched_requests_send(self, request, **kwargs):
        if "/chat/completions" in str(request.url) or "/api/chat" in str(request.url):
            try:
                body = json.loads(request.body)
                captured_requests.append({
                    "url": str(request.url),
                    "messages": body.get("messages", []),
                    "response_format": body.get("response_format"),
                    "temperature": body.get("temperature"),
                    "model": body.get("model"),
                    "tools": body.get("tools"),
                    "extra_keys": [k for k in body.keys() if k not in ("messages", "response_format", "temperature", "model", "tools", "stream", "max_tokens")],
                })
            except:
                pass
        return _original_requests_send(self, request, **kwargs)
    requests.Session.send = _patched_requests_send
except ImportError:
    pass

REQUIREMENT = "The software must start to execute the sequence of tasks allocated to the execution schedule in less than 2 seconds after processor power-up."


def test_pydantic_ai():
    import asyncio
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider
    from schemas import EntityExtractionResult
    from prompts import ENTITY_SYSTEM, task_prompt

    provider = OpenAIProvider(base_url="http://172.31.128.1:11434/v1", api_key="ollama")
    profile = OpenAIModelProfile(supports_json_schema_output=False, supports_json_object_output=True, default_structured_output_mode="prompted")
    agent = Agent(OpenAIChatModel("qwen2.5:14b", provider=provider, profile=profile), output_type=EntityExtractionResult, system_prompt=ENTITY_SYSTEM, retries=1)

    async def run():
        return await agent.run(task_prompt(f"Extract entities from this requirement:\n\n{REQUIREMENT}"))
    return asyncio.run(run())


def test_haystack():
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret
    from prompts import ENTITY_SYSTEM, task_prompt

    gen = OpenAIChatGenerator(api_key=Secret.from_token("ollama"), model="qwen2.5:14b", api_base_url="http://172.31.128.1:11434/v1", generation_kwargs={"response_format": {"type": "json_object"}})
    messages = [ChatMessage.from_system(ENTITY_SYSTEM), ChatMessage.from_user(task_prompt(f"Extract entities:\n\n{REQUIREMENT}"))]
    return gen.run(messages=messages)


def test_langgraph():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from prompts import ENTITY_SYSTEM, task_prompt

    llm = ChatOpenAI(model="qwen2.5:14b", base_url="http://172.31.128.1:11434/v1", api_key="ollama")
    messages = [SystemMessage(content=ENTITY_SYSTEM), HumanMessage(content=task_prompt(f"Extract entities:\n\n{REQUIREMENT}"))]
    return llm.invoke(messages, response_format={"type": "json_object"})


def test_crewai():
    from crewai import Agent, Task, Crew, LLM
    from prompts import ENTITY_SYSTEM

    llm = LLM(model="ollama/qwen2.5:14b", base_url="http://172.31.128.1:11434")
    agent = Agent(role="Entity Extractor", goal="Extract entities", backstory=ENTITY_SYSTEM, llm=llm, verbose=False)
    task = Task(description=f"Extract entities from this requirement:\n\n{REQUIREMENT}", expected_output="JSON", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return crew.kickoff()


frameworks = {
    "pydantic-ai": test_pydantic_ai,
    "haystack": test_haystack,
    "langgraph": test_langgraph,
    "crewai": test_crewai,
}

for name, fn in frameworks.items():
    captured_requests.clear()
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    start = time.time()
    try:
        fn()
    except Exception as e:
        print(f"  Error: {e}")
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Requests captured: {len(captured_requests)}")

    for i, req in enumerate(captured_requests):
        print(f"\n  Request {i+1}:")
        print(f"    URL: {req['url']}")
        print(f"    Model: {req['model']}")
        print(f"    response_format: {req['response_format']}")
        print(f"    temperature: {req['temperature']}")
        print(f"    tools: {'yes' if req['tools'] else 'no'}")
        print(f"    extra_keys: {req['extra_keys']}")
        print(f"    Messages ({len(req['messages'])}):")
        for msg in req["messages"]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                preview = content[:150].replace("\n", "\\n")
                print(f"      [{role}] ({len(content)} chars) {preview}...")
            else:
                print(f"      [{role}] (non-string content)")
