
import os
import re
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from e2b_code_interpreter import AsyncSandbox

load_dotenv()

app = FastAPI(title="IITM AI Code Interpreter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ai = AsyncOpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://aipipe.org/openrouter/v1"
    ),
)

MODEL = os.getenv("AI_MODEL", "openai/gpt-4.1-nano")


class CodeRequest(BaseModel):
    code: str = Field(max_length=20000)


class CodeResponse(BaseModel):
    error: list[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


async def execute_python_code(code: str) -> dict:
    """
    Execute code in an external E2B sandbox.
    Never run untrusted code in the API process.
    """
    sandbox = await AsyncSandbox.create(
        timeout=60
    )

    try:
        # Create an actual Python file so tracebacks
        # refer to the submitted source line numbers.
        await sandbox.files.write(
            "/tmp/user_code.py", code
        )

        execution = await sandbox.commands.run(
            "python -u /tmp/user_code.py",
            timeout=10
        )

        stdout = execution.stdout
        stderr = execution.stderr

        success = execution.exit_code == 0

        # Normal Python tracebacks are written to stderr.
        # Return traceback without altering its contents.
        if success:
            output = stdout + stderr
        else:
            output = stdout + stderr

        return {
            "success": success,
            "output": output
        }

    finally:
        await sandbox.kill()


def traceback_lines(code: str, tb: str) -> list[int]:
    """Fallback using Python traceback locations."""
    matches = re.findall(
        r'File "/tmp/user_code.py", line (\d+)',
        tb
    )

    maximum = len(code.splitlines())

    return sorted({
        int(n) for n in matches
        if 1 <= int(n) <= maximum
    })


async def analyze_error_with_ai(
    code: str,
    tb: str
) -> list[int]:

    numbered_code = "\n".join(
        f"{i}: {line}"
        for i, line in enumerate(
            code.splitlines(), start=1
        )
    )

    response = await ai.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze Python errors. "
                    "Return JSON containing error_lines, "
                    "an array of 1-based source line "
                    "numbers. Use the traceback as "
                    "primary evidence. Do not include "
                    "library or framework lines."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CODE:\n{numbered_code}\n\n"
                    f"TRACEBACK:\n{tb}\n\n"
                    "Return JSON like "
                    '{"error_lines":[3]}'
                )
            }
        ]
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty AI response")

    parsed = ErrorAnalysis.model_validate_json(content)

    maximum = len(code.splitlines())

    return sorted({
        n for n in parsed.error_lines
        if 1 <= n <= maximum
    })


@app.post(
    "/code-interpreter",
    response_model=CodeResponse
)
async def code_interpreter(request: CodeRequest):

    try:
        execution = await execute_python_code(
            request.code
        )

    except Exception as exc:
        # Infrastructure errors are not user-code errors.
        raise HTTPException(
            status_code=503,
            detail="Code execution sandbox unavailable"
        ) from exc

    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"]
        )

    # AI runs only when execution fails.
    try:
        lines = await asyncio.wait_for(
            analyze_error_with_ai(
                request.code,
                execution["output"]
            ),
            timeout=20
        )

        if not lines:
            lines = traceback_lines(
                request.code,
                execution["output"]
            )

    except Exception:
        lines = traceback_lines(
            request.code,
            execution["output"]
        )

    return CodeResponse(
        error=lines,
        result=execution["output"]
    )


@app.get("/")
async def health():
    return {"status": "ok"}
