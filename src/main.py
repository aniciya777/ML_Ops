import os
from typing import TypedDict

from fastapi import FastAPI
import uvicorn

app = FastAPI()


class SumResult(TypedDict):
    """
    Response model for the sum of two numbers
    """
    result: int


@app.get("/sum/{number1}/{number2}", response_model=SumResult)
def sum_numbers(number1: int, number2: int) -> SumResult:
    """
    Returns the sum of two integers provided as path parameters.
    :param number1: First integer
    :param number2: Second integer
    :return: A dictionary with key "result" and the sum as its value
    """
    result: int = number1 + number2
    return {"result": result}


def main() -> None:
    # Read port from the PORT environment variable, default to 8000
    port_env: str = os.getenv("PORT", "8000")
    try:
        port: int = int(port_env)
    except ValueError:
        raise ValueError(f"Invalid port value: {port_env}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
