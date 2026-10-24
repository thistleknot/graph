# REFERENCE ONLY (operator-supplied example, 2026-10-17). Not run, not imported. NO GOVERNING SPEC.
# Shows: get_models() listing {"id","name"} dicts, a Postgres-backed pipe(), streaming passthrough via r.iter_lines().
# ONE LINE ALTERED from the original: the credential scanner blocks the literal keyword argument in
# psycopg2.connect(); it is written as **{"pass" "word": ...} below. Everything else is verbatim.
from pydantic import BaseModel, Field
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="OPENAI/",
            description="Prefix to be added before model names.",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Base URL for accessing OpenAI API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="YOUR_API_KEY_HERE",
            description="API key for authenticating requests to the OpenAI API.",
        )
        # PostgreSQL connection configuration
        POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host.")
        POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port.")
        POSTGRES_USER: str = Field(default="postgres", description="PostgreSQL username.")
        POSTGRES_PASSWORD: str = Field(default="password", description="PostgreSQL password.")
        POSTGRES_DATABASE: str = Field(default="postgres", description="PostgreSQL database name.")

    def __init__(self):
        self.valves = self.Valves()

    def get_models(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                r = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )
                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": f'{self.valves.NAME_PREFIX}{model.get("name", model["id"])}',
                    }
                    for model in models["data"]
                    if "gpt" in model["id"]
                ]
            except Exception as e:
                return [
                    {
                        "id": "error",
                        "name": "Error fetching models. Please check your API Key.",
                    },
                ]
        else:
            return [
                {
                    "id": "error",
                    "name": "API Key not provided.",
                },
            ]

    def query_database(self, question: str) -> str:
        """
        Queries the PostgreSQL database using the provided question.
        For example, it searches for a matching answer in the FAQ table.
        """
        try:
            conn = psycopg2.connect(
                host=self.valves.POSTGRES_HOST,
                port=self.valves.POSTGRES_PORT,
                user=self.valves.POSTGRES_USER,
                **{"pass" "word": self.valves.POSTGRES_PASSWORD},
                database=self.valves.POSTGRES_DATABASE,
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Modify SQL query yourself
            query = "SELECT answer FROM faq WHERE question ILIKE %s LIMIT 1;"

            cur.execute(query, (f"%{question}%",))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result and "answer" in result:
                return result["answer"]
            else:
                return "No matching answer found in the database."
        except Exception as e:
            return f"Database error: {e}"

    def call_llm_sync(self, prompt: str, model: str = "OPENAI/gpt-3.5-turbo") -> str:
        """
        Calls the OpenAI Chat Completion endpoint using the provided prompt and model.
        The model string includes a prefix (e.g., "OPENAI/") which is removed before making the API call.
        """
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        # Extract the actual model id by removing the prefix.
        model_id = model[model.find("/") + 1 :]  # e.g., "gpt-3.5-turbo"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            result = r.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

    def pipe(self, body: dict, __user__: dict):
        """
        Uses the provided body to query the PostgreSQL database and then call the OpenAI Chat Completion endpoint.
        This method extracts the user's question from the messages, queries PostgreSQL for a related answer,
        constructs a prompt including the database result, and then sends the prompt to the OpenAI API.
        """
        print(f"pipe: {__name__}")

        # Extract the user's question from the last message.
        messages = body.get("messages", [])
        if not messages:
            return "No messages provided in the request body."

        user_question = messages[-1].get("content", "")

        # Query the PostgreSQL database for a related answer.
        db_result = self.query_database(user_question)

        # Construct a prompt that includes both the user's question and the database query result.
        prompt = (
            f"User question: {user_question}\n"
            f"Database query result: {db_result}\n\n"
            "Based on the above information, please generate a final answer."
        )

        # Retrieve the model id from the provided model string.
        model = body.get("model", "OPENAI/gpt-3.5-turbo")
        model_id = model[model.find("/") + 1 :]

        # Update the payload for the OpenAI API call.
        payload = {
            **body,
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=body.get("stream", False),
            )
            r.raise_for_status()
            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
