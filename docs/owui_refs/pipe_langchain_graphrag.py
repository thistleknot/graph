# REFERENCE ONLY (operator-supplied example, 2026-10-17). Not run, not imported. NO GOVERNING SPEC.
# Shows: Pipe class, Valves(BaseModel), async pipe(body, __user__, __event_emitter__), throttled status emits.
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import os
import time
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager
from llama_index.core import SimpleDirectoryReader

class Pipe:
    class Valves(BaseModel):
        openai_api_key: str = Field(default="...")
        embedding_model_name: str = Field(default="sentence-transformers/all-mpnet-base-v2")
        emit_interval: float = Field(default=2.0)
        enable_status_indicator: bool = Field(default=True)

    def __init__(self):
        self.type = "pipe"
        self.id = "langchain_pipe"
        self.name = "LangChain GraphRAG Pipe"
        self.valves = self.Valves()
        self.last_emit_time = 0

        self.llm = ChatOpenAI(api_key=self.valves.openai_api_key, model=self.valves.openai_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.valves.embedding_model_name)

        self.documents = SimpleDirectoryReader("./data").load_data()
        self.vector_store = Chroma(embedding_function=self.embeddings)
        self.vector_store.add_documents(documents=self.documents)

        self.traversal_retriever = GraphRetriever(
            store=self.vector_store,
            edges=[
                ("Has symptom", "Has symptom"),
                ("Increases risk", "Increases risk"),
                ("Treated with", "Treated with"),
                ("May require", "May require"),
                ("Can lead to", "Can lead to"),
                ("Studied by", "Studied by"),
                ("Has outcome", "Has outcome"),
            ],
            strategy=Eager(k=5, start_k=5, max_depth=2),
        )

        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the context provided.
            Return your answer as well as the papers cited and their authors.

            Context: {context}

            Question: {question}"""
        )

        self.chain = (
            {"context": self.traversal_retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(f"text: {doc.page_content} metadata: {doc.metadata}" for doc in docs)

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__({
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            })
            self.last_emit_time = current_time

    async def pipe(self, body: dict,
                   __user__: Optional[dict] = None,
                   __event_emitter__: Callable[[dict], Awaitable[None]] = None,
                   __event_call__: Callable[[dict], Awaitable[dict]] = None,
                   ) -> Optional[dict]:

        await self.emit_status(__event_emitter__, "info", "Initiating GraphRAG Chain", False)

        messages = body.get("messages", [])
        if messages:
            question = messages[-1]["content"]
            try:
                await self.emit_status(__event_emitter__, "info", "Executing Retrieval and Generation", False)
                response = self.chain.invoke(question)
                body["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                await self.emit_status(__event_emitter__, "error", f"Error during execution: {str(e)}", True)
                return {"error": str(e)}
        else:
            await self.emit_status(__event_emitter__, "error", "No messages found in request body", True)
            body["messages"].append({"role": "assistant", "content": "No messages found in the request body"})

        await self.emit_status(__event_emitter__, "info", "Complete", True)
        return body
