import os
import uuid
import httpx
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import Store, NotFoundError
from chatkit.types import (
    ThreadMetadata, Page, Thread, ActiveStatus,
    AssistantMessageItem, AssistantMessageContent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartAdded,
    ThreadItemAddedEvent, ThreadItemDoneEvent,
    UserMessageItem
)
from typing import Any, AsyncIterator

SWOTT_BACKEND_URL = os.environ.get("SWOTT_BACKEND_URL", "http://localhost:3000")


class InMemoryStore(Store):
    def __init__(self):
        self._threads = {}
        self._items = {}
        self._attachments = {}

    async def load_thread(self, thread_id, context):
        if thread_id not in self._threads:
            thread = Thread(
                id=thread_id,
                title=None,
                created_at=datetime.now(timezone.utc),
                status=ActiveStatus(),
                metadata={}
            )
            self._threads[thread_id] = thread
        return self._threads[thread_id]

    async def save_thread(self, thread, context):
        self._threads[thread.id] = thread

    async def load_thread_items(self, thread_id, after, limit, order, context):
        items = list(self._items.get(thread_id, {}).values())
        return Page(data=items, has_more=False)

    async def add_thread_item(self, thread_id, item, context):
        if thread_id not in self._items:
            self._items[thread_id] = {}
        self._items[thread_id][item.id] = item

    async def save_item(self, thread_id, item, context):
        if thread_id not in self._items:
            self._items[thread_id] = {}
        self._items[thread_id][item.id] = item

    async def load_item(self, thread_id, item_id, context):
        if thread_id not in self._items or item_id not in self._items[thread_id]:
            raise NotFoundError(f"Item {item_id} not found")
        return self._items[thread_id][item_id]

    async def delete_thread(self, thread_id, context):
        self._threads.pop(thread_id, None)
        self._items.pop(thread_id, None)

    async def delete_thread_item(self, thread_id, item_id, context):
        if thread_id in self._items:
            self._items[thread_id].pop(item_id, None)

    async def load_threads(self, limit, after, order, context):
        threads = list(self._threads.values())
        return Page(data=threads, has_more=False)

    async def save_attachment(self, attachment, context):
        self._attachments[attachment.id] = attachment

    async def load_attachment(self, attachment_id, context):
        if attachment_id not in self._attachments:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return self._attachments[attachment_id]

    async def delete_attachment(self, attachment_id, context):
        self._attachments.pop(attachment_id, None)


class SwottChatKitServer(ChatKitServer):
    def __init__(self, data_store):
        super().__init__(data_store)

    async def respond(self, thread: ThreadMetadata, input: UserMessageItem | None, context: Any) -> AsyncIterator[Any]:
        message_text = ""
        if input and hasattr(input, "content"):
            if isinstance(input.content, str):
                message_text = input.content
            elif isinstance(input.content, list):
                for part in input.content:
                    if hasattr(part, "text"):
                        message_text += part.text

        print(f">>> Message recu: {message_text}")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{SWOTT_BACKEND_URL}/chat",
                json={"session_id": thread.id, "message": message_text}
            )
            data = response.json()
            response_text = data.get("response", "")

        print(f">>> Reponse: {response_text[:100]}")

        item_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Créer l'item assistant
        assistant_item = AssistantMessageItem(
            id=item_id,
            thread_id=thread.id,
            created_at=now,
            content=[AssistantMessageContent(type="output_text", text="", annotations=[])]
        )

        # Signaler au framework que l'item est ajouté
        yield ThreadItemAddedEvent(item=assistant_item)

        # Streamer le texte mot par mot
        yield AssistantMessageContentPartAdded(
            content_index=0,
            content={"type": "output_text", "text": ""}
        )

        words = response_text.split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            yield AssistantMessageContentPartTextDelta(content_index=0, delta=text)

        yield AssistantMessageContentPartDone(
            content_index=0,
            content={"type": "output_text", "text": response_text}
        )

        # Item final avec le texte complet
        final_item = AssistantMessageItem(
            id=item_id,
            thread_id=thread.id,
            created_at=now,
            content=[AssistantMessageContent(type="output_text", text=response_text, annotations=[])]
        )
        yield ThreadItemDoneEvent(item=final_item)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
data_store = InMemoryStore()
server = SwottChatKitServer(data_store)


@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    result = await server.process(await request.body(), {})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")


@app.get("/")
async def healthcheck():
    return {"status": "ok", "service": "swott-chatkit"}
