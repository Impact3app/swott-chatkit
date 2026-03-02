import os
import uuid
import base64
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import Store, AttachmentStore, NotFoundError
from chatkit.types import (
    ThreadMetadata, Page, Thread, ActiveStatus,
    AssistantMessageItem, AssistantMessageContent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartAdded,
    ThreadItemAddedEvent, ThreadItemDoneEvent,
    UserMessageItem, FileAttachment,
)

SWOTT_BACKEND_URL = os.environ.get("SWOTT_BACKEND_URL", "http://localhost:3000")


# ─── In-Memory Thread Store ───────────────────────────────────────────────────

class InMemoryStore(Store):
    def __init__(self):
        self._threads = {}
        self._items = {}
        self._attachments = {}  # partagé avec InMemoryAttachmentStore

    async def load_thread(self, thread_id, context):
        if thread_id not in self._threads:
            self._threads[thread_id] = Thread(
                id=thread_id,
                title=None,
                created_at=datetime.now(timezone.utc),
                status=ActiveStatus(),
                items=Page(data=[], has_more=False),
                metadata={}
            )
        return self._threads[thread_id]

    async def save_thread(self, thread, context):
        self._threads[thread.id] = thread

    async def load_thread_items(self, thread_id, after, limit, order, context):
        items = list(self._items.get(thread_id, {}).values())
        return Page(data=items, has_more=False)

    async def add_thread_item(self, thread_id, item, context):
        self._items.setdefault(thread_id, {})[item.id] = item

    async def save_item(self, thread_id, item, context):
        self._items.setdefault(thread_id, {})[item.id] = item

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
        return Page(data=list(self._threads.values()), has_more=False)

    def generate_item_id(self, item_type, thread, context):
        return f"{item_type}_{uuid.uuid4().hex[:12]}"

    async def save_attachment(self, attachment, context):
        self._attachments[attachment.id] = attachment

    async def load_attachment(self, attachment_id, context):
        if attachment_id not in self._attachments:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return self._attachments[attachment_id]

    async def delete_attachment(self, attachment_id, context):
        self._attachments.pop(attachment_id, None)


# ─── In-Memory Attachment Store ───────────────────────────────────────────────

class InMemoryAttachmentStore(AttachmentStore):
    def __init__(self, store: "InMemoryStore"):
        self._store = store
        self._bytes = {}
        self._texts = {}

    @property
    def _attachments(self):
        return self._store._attachments

    async def create_attachment(self, input, context):
        att_id = f"att_{uuid.uuid4().hex[:12]}"
        content = getattr(input, 'content', None)
        if content:
            raw = base64.b64decode(content) if isinstance(content, str) else content
            self._bytes[att_id] = raw
            name = getattr(input, 'name', 'fichier')
            text = extract_text(name, raw)
            self._texts[att_id] = text
            print(f">>> Fichier recu: {name} ({len(raw)} bytes, {len(text)} chars)")

        att = FileAttachment(
            id=att_id,
            thread_id=getattr(input, 'thread_id', ''),
            created_at=datetime.now(timezone.utc),
            name=getattr(input, 'name', 'fichier'),
            mime_type=getattr(input, 'mime_type', 'application/octet-stream'),
            size=getattr(input, 'size', len(self._bytes.get(att_id, b''))),
        )
        self._attachments[att_id] = att
        return att

    async def delete_attachment(self, attachment_id, context):
        self._store._attachments.pop(attachment_id, None)
        self._bytes.pop(attachment_id, None)
        self._texts.pop(attachment_id, None)

    def get_text(self, attachment_id):
        return self._texts.get(attachment_id, '')

    def get_attachment(self, attachment_id):
        return self._attachments.get(attachment_id)


# ─── Extraction de texte ──────────────────────────────────────────────────────

def extract_text(filename, content_bytes):
    ext = filename.lower().rsplit('.', 1)[-1]
    try:
        if ext == 'pdf':
            import io, PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
            return "\n".join(p.extract_text() or '' for p in reader.pages).strip()
        elif ext in ('docx', 'doc'):
            import io, docx
            doc = docx.Document(io.BytesIO(content_bytes))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        elif ext in ('xlsx', 'xls'):
            import io, openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content_bytes), data_only=True)
            lines = []
            for sheet in wb.worksheets:
                lines.append(f"[Feuille: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    row_str = "\t".join(str(c) if c is not None else '' for c in row)
                    if row_str.strip():
                        lines.append(row_str)
            return "\n".join(lines).strip()
        elif ext == 'txt':
            return content_bytes.decode('utf-8', errors='ignore')
        else:
            return f"[Fichier: {filename} — format non supporte]"
    except Exception as e:
        return f"[Erreur extraction {filename}: {e}]"


# ─── ChatKit Server ───────────────────────────────────────────────────────────

class SwottChatKitServer(ChatKitServer):
    def __init__(self, data_store, attachment_store):
        super().__init__(data_store, attachment_store=attachment_store)
        self.att_store = attachment_store

    async def respond(self, thread, input, context):
        message_text = ""
        attachment_ids = []

        if input and hasattr(input, "content"):
            content = input.content
            if isinstance(content, str):
                message_text = content
            elif isinstance(content, list):
                for part in content:
                    if hasattr(part, 'text'):
                        message_text += part.text
                    elif hasattr(part, 'attachment_id'):
                        attachment_ids.append(part.attachment_id)

        if input and hasattr(input, 'attachments') and input.attachments:
            for att_ref in input.attachments:
                att_id = getattr(att_ref, 'id', None) or getattr(att_ref, 'attachment_id', None)
                if att_id:
                    attachment_ids.append(att_id)

        if attachment_ids:
            files_context = "\n\n---DOCUMENTS FOURNIS PAR L'UTILISATEUR---\n"
            for att_id in attachment_ids:
                att = self.att_store.get_attachment(att_id)
                text = self.att_store.get_text(att_id)
                name = att.name if att else att_id
                files_context += f"\n[Fichier: {name}]\n{text}\n"
            files_context += "---FIN DES DOCUMENTS---\n"
            message_text = message_text + files_context
            print(f">>> {len(attachment_ids)} fichier(s) injecte(s)")

        print(f">>> Message: {message_text[:120]}")

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{SWOTT_BACKEND_URL}/chat",
                json={"session_id": thread.id, "message": message_text or " "}
            )
            data = response.json()
            response_text = data.get("response", "")

        print(f">>> Reponse: {response_text[:100]}")

        item_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        assistant_item = AssistantMessageItem(
            id=item_id,
            thread_id=thread.id,
            created_at=now,
            content=[AssistantMessageContent(type="output_text", text="", annotations=[])]
        )
        yield ThreadItemAddedEvent(item=assistant_item)

        yield AssistantMessageContentPartAdded(
            content_index=0,
            content={"type": "output_text", "text": ""}
        )

        for i, word in enumerate(response_text.split(" ")):
            text = word if i == 0 else " " + word
            yield AssistantMessageContentPartTextDelta(content_index=0, delta=text)
            await asyncio.sleep(0.03)

        yield AssistantMessageContentPartDone(
            content_index=0,
            content={"type": "output_text", "text": response_text}
        )

        yield ThreadItemDoneEvent(item=AssistantMessageItem(
            id=item_id,
            thread_id=thread.id,
            created_at=now,
            content=[AssistantMessageContent(type="output_text", text=response_text, annotations=[])]
        ))


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = InMemoryStore()
att_store = InMemoryAttachmentStore(data_store)
server = SwottChatKitServer(data_store, att_store)


@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    result = await server.process(await request.body(), {})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")


@app.get("/")
async def healthcheck():
    return {"status": "ok", "service": "swott-chatkit"}
