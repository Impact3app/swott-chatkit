import os
import uuid
import base64
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
    UserMessageItem, Attachment
)
from typing import Any, AsyncIterator

SWOTT_BACKEND_URL = os.environ.get("SWOTT_BACKEND_URL", "http://localhost:3000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class InMemoryStore(Store):
    def __init__(self):
        self._threads = {}
        self._items = {}
        self._attachments = {}
        self._user_files = {}  # session_id -> list of extracted texts

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

    def add_user_file(self, session_id: str, filename: str, content: str):
        if session_id not in self._user_files:
            self._user_files[session_id] = []
        self._user_files[session_id].append({
            "filename": filename,
            "content": content
        })
        print(f">>> Fichier ajouté pour session {session_id}: {filename} ({len(content)} chars)")

    def get_user_files(self, session_id: str):
        return self._user_files.get(session_id, [])

    def clear_user_files(self, session_id: str):
        self._user_files.pop(session_id, None)


def extract_text_from_file(filename: str, content_bytes: bytes) -> str:
    """Extrait le texte d'un fichier PDF, Word ou Excel"""
    ext = filename.lower().split('.')[-1]

    try:
        if ext == 'pdf':
            import io
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            except ImportError:
                return f"[PDF reçu: {filename} - installez PyPDF2 pour l'extraction]"

        elif ext in ['docx', 'doc']:
            import io
            try:
                import docx
                doc = docx.Document(io.BytesIO(content_bytes))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text.strip()
            except ImportError:
                return f"[Word reçu: {filename} - installez python-docx pour l'extraction]"

        elif ext in ['xlsx', 'xls']:
            import io
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(content_bytes))
                text = ""
                for sheet in wb.worksheets:
                    text += f"\n[Feuille: {sheet.title}]\n"
                    for row in sheet.iter_rows(values_only=True):
                        row_text = "\t".join([str(c) if c is not None else "" for c in row])
                        if row_text.strip():
                            text += row_text + "\n"
                return text.strip()
            except ImportError:
                return f"[Excel reçu: {filename} - installez openpyxl pour l'extraction]"

        elif ext == 'txt':
            return content_bytes.decode('utf-8', errors='ignore')

        else:
            return f"[Fichier reçu: {filename} - format non supporté pour extraction]"

    except Exception as e:
        return f"[Erreur extraction {filename}: {str(e)}]"


class SwottChatKitServer(ChatKitServer):
    def __init__(self, data_store: InMemoryStore):
        super().__init__(data_store)
        self.data_store = data_store

    async def create_attachment(self, params, context):
        """Intercepte l'upload de fichier et extrait le texte"""
        attachment = await super().create_attachment(params, context)

        try:
            filename = getattr(params, 'filename', None) or getattr(params, 'name', 'fichier')
            content_bytes = getattr(params, 'content', None)

            if content_bytes:
                if isinstance(content_bytes, str):
                    content_bytes = base64.b64decode(content_bytes)

                text = extract_text_from_file(filename, content_bytes)
                session_id = context.get('session_id', 'unknown') if isinstance(context, dict) else 'unknown'
                self.data_store.add_user_file(session_id, filename, text)

        except Exception as e:
            print(f">>> Erreur lors de l'extraction du fichier: {e}")

        return attachment

    async def respond(self, thread: ThreadMetadata, input: UserMessageItem | None, context: Any) -> AsyncIterator[Any]:
        message_text = ""
        if input and hasattr(input, "content"):
            if isinstance(input.content, str):
                message_text = input.content
            elif isinstance(input.content, list):
                for part in input.content:
                    if hasattr(part, "text"):
                        message_text += part.text

        # Récupérer les fichiers uploadés pour cette session
        user_files = self.data_store.get_user_files(thread.id)
        if user_files:
            files_context = "\n\n---DOCUMENTS FOURNIS PAR L'UTILISATEUR---\n"
            for f in user_files:
                files_context += f"\n[Fichier: {f['filename']}]\n{f['content']}\n"
            files_context += "---FIN DES DOCUMENTS---\n"
            message_text = message_text + files_context
            print(f">>> {len(user_files)} fichier(s) injecté(s) dans le message")

        print(f">>> Message recu: {message_text[:100]}")

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{SWOTT_BACKEND_URL}/chat",
                json={"session_id": thread.id, "message": message_text}
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

        words = response_text.split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            yield AssistantMessageContentPartTextDelta(content_index=0, delta=text)

        yield AssistantMessageContentPartDone(
            content_index=0,
            content={"type": "output_text", "text": response_text}
        )

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
