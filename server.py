"""
server.py — swott-chatkit
Architecture : openai-chatkit + openai-agents SDK (Python natif)
Persistence : Supabase (PostgreSQL)
"""

import os
import uuid
import base64
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import Store, AttachmentStore, NotFoundError
from chatkit.types import (
    Page, Thread, ActiveStatus,
    AssistantMessageItem, AssistantMessageContent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartAdded,
    ThreadItemAddedEvent, ThreadItemDoneEvent,
    FileAttachment,
)
from agents import Agent, ModelSettings, Runner, RunConfig, trace
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
from supabase import create_client, Client

import prompts as P

# ── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WORKFLOW_ID = "wf_696b4c50579481908a889f44236f130108bc443970089c82"


# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def db_save_thread(thread_id: str, user_id: str, client_name: str = ""):
    try:
        supabase.table("threads").upsert({
            "thread_id": thread_id,
            "user_id": user_id,
            "client_name": client_name,
        }).execute()
    except Exception as e:
        print(f"[Supabase] db_save_thread error: {e}")

def db_append_message(thread_id: str, role: str, content: str, tokens_used: int = 0):
    try:
        msg_id = f"{role}_{uuid.uuid4().hex[:12]}"
        supabase.table("messages").insert({
            "id": msg_id,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "tokens_used": tokens_used,
        }).execute()
    except Exception as e:
        print(f"[Supabase] db_append_message error: {e}")

def db_get_history(thread_id: str) -> list:
    try:
        res = supabase.table("messages") \
            .select("role, content") \
            .eq("thread_id", thread_id) \
            .order("created_at") \
            .execute()
        return [{"role": r["role"], "content": r["content"]} for r in res.data]
    except Exception as e:
        print(f"[Supabase] db_get_history error: {e}")
        return []

def db_get_threads_for_user(user_id: str) -> list:
    try:
        res = supabase.table("threads") \
            .select("thread_id, title, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        threads = []
        for t in res.data:
            # Récupérer le dernier message de ce thread
            last = supabase.table("messages") \
                .select("content") \
                .eq("thread_id", t["thread_id"]) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            last_msg = last.data[0]["content"] if last.data else None
            threads.append({
                "thread_id": t["thread_id"],
                "title": t.get("title"),
                "created_at": t["created_at"],
                "last_msg": last_msg,
            })
        return threads
    except Exception as e:
        print(f"[Supabase] db_get_threads_for_user error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

class ClassifierSchema(BaseModel):
    category: str
    message: str

class JacquesSchema(BaseModel):
    smr_axis: str
    message: str

class HenrySchema(BaseModel):
    sml_axis: str
    message: str

class SherlockFastSchema(BaseModel):
    objet: str
    zone: str
    contraintes: str
    urgence: str
    launch_deep: bool
    confidence: str

class FreyaFastSchema(BaseModel):
    company: str
    solution: str
    geographies: str
    objectives_focus: str
    launch_deep: bool
    confidence: str


def _build_agents():
    from agents import WebSearchTool, FileSearchTool, CodeInterpreterTool
    web = WebSearchTool()
    ci  = CodeInterpreterTool(tool_config={"type": "code_interpreter", "container": {"type": "auto"}})

    def fs(vs_id):
        return FileSearchTool(vector_store_ids=[vs_id])

    classifier = Agent(
        name="Agent_IfElse_JSON",
        instructions=P.AGENT_CLASSIFIER,
        model="gpt-4.1",
        output_type=ClassifierSchema,
        model_settings=ModelSettings(temperature=0.1, max_tokens=300, store=True),
    )

    cortex_routage = Agent(
        name="CorteX_Routage",
        instructions=P.AGENT_CORTEX_ROUTAGE,
        model="gpt-5-nano",
        model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low")),
    )

    agents = {
        "cortex_routage": cortex_routage,
        "classifier":     classifier,
        "marcel":    Agent(name="Marcel Politique Achats",   instructions=P.AGENT_MARCEL,    model="gpt-5.2", tools=[fs("vs_696b5cbccd5c8191bf05ba182288941f"), web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "leonard":   Agent(name="Leonard Diag orga",         instructions=P.AGENT_LEONARD,   model="gpt-5.2", tools=[fs("vs_696b5f7c2a1c8191af6d9faf45de8cf5"), web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "hector":    Agent(name="Hector négociation",        instructions=P.AGENT_HECTOR,    model="gpt-5.2", tools=[web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "gustave":   Agent(name="Gustave Data expert",       instructions=P.AGENT_GUSTAVE,   model="gpt-5.2", tools=[web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "eustache":  Agent(name="Eustache.ia",               instructions=P.AGENT_EUSTACHE,  model="gpt-5",   tools=[fs("vs_696b62901dd48191832a5f82c6fca59f"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "marguerite":Agent(name="Marguerite.ia",             instructions=P.AGENT_MARGUERITE,model="gpt-5",   tools=[fs("vs_696b62bf478c8191ac2db36a2167624e"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "luther":    Agent(name="Luther.ia",                 instructions=P.AGENT_LUTHER,    model="gpt-5",   tools=[fs("vs_696b62ea15a881918bdc13cfefc2e3da"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "chan":       Agent(name="Chan.ia",                   instructions=P.AGENT_CHAN,       model="gpt-5",   tools=[fs("vs_696b6377fa4c8191ad34646de598f0ec"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "savannah":  Agent(name="Savannah.ia",               instructions=P.AGENT_SAVANNAH,  model="gpt-5",   model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "albert":    Agent(name="Albert.ia",                 instructions=P.AGENT_ALBERT,    model="gpt-5",   model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "catherine": Agent(name="Catherine.ia",              instructions=P.AGENT_CATHERINE, model="gpt-5",   model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "michele":   Agent(name="Michèle.ia",                instructions=P.AGENT_MICHELE,   model="gpt-5",   tools=[fs("vs_696cb91a5f4881919af132e343839bd3")], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "achille":   Agent(name="Achille TCO",               instructions=P.AGENT_ACHILLE,   model="gpt-5",   tools=[web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "hypathie":  Agent(name="Hypathie Juriste",          instructions=P.AGENT_HYPATHIE,  model="gpt-5.2-pro", tools=[web], model_settings=ModelSettings(store=True)),
        "sherlock_cadrage": Agent(name="Sherlock Sourcing Cadrage", instructions=P.AGENT_SHERLOCK_CADRAGE, model="gpt-5.2", model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low", summary="auto"))),
        "sherlock_fast":    Agent(name="Sherlock_fast.json.ai",     instructions=P.AGENT_SHERLOCK_FAST_JSON, model="gpt-4.1", output_type=SherlockFastSchema, model_settings=ModelSettings(temperature=0.21, max_tokens=800, store=True)),
        "sherlock_deep":    Agent(name="Sherlock_Deep",             instructions=P.AGENT_SHERLOCK_DEEP, model="gpt-5.2-pro", tools=[web], model_settings=ModelSettings(store=True, reasoning=Reasoning(summary="auto"))),
        "hercule":   Agent(name="Hercule comparaison offres",instructions=P.AGENT_HERCULE,   model="gpt-5.2", tools=[web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="xhigh", summary="auto"))),
        "clint":     Agent(name="Clint.ai",                  instructions=P.AGENT_CLINT,     model="gpt-5.2", tools=[fs("vs_69737b5136288191b4631b0c59ddfee4"), web, ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "barack":    Agent(name="Barack.ai",                 instructions=P.AGENT_BARACK,    model="gpt-5.2", tools=[fs("vs_69788788dd048191973251dfa884c9c7")], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low"))),
        "isaac":     Agent(name="Isaac plan action",         instructions=P.AGENT_ISAAC,     model="gpt-5.2", tools=[fs("vs_6981b69702e88191acf84e6584eae9e4"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "mazarin":   Agent(name="Mazarin Diplomate",         instructions=P.AGENT_MAZARIN,   model="gpt-5.2", model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low", summary="auto"))),
        "sebus":     Agent(name="Sebus Excel expert",        instructions=P.AGENT_SEBUS,     model="gpt-5.2", tools=[ci], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "franklin":  Agent(name="Franklin CR",               instructions=P.AGENT_FRANKLIN,  model="gpt-5.2", model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium", summary="auto"))),
        "augustine": Agent(name="Augustine CDC",             instructions=P.AGENT_AUGUSTINE, model="gpt-5.2", tools=[fs("vs_6981fb58587c8191a97c5a9a6df28d08"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "freya_cadrage": Agent(name="Freya Benchmark cadrage", instructions=P.AGENT_FREYA_CADRAGE, model="gpt-5.2", tools=[web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium", summary="auto"))),
        "freya_fast":    Agent(name="Freya_json",              instructions=P.AGENT_FREYA_FAST_JSON, model="gpt-4.1", output_type=FreyaFastSchema, model_settings=ModelSettings(temperature=1, max_tokens=2048, store=True)),
        "freya_deep":    Agent(name="Freya Deep",              instructions=P.AGENT_FREYA_DEEP, model="gpt-5.2", tools=[web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="xhigh", summary="auto"))),
        "hilda":     Agent(name="Hilda RFAR",                instructions=P.AGENT_HILDA,     model="gpt-5.2", tools=[fs("vs_699710f8b48881918c6f77e0b516c2a7"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "hermes":    Agent(name="Hermes Bilan carbone",      instructions=P.AGENT_HERMES,    model="gpt-5.2", tools=[web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low", summary="auto"))),
        "iris":      Agent(name="Iris Processus Achats",     instructions=P.AGENT_IRIS,      model="gpt-5.2", tools=[web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "ariane":    Agent(name="Ariane Assistante RH",      instructions=P.AGENT_ARIANE,    model="gpt-5.2", model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low", summary="auto"))),
        "cortex_core": Agent(name="Cortex_core",             instructions=P.AGENT_CORTEX_CORE, model="gpt-5.2", tools=[fs("vs_6981b69702e88191acf84e6584eae9e4"), web], model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="high", summary="auto"))),
        "jacques_json": Agent(name="Jacques Strat portefeuilles", instructions=P.AGENT_JACQUES_JSON, model="gpt-5.2", output_type=JacquesSchema, model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low"))),
        "henry_json":   Agent(name="Henry Leviers Achats",   instructions=P.AGENT_HENRY_JSON, model="gpt-5-nano", output_type=HenrySchema, model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low"))),
        "jacques_ia":   Agent(name="Jacques.ia",             instructions=P.AGENT_JACQUES_IA, model="gpt-5",   model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
        "henry_ia":     Agent(name="Henry.ia",               instructions=P.AGENT_HENRY_IA,   model="gpt-5",   model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="medium"))),
    }
    return agents


# Agents initialisés à la première requête (lazy)
_AGENTS = None

def get_agents():
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = _build_agents()
    return _AGENTS


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

def _run_config():
    return RunConfig(trace_metadata={
        "__trace_source__": "swott-chatkit-python",
        "workflow_id": WORKFLOW_ID
    })

async def run_workflow(history: list) -> str:
    a = get_agents()

    async def _run(agent, h=None):
        inp = list(h) if h is not None else list(history)
        r = await Runner.run(agent, input=inp, run_config=_run_config())
        return r

    with trace("Impact3_CorteX"):

        r_cls = await _run(a["classifier"])
        parsed = r_cls.final_output
        category = parsed.category if hasattr(parsed, "category") else parsed.get("category", "")

        DIRECT = {
            "": "cortex_routage",
            "WAIT_CONFIRMATION": "cortex_routage",
            "DIAGNOSTIC_ORGANISATIONNEL": "leonard",
            "PLAN_ACTION_OEP": "isaac",
            "ANALYSE_DONNEES": "gustave",
            "DECOMPOSITION_COUTS": "achille",
            "JURIDIQUE_CONTRATS": "hypathie",
            "COMPARAISON_OFFRES": "hercule",
            "REDACTION_AO": "clint",
            "MATURITE_ACHATS": "barack",
            "CORTEX_CORE": "cortex_core",
            "POLITIQUE_ACHATS": "marcel",
            "PREPARATION_NEGOCIATION": "hector",
            "EMAILS_COMMUNICATION": "mazarin",
            "SEBUS_EXCEL": "sebus",
            "COMPTE_RENDU_CR": "franklin",
            "CAHIER_DES_CHARGES": "augustine",
            "RFAR_LABEL_DIAGNOSTIC": "hilda",
            "MESURE_IMPACT_CARBONE": "hermes",
            "REDACTION_PROCESSUS_ACHATS": "iris",
            "RH_ASSISTANCE": "ariane",
        }

        if category in DIRECT:
            r = await _run(a[DIRECT[category]])
            return r.final_output_as(str)

        elif category == "STRATEGIE_PORTEFEUILLE":
            r_j = await _run(a["jacques_json"])
            smr_axis = r_j.final_output.smr_axis if hasattr(r_j.final_output, "smr_axis") else ""
            h2 = list(history) + [item.to_input_item() for item in r_j.new_items]
            SMR = {"SMR_E": "eustache", "SMR_R": "marguerite", "SMR_CSR": "luther", "SMR_SH": "chan"}
            r = await _run(a.get(SMR.get(smr_axis, ""), a["jacques_ia"]), h2)
            return r.final_output_as(str)

        elif category == "LEVIERS_OPTIMISATION_PROJET":
            r_h = await _run(a["henry_json"])
            sml_axis = r_h.final_output.sml_axis if hasattr(r_h.final_output, "sml_axis") else ""
            h2 = list(history) + [item.to_input_item() for item in r_h.new_items]
            SML = {"SML_E": "michele", "SML_R": "albert", "SML_CSR": "savannah", "SML_SH": "catherine"}
            r = await _run(a.get(SML.get(sml_axis, ""), a["henry_ia"]), h2)
            return r.final_output_as(str)

        elif category == "SOURCING_MARCHE_FOURNISSEUR":
            r_cadrage = await _run(a["sherlock_cadrage"])
            h2 = list(history) + [item.to_input_item() for item in r_cadrage.new_items]
            r_fast = await _run(a["sherlock_fast"], h2)
            launch = r_fast.final_output.launch_deep if hasattr(r_fast.final_output, "launch_deep") else False
            if launch:
                r_deep = await _run(a["sherlock_deep"], h2)
                return r_deep.final_output_as(str)
            return r_cadrage.final_output_as(str)

        elif category == "BENCHMARK_CONCURRENTIEL":
            r_cadrage = await _run(a["freya_cadrage"])
            h2 = list(history) + [item.to_input_item() for item in r_cadrage.new_items]
            r_fast = await _run(a["freya_fast"], h2)
            launch = r_fast.final_output.launch_deep if hasattr(r_fast.final_output, "launch_deep") else False
            if launch:
                r_deep = await _run(a["freya_deep"], h2)
                return r_deep.final_output_as(str)
            return r_cadrage.final_output_as(str)

        else:
            r = await _run(a["cortex_routage"])
            return r.final_output_as(str)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION DE TEXTE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(filename: str, content_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    try:
        if ext == "pdf":
            import io, PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
            return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        elif ext in ("docx", "doc"):
            import io, docx
            doc = docx.Document(io.BytesIO(content_bytes))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        elif ext in ("xlsx", "xls"):
            import io, openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content_bytes), data_only=True)
            lines = []
            for sheet in wb.worksheets:
                lines.append(f"[Feuille: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    row_str = "\t".join(str(c) if c is not None else "" for c in row)
                    if row_str.strip():
                        lines.append(row_str)
            return "\n".join(lines).strip()
        elif ext == "txt":
            return content_bytes.decode("utf-8", errors="ignore")
        else:
            return f"[Fichier: {filename} — format non supporté]"
    except Exception as e:
        return f"[Erreur extraction {filename}: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORES (ChatKit)
# ═══════════════════════════════════════════════════════════════════════════════

class InMemoryStore(Store):
    def __init__(self):
        self._threads = {}
        self._items = {}
        self._attachments = {}

    async def load_thread(self, thread_id, context):
        if thread_id not in self._threads:
            self._threads[thread_id] = Thread(
                id=thread_id, title=None,
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


class InMemoryAttachmentStore(AttachmentStore):
    def __init__(self, store: InMemoryStore):
        self._store = store
        self._bytes = {}
        self._texts = {}

    @property
    def _attachments(self):
        return self._store._attachments

    async def create_attachment(self, input, context):
        att_id = f"att_{uuid.uuid4().hex[:12]}"
        content = getattr(input, "content", None)
        if content:
            raw = base64.b64decode(content) if isinstance(content, str) else content
            self._bytes[att_id] = raw
            name = getattr(input, "name", "fichier")
            self._texts[att_id] = extract_text(name, raw)
            print(f">>> Fichier reçu: {name} ({len(raw)} bytes)")

        att = FileAttachment(
            id=att_id,
            thread_id=getattr(input, "thread_id", ""),
            created_at=datetime.now(timezone.utc),
            name=getattr(input, "name", "fichier"),
            mime_type=getattr(input, "mime_type", "application/octet-stream"),
            size=getattr(input, "size", len(self._bytes.get(att_id, b""))),
        )
        self._attachments[att_id] = att
        return att

    async def delete_attachment(self, attachment_id, context):
        self._store._attachments.pop(attachment_id, None)
        self._bytes.pop(attachment_id, None)
        self._texts.pop(attachment_id, None)

    def get_text(self, att_id):
        return self._texts.get(att_id, "")

    def get_attachment(self, att_id):
        return self._attachments.get(att_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CHATKIT SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class SwottChatKitServer(ChatKitServer):
    def __init__(self, data_store, attachment_store):
        super().__init__(data_store, attachment_store=attachment_store)
        self.att_store = attachment_store

    async def respond(self, thread, input, context):
        message_text = ""
        attachment_ids = []

        if input and hasattr(input, "content"):
            c = input.content
            if isinstance(c, str):
                message_text = c
            elif isinstance(c, list):
                for part in c:
                    if hasattr(part, "text"):
                        message_text += part.text
                    elif hasattr(part, "attachment_id"):
                        attachment_ids.append(part.attachment_id)

        if input and hasattr(input, "attachments") and input.attachments:
            for ref in input.attachments:
                aid = getattr(ref, "id", None) or getattr(ref, "attachment_id", None)
                if aid:
                    attachment_ids.append(aid)

        if attachment_ids:
            files_ctx = "\n\n---DOCUMENTS FOURNIS PAR L'UTILISATEUR---\n"
            for aid in attachment_ids:
                att = self.att_store.get_attachment(aid)
                text = self.att_store.get_text(aid)
                name = att.name if att else aid
                files_ctx += f"\n[Fichier: {name}]\n{text}\n"
            files_ctx += "---FIN DES DOCUMENTS---\n"
            message_text = message_text + files_ctx

        user_id = context.get("user_id", "anon") if isinstance(context, dict) else "anon"

        # Sauvegarder le thread dans Supabase
        client_name = context.get("client_name", "") if isinstance(context, dict) else ""
        db_save_thread(thread.id, user_id, client_name)

        # Charger l'historique depuis Supabase
        history = db_get_history(thread.id)
        history.append({"role": "user", "content": message_text or " "})

        print(f">>> [{thread.id}] user_id={user_id} msg={message_text[:80]}")

        # Appel workflow
        import re
        response_text = await run_workflow(history)
        response_text = re.sub(r'filecite\S+', '', response_text).strip()

        # Persister dans Supabase
        tokens = len((message_text or " ").split()) + len(response_text.split())
        db_append_message(thread.id, "user", message_text or " ")
        db_append_message(thread.id, "assistant", response_text, tokens_used=tokens)

        print(f">>> Réponse: {response_text[:80]}")

        # Stream la réponse
        item_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        assistant_item = AssistantMessageItem(
            id=item_id, thread_id=thread.id, created_at=now,
            content=[AssistantMessageContent(type="output_text", text="", annotations=[])]
        )
        yield ThreadItemAddedEvent(item=assistant_item)
        yield AssistantMessageContentPartAdded(content_index=0, content={"type": "output_text", "text": ""})

        for i, word in enumerate(response_text.split(" ")):
            txt = word if i == 0 else " " + word
            yield AssistantMessageContentPartTextDelta(content_index=0, delta=txt)
            await asyncio.sleep(0.02)

        yield AssistantMessageContentPartDone(
            content_index=0,
            content={"type": "output_text", "text": response_text}
        )
        yield ThreadItemDoneEvent(item=AssistantMessageItem(
            id=item_id, thread_id=thread.id, created_at=now,
            content=[AssistantMessageContent(type="output_text", text=response_text, annotations=[])]
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = InMemoryStore()
att_store  = InMemoryAttachmentStore(data_store)
server     = SwottChatKitServer(data_store, att_store)


@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    body    = await request.body()
    user_id = request.headers.get("x-user-id") or request.query_params.get("user_id", "anon")
    client_name = request.headers.get("x-client-name") or request.query_params.get("client_name", "")
    result  = await server.process(body, {"user_id": user_id, "client_name": client_name})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    messages = db_get_history(thread_id)
    return {"thread_id": thread_id, "messages": messages}


@app.get("/threads")
async def get_threads(user_id: str = "anon"):
    threads = db_get_threads_for_user(user_id)
    return {"user_id": user_id, "threads": threads}

@app.get("/stats")
async def get_stats(user_id: str = None):
    try:
        query = supabase.table("threads") \
            .select("user_id, client_name, thread_id") \
            .execute()
        threads = query.data

        stats = {}
        for t in threads:
            key = t["user_id"]
            if key not in stats:
                stats[key] = {
                    "user_id": t["user_id"],
                    "client_name": t.get("client_name") or "—",
                    "nb_conversations": 0,
                    "nb_messages": 0,
                    "total_tokens": 0
                }
            stats[key]["nb_conversations"] += 1

            msgs = supabase.table("messages") \
                .select("id, tokens_used") \
                .eq("thread_id", t["thread_id"]) \
                .eq("role", "assistant") \
                .execute()
            stats[key]["nb_messages"] += len(msgs.data)
            stats[key]["total_tokens"] += sum(m["tokens_used"] or 0 for m in msgs.data)

        result = list(stats.values())
        if user_id:
            result = [r for r in result if r["user_id"] == user_id]
        return {"stats": sorted(result, key=lambda x: x["total_tokens"], reverse=True)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def healthcheck():
    return {"status": "ok", "service": "swott-chatkit", "mode": "agents-python-native+supabase"}
