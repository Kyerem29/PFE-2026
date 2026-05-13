import gradio as gr
import os, sqlite3, json

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ── Config ───────────────────────────────────────────────────────────────────
DB_CHROMA = "./chroma_db_matplotlib"
DATA_PATH = "./data"
DB_SQL    = "chat_history.db"

llm        = OllamaLLM(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if not os.path.exists(DB_CHROMA):
    os.makedirs(DATA_PATH, exist_ok=True)
    docs   = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70).split_documents(docs or [])
    vs     = Chroma.from_documents(splits, embeddings, persist_directory=DB_CHROMA)
else:
    vs = Chroma(persist_directory=DB_CHROMA, embedding_function=embeddings)

retriever = vs.as_retriever(search_kwargs={"k": 3})

# ── Base de données ──────────────────────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_SQL)
    conn.execute("CREATE TABLE IF NOT EXISTS chats (user,chat_id,chat_name,messages)")
    conn.commit()
    return conn

def load_chats(user):
    rows = db().execute("SELECT chat_id,chat_name,messages FROM chats WHERE user=?", (user,)).fetchall()
    return {r[0]: {"name": r[1], "msgs": json.loads(r[2])} for r in rows} or {"chat_1": {"name": "Discussion 1", "msgs": []}}

def save_chat(user, cid, name, msgs):
    conn = db()
    if conn.execute("SELECT 1 FROM chats WHERE user=? AND chat_id=?", (user, cid)).fetchone():
        conn.execute("UPDATE chats SET messages=?,chat_name=? WHERE user=? AND chat_id=?", (json.dumps(msgs), name, user, cid))
    else:
        conn.execute("INSERT INTO chats VALUES (?,?,?,?)", (user, cid, name, json.dumps(msgs)))
    conn.commit()

def del_chat(user, cid):
    conn = db(); conn.execute("DELETE FROM chats WHERE user=? AND chat_id=?", (user, cid)); conn.commit()

# ── Logique ──────────────────────────────────────────────────────────────────
def get_answer(message, history, cid, request: gr.Request):
    docs    = retriever.invoke(message)
    context = "\n\n".join(d.page_content for d in docs)
    sources = list({os.path.basename(d.metadata.get("source","")) for d in docs if d.metadata.get("source")})
    hist_str = "".join(f"{'User' if m['role']=='user' else 'AI'}: {m['content']}\n" for m in history)

    prompt   = f"Expert Data Science. Réponds en FRANÇAIS.\nCONTEXTE: {context}\nHISTORIQUE:\n{hist_str}\nQUESTION: {message}\nRÉPONSE:"
    response = llm.invoke(prompt)

    if sources:
        response += "\n\n" + " ".join(f"`📄 {s}`" for s in sources)

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": response})

    chats = load_chats(request.username)
    save_chat(request.username, cid, chats.get(cid, {}).get("name", "Discussion"), history)
    return "", history

# ── Callbacks UI ─────────────────────────────────────────────────────────────
def on_load(request: gr.Request):
    chats   = load_chats(request.username)
    choices = [(v["name"], k) for k, v in chats.items()]
    fid     = choices[0][1]
    return gr.update(choices=choices, value=fid), chats[fid]["msgs"], fid, chats[fid]["name"]

def new_chat(request: gr.Request):
    chats  = load_chats(request.username)
    nid    = f"chat_{len(chats)+1}_{os.urandom(2).hex()}"
    nname  = f"Nouvelle Discussion {len(chats)+1}"
    save_chat(request.username, nid, nname, [])
    chats  = load_chats(request.username)
    return gr.update(choices=[(v["name"],k) for k,v in chats.items()], value=nid), [], nid, nname

def rename(name, cid, request: gr.Request):
    chats = load_chats(request.username)
    save_chat(request.username, cid, name, chats.get(cid, {}).get("msgs", []))
    chats = load_chats(request.username)
    return gr.update(choices=[(v["name"],k) for k,v in chats.items()], value=cid)

def delete(cid, request: gr.Request):
    del_chat(request.username, cid)
    return on_load(request)

def switch(cid, request: gr.Request):
    chats = load_chats(request.username)
    return (chats[cid]["msgs"], cid, chats[cid]["name"]) if cid in chats else ([], cid, "")

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

.gradio-container { font-family: 'DM Sans', sans-serif !important; max-width: 100% !important; }

.sidebar { background: #F1EFE8 !important; border-right: 1px solid #D3D1C7 !important; padding: 20px 14px !important; }

.chat-header { border-bottom: 1px solid #E8E6DF; padding: 14px 20px; background: #fff; margin-bottom: 0; }

.chatbot .user  > .message { background: #534AB7 !important; color: #fff !important; border-radius: 18px 18px 4px 18px !important; }
.chatbot .bot   > .message { background: #F1EFE8 !important; border: 1px solid #D3D1C7 !important; border-radius: 18px 18px 18px 4px !important; }
.chatbot .bot   > .message code { background: #EEEDFE !important; color: #3C3489 !important; border: 1px solid #AFA9EC !important; border-radius: 20px !important; padding: 2px 10px !important; font-size: 11px !important; }

.msg-box textarea { border-radius: 24px !important; border: 1px solid #D3D1C7 !important; background: #F8F7F5 !important; padding: 10px 18px !important; font-family: 'DM Sans', sans-serif !important; }
.msg-box textarea:focus { border-color: #534AB7 !important; background: #fff !important; box-shadow: 0 0 0 3px #EEEDFE !important; }

.send-btn button { background: #534AB7 !important; color: #fff !important; border-radius: 24px !important; border: none !important; font-weight: 500 !important; }
.send-btn button:hover { background: #3C3489 !important; }

.btn-new button { background: #fff !important; border: 1px solid #D3D1C7 !important; color: #444 !important; border-radius: 8px !important; }
.btn-new button:hover { background: #EEEDFE !important; border-color: #AFA9EC !important; color: #3C3489 !important; }

.btn-del button { background: #FCEBEB !important; border: 1px solid #F09595 !important; color: #A32D2D !important; border-radius: 8px !important; }

.btn-save button { background: #fff !important; border: 1px solid #D3D1C7 !important; color: #444 !important; border-radius: 8px !important; width: 100% !important; }
.btn-save button:hover { background: #F1EFE8 !important; }

::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: #D3D1C7; border-radius: 10px; }
"""

# ── Thème ────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
).set(
    body_background_fill="#F1EFE8",
    block_background_fill="#FFFFFF",
    block_border_color="#D3D1C7",
    block_shadow="none",
    button_primary_background_fill="#534AB7",
    button_primary_background_fill_hover="#3C3489",
    button_primary_text_color="#FFFFFF",
    input_background_fill="#FFFFFF",
    input_border_color="#D3D1C7",
    input_border_color_focus="#534AB7",
)

# ── Interface ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Assistant Data Science") as demo:
    cid = gr.State()

    with gr.Row():
        with gr.Column(scale=1, min_width=250, elem_classes="sidebar"):
            gr.Markdown("### 💬 Historique personnel")
            dropdown = gr.Dropdown(label="Mes discussions", choices=[], interactive=True)
            with gr.Row():
                btn_new = gr.Button("＋ Nouveau",   size="sm", elem_classes="btn-new")
                btn_del = gr.Button("⊘ Supprimer", size="sm", variant="stop", elem_classes="btn-del")
            gr.HTML("<hr style='border:none;border-top:1px solid #D3D1C7;margin:8px 0'>")
            name_input = gr.Textbox(label="Renommer", placeholder="Nom du chat…")
            btn_save   = gr.Button("💾 Sauvegarder", size="sm", elem_classes="btn-save")
            gr.HTML("""
                <div style='margin-top:24px;font-size:12px;color:#888'>
                    <a href='/logout' style='color:#534AB7;text-decoration:none'>↪ Se déconnecter</a><br><br>
                    <span style='background:#EEEDFE;color:#3C3489;padding:3px 10px;border-radius:20px;
                                 border:1px solid #AFA9EC;font-size:11px;font-weight:600'>PFE 2026</span>
                </div>
            """)

        with gr.Column(scale=4):
            gr.HTML("""
                <div class='chat-header'>
                    <div style='display:flex;align-items:center;gap:12px'>
                        <div style='width:34px;height:34px;border-radius:50%;background:#EEEDFE;
                                    display:flex;align-items:center;justify-content:center;font-size:16px'>✦</div>
                        <div>
                            <b style='font-size:15px;color:#2C2C2A'>Assistant expert Data Science</b><br>
                            <span style='font-size:12px;color:#888'>Powered by Mistral · RAG Matplotlib</span>
                        </div>
                        <div style='margin-left:auto;background:#EAF3DE;color:#27500A;font-size:11px;
                                    padding:4px 12px;border-radius:20px;border:1px solid #97C459'>✓ Sécurisé</div>
                    </div>
                </div>
            """)
            chatbot = gr.Chatbot(label="", height=520, elem_classes="chatbot")
            with gr.Row():
                msg      = gr.Textbox(placeholder="Posez votre question…", scale=5,
                                      show_label=False, elem_classes="msg-box")
                btn_send = gr.Button("Envoyer ➤", variant="primary", scale=1, elem_classes="send-btn")

    demo.load(on_load,  None,              [dropdown, chatbot, cid, name_input])
    btn_send.click(get_answer, [msg, chatbot, cid], [msg, chatbot])
    msg.submit(get_answer,     [msg, chatbot, cid], [msg, chatbot])
    btn_new.click(new_chat,  None,              [dropdown, chatbot, cid, name_input])
    btn_save.click(rename,   [name_input, cid], [dropdown])
    btn_del.click(delete,    [cid],             [dropdown, chatbot, cid, name_input])
    dropdown.change(switch,  [dropdown],        [chatbot, cid, name_input])

if __name__ == "__main__":
    demo.launch(
        auth=[("kevin", "pfe2026"), ("admin", "pfe2026")],
        auth_message="Espace de travail confidentiel — Authentification requise.",
        server_port=7860,
        theme=theme,
        css=CSS,
    )
