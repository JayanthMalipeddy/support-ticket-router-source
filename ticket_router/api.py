import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI,Header,HTTPException
from pydantic import BaseModel,ConfigDict,Field,field_validator
from .domain import decision,PROMPT_VERSION

class Ticket(BaseModel):
    model_config=ConfigDict(extra="forbid")
    ticket_id:str=Field(min_length=1,max_length=120)
    text:str=Field(min_length=3,max_length=12000)
    @field_validator("ticket_id","text")
    @classmethod
    def nonblank(cls,value):
        if not value.strip(): raise ValueError("Value must not be blank")
        return value

class RoutingResult(BaseModel):
    schema_version:Literal["1"]="1"
    ticket_id:str
    label:Literal["Active Directory","Computer-Services","EOL","Fileservice","O365","Software","Support general"]|None
    queue:str|None
    needs_human_review:bool
    reason:str
    model_id:str
    prompt_version:str
    latency_ms:float

def create_app(classifier=None,token=None):
    token=token if token is not None else os.getenv("ROUTER_API_TOKEN","")
    lock=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        if app.state.classifier is None and os.getenv("ROUTER_MODEL_PATH"):
            from .inference import HFClassifier
            app.state.classifier=HFClassifier(os.environ["ROUTER_MODEL_PATH"])
        yield
    app=FastAPI(title="Support Ticket Router",version="0.1.0",lifespan=lifespan)
    app.state.classifier=classifier
    @app.get("/health")
    def health():
        return {"status":"ready" if app.state.classifier and token else "not_ready",
                "model_loaded":app.state.classifier is not None,"auth_configured":bool(token)}
    @app.post("/v1/route",response_model=RoutingResult)
    def route(ticket:Ticket,authorization:str=Header(default="")):
        if not token: raise HTTPException(503,"ROUTER_API_TOKEN not configured")
        if not secrets.compare_digest(authorization,"Bearer "+token): raise HTTPException(401,"Bearer token required")
        if app.state.classifier is None: raise HTTPException(503,"No trained model loaded")
        if not lock.acquire(blocking=False): raise HTTPException(429,"Router busy; retry later")
        try:
            result=app.state.classifier.predict(ticket.text)
            return RoutingResult(ticket_id=ticket.ticket_id,**decision(result["raw_output"]),model_id=result["model_id"],prompt_version=PROMPT_VERSION,latency_ms=result["latency_ms"])
        except ValueError as exc: raise HTTPException(422,str(exc)) from exc
        except Exception as exc: raise HTTPException(503,"Inference failed; retain ticket for human triage") from exc
        finally: lock.release()
    return app
app=create_app()
