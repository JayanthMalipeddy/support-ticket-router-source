"""Explicit consumer adapters; never mutate Week 3/4 databases."""
from .api import RoutingResult
from .domain import LABELS

def week3_intake(ticket_id,incident):
    return {"ticket_id":ticket_id,"text":f"{incident['title']}\n\n{incident['logs']}"}
def week3_annotation(result):
    return {"support_routing":RoutingResult.model_validate(result).model_dump(),
            "requires_existing_planner":True,"approval_granted":False}
def week4_case(row):
    if row["label"] not in LABELS: raise ValueError("Unknown support label")
    return {"schema_version":"1","task_type":"support_ticket_classification","case_id":row["id"],
            "input":{"ticket_id":row["id"],"text":row["text"]},"expected":{"label":row["label"]}}
class RouterClient:
    def __init__(self,base_url,token,timeout=120):
        import httpx
        self.client=httpx.Client(base_url=base_url,headers={"Authorization":"Bearer "+token},timeout=timeout)
    def route(self,ticket_id,text):
        response=self.client.post("/v1/route",json={"ticket_id":ticket_id,"text":text})
        response.raise_for_status()
        return RoutingResult.model_validate(response.json()).model_dump()
    def close(self): self.client.close()
