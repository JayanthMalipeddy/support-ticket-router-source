# Integration contract

The router makes a recommendation. It does not send customer responses, create tickets, execute recovery or grant approval.

## Week 3: Integration Hub

Existing `Incident` inputs contain title, logs and service. `week3_intake()` produces the separate ticket request using title and logs. Call `RouterClient.route()` before or alongside the existing planner, and retain `week3_annotation()` as separate metadata. Do not add these keys to the existing strict Incident schema; a future consumer change must introduce a separate storage field.

Queue labels do not map directly to API/authentication, database and messaging investigation roles. The current planner still decides whether an incident fits those roles. Model unavailability, timeout, invalid output and general triage must keep the incident available to the human workflow. Existing approval is always required. See `examples/week3.py` for the callable integration seam.

## Week 4: Incident Replay Lab

The lab currently evaluates investigation answers and Laya's judge/human decisions. Those are different tasks. `examples/week4_export.py` exports labelled classification cases; expected labels are outside the model input. `ticket_router.evaluate` exports paired reports with task type `support_ticket_classification` and a shared dataset fingerprint.

To expose these inside Replay Lab, add a new experiment type and importer that accepts this exchange format, then render its exact classification metrics. Do not insert these records into existing investigation cases or count them toward the 40-case benchmark or judge-calibration results. No importer is installed into Week 4 by this project.

## HTTP example

```http
POST http://127.0.0.1:8025/v1/route
Authorization: Bearer <private-token>
Content-Type: application/json

{"ticket_id":"IT-100","text":"Outlook cannot send messages."}
```

Success includes schema_version, ticket_id, label, queue, needs_human_review, reason, model_id, prompt_version and latency_ms. Callers must check needs_human_review even for HTTP 200. Null label means invalid output, not a correct general-triage prediction. The service does not return a confidence score.

The OpenAPI document is generated from the FastAPI models; regenerate it after changes. Auth is required only for routing. Deployment beyond loopback needs a separate security/deployment review; this project is a local single-operator component.
