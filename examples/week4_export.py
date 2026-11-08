"""Write an exchange bundle for a future Week 4 classification experiment type."""
import json
from pathlib import Path
from ticket_router.integrations import week4_case
rows=json.loads(Path('data/prepared/validation.json').read_text(encoding='utf-8'))
out=Path('artifacts/week4-cases.json'); out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({'schema_version':'1','task_type':'support_ticket_classification','cases':[week4_case(r) for r in rows]},indent=2),encoding='utf-8')
print(out)
