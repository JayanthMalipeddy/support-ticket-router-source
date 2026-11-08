import json
from pathlib import Path
from ticket_router.api import create_app
out=Path('docs/openapi.json'); out.parent.mkdir(exist_ok=True)
spec=create_app().openapi()
spec.setdefault('components',{}).setdefault('securitySchemes',{})['BearerAuth']={'type':'http','scheme':'bearer'}
spec['paths']['/v1/route']['post']['security']=[{'BearerAuth':[]}]
out.write_text(json.dumps(spec,indent=2),encoding='utf-8')
print(out)
