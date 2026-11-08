"""Run from the Week 5 project; makes only a routing request."""
import os
from ticket_router.integrations import RouterClient,week3_intake,week3_annotation
client=RouterClient('http://127.0.0.1:8025',os.environ['ROUTER_API_TOKEN'])
try:
    payload=week3_intake('example-1',{'title':'Outlook cannot send mail','logs':'Messages remain in Outbox.'})
    print(week3_annotation(client.route(**payload)))
finally:
    client.close()
