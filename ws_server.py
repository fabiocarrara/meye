import asyncio
import json
import websockets


async def handler(websocket, path):
    async for data in websocket:
        data = json.loads(data)
        print(f'Data receieved: {data}')


host = 'localhost'
port = 8000
start_server = websockets.serve(handler, host, port)
asyncio.get_event_loop().run_until_complete(start_server)

print(f'Serving at ws://{host}:{port}')
asyncio.get_event_loop().run_forever()