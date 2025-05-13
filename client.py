import asyncio
import websockets
import json
import base64



def save_frame_on_folder(image_base64, frame_number):
    # Decode the base64 image
    image_data = base64.b64decode(image_base64)
    
    # Save the image to a file
    with open(f"socket_frames/frame_{frame_number}.jpg", "wb") as image_file:
        image_file.write(image_data)
    print(f"Saved frame {frame_number} to disk.")


async def listen():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to the WebSocket server.")
        try:
            while True:
                data = await websocket.recv()
                json_data = json.loads(data)
                print("frame number:", json_data.get("frame_number"))
                print("Class name:", json_data.get("class_name"))
                if json_data.get("frame_base64") is not None:   
                    image_base64 = json_data.get("frame_base64")
                    save_frame_on_folder(image_base64, json_data.get("frame_number"))
        except websockets.ConnectionClosed:
            print("Connection closed.")

if __name__ == "__main__":
    asyncio.run(listen())



#     # server.py
# import asyncio
# import websockets
# import json

# async def handler(websocket, path):
#     async for message in websocket:
#         data = json.loads(message)
#         print("Received detection:", data)

# start_server = websockets.serve(handler, "0.0.0.0", 8765)

# asyncio.get_event_loop().run_until_complete(start_server)
# print("WebSocket server started on ws://localhost:8765")
# asyncio.get_event_loop().run_forever()
