from video_rag_query.graph_api import GraphAPI
from video_rag_query.utils import extract_keywords
import os
from dotenv import load_dotenv

load_dotenv()

api = GraphAPI()
clips = ['303cbc17_5.34_10.98', '303cbc17_51.92_59.13']
props = api.get_node_properties(clips)

query = "is Airport Constructed on Swamp?"
keywords = extract_keywords(query)

print(f"Query Keywords: {keywords}")

for cid in clips:
    p = props.get(cid, {})
    text = " ".join([str(p.get('transcript', '')), str(p.get('ocr', '')), str(p.get('summary', ''))]).lower()
    matches = [kw for kw in keywords if kw in text]
    print(f"\nClip: {cid}")
    print(f"  Transcript: {p.get('transcript')[:100]}...")
    print(f"  Matches: {matches}")
    
api.close()
