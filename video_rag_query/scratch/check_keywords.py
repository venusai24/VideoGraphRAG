from video_rag_query.graph_api import GraphAPI
import os
from dotenv import load_dotenv

load_dotenv()

api = GraphAPI()
props = api.get_node_properties(['303cbc17_31.00_41.46'])
print(props)
api.close()
