import logging
from graph_store.connection import MultiGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_clips(video_id):
    with MultiGraphManager() as manager:
        query = "MATCH (c:Clip {video_id: $vid}) RETURN c.id as id"
        clips = manager.clip_graph.execute_query(query, {"vid": video_id})
        logger.info(f"Found {len(clips)} clips for video {video_id} in Graph 1.")
        return clips

if __name__ == "__main__":
    check_clips("79f019e3")
