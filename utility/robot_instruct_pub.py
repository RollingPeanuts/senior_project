# Test script to send commands to robot directly without needing cv component
# Simply start up robot arm and finish calibration, then run this script and send commands over
from google.cloud import pubsub_v1

# TODO(developer)
project_id = "robotic-haven-256402"
topic_id = "test-topic"

publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
topic_path = publisher.topic_path(project_id, topic_id)

while(True):
    data_str = input("Send command to robot (<piece1> <location_index1>, <piece2> <location_index2>):")
    # Data must be a bytestring
    data = data_str.encode("utf-8")
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data)
    print(future.result())

print(f"Published messages to {topic_path}.")
