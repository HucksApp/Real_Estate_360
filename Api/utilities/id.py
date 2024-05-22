import random
import uuid
def uniqueid():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

def generate_id(file_name=None):
    id = uuid.uuid4()
    if file_name:
        return "{}.{}".format(file_name, id)
    return id