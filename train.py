import json 

def load_model_cfg(json_file):
    cfg = json.load(json_file)
    return cfg

if __name__ == "__main__":
    print("training your model here.")