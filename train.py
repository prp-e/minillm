import json 

def load_model_cfg(json_file):
    with open(json_file) as f:
        cfg = json.load(f)
    return cfg

if __name__ == "__main__":
    print("training your model here.")