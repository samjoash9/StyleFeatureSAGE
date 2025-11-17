import torch

data = torch.load("pretrained_models/vgg_face_editor.pt", map_location="cpu")

print("Type:", type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())

print(data['state_dict']) 