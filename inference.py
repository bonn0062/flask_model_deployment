import json

from commons import get_model, get_tensor

with open('cat_to_name.json') as f:
	cat_to_name = json.load(f)

with open('class_to_idx.json') as f:
	class_to_idx = json.load(f)


idx_to_class = {v:k for k, v in class_to_idx.items()}

model = get_model()

def get_flower_name(image_bytes):
	tensor = get_tensor(image_bytes)
	outputs = model.forward(tensor)
	_, prediction = outputs.max(1)
	category = prediction.item()
	class_idx = idx_to_class[category]
	flower_name = cat_to_name[class_idx]
	return category, flower_name