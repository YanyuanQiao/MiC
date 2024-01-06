import h5py
from ipdb import set_trace
import torch
import clip
import numpy
import json
import csv
import numpy as np
import base64


device = "cuda" if torch.cuda.is_available() else "cpu"
outfile = 'CLIP-ViT-B-32-views.tsv'

tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
features = {}
views = 36

room_classes = ["bathroom", "bedroom", "closet", "dining room", "entry-way", "familyroom", "garage", "hallway", "library", \
                "laundryroom", "kitchen", "livingroom", "meetingroom", "lounge", "office", "porch", "recroom", "stairs", "toilet", \
                "utilityroom", "gym", "outdoor", "other-room", "bar", "classroom", "diningbooth", "spa", "junk"]

text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in room_classes]).to(device)

# You can choose different image features, such as "ResNet-50" "ResNet-50x4" "ResNet-101" "ViT-B/32"
model, preprocess = clip.load("ViT-B/32", device=device)
# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

text_features /= text_features.norm(dim=-1, keepdim=True)

room_list = {}

with open(outfile, "r") as tsv_in_file:   
	reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
	for item in reader:
		long_id = item['scanId'] + "_" + item['viewpointId']

		features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                           dtype=np.float32).reshape((views, -1))
		data = features[long_id]

		image_features = torch.from_numpy(numpy.array(data)).cuda()
		image_features /= image_features.norm(dim=-1, keepdim=True)
		similarity = (100.0 * image_features @ text_features.T.float()).softmax(dim=-1)

		values, indices = similarity.mean(0).topk(1)
		room = [room_classes[ind] for ind in indices]
		room_list[long_id] = room

with open('room-VitB32.json', 'w') as obj:
 	json.dump(room_list, obj)	
