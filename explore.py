from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)

scene = nusc.scene[0]
print("Scene name:", scene['name'])
print("Description:", scene['description'])
print("Number of samples:", scene['nbr_samples'])

first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)
print("\nFirst sample timestamp:", sample['timestamp'])

# Get annotations directly from sample annotations
ann_tokens = sample['anns']
print("Objects in first frame:", len(ann_tokens))

for ann_token in ann_tokens:
    ann = nusc.get('sample_annotation', ann_token)
    category = ann['category_name']
    x = ann['translation'][0]
    y = ann['translation'][1]
    print(f"  {category:40s} at ({x:.1f}, {y:.1f}) meters")
instance=nusc.instance[0]
print("\nTracking instanc:",instance['token'])
print("Category:",nusc.get('category',instance['category_token'])['name'])
print("Total annotations:",instance['nbr_annotations'])
ann_token=instance['first_annotation_token']
positions=[]
while ann_token:
    ann=nusc.get('sample_annotation',ann_token)
    x=ann['translation'][0]
    y=ann['translation'][1]
    positions.append((x,y))
    ann_token=ann['next']
print(f"\nTrajecory({len(positions)}frames):")
for i,(x,y)in enumerate(positions):
    print(f"Frame{i:2d}: ({i:2d}: ({x:.1f},{y:.1f}))")


