from dataloader_mhealth_temporal import load_mhealth_temporal

dataset, dims, view, data_size, class_num = load_mhealth_temporal(
    window_size=50,
    stride=25,
    max_samples=1000
)

print("dims =", dims)
print("view =", view)
print("data_size =", data_size)
print("class_num =", class_num)

xs, y, idx = dataset[0]
print("Label:", y)
for i, x in enumerate(xs):
    print(f"View {i+1} tensor shape:", x.shape)