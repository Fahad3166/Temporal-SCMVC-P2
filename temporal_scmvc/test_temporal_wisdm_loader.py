from dataloader_wisdm_temporal import load_wisdm_temporal

dataset, dims, view, data_size, class_num = load_wisdm_temporal(
    window_size=200,
    max_files_per_view=5,
    selected_view_indices=[0, 1, 2],
    max_samples=1000
)

print("\ndims =", dims)
print("view =", view)
print("data_size =", data_size)
print("class_num =", class_num)

xs, y, idx = dataset[0]
print("Label:", y)

for i, x in enumerate(xs):
    print(f"View {i+1} tensor shape:", x.shape)

    