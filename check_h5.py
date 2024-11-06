import h5py

file_path = "F:/python project/RAG_dense_vector_search/dataset/msmarco_passages_embeddings_subset.h5" #change the path to your own path

# 打开 .h5 文件并打印其中的数据集结构
with h5py.File(file_path, 'r') as f:
    print("File structure:")
    def print_attrs(name, obj):
        print(name)
    f.visititems(print_attrs)
