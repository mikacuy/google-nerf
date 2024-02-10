from dciknn_cuda import DCI
import torch
import random
import datetime

import resource

random.seed(10)
torch.manual_seed(0)

def gen_data(ambient_dim, intrinsic_dim, num_points):
    latent_data = torch.randn((num_points, intrinsic_dim))
    transformation = torch.randn((intrinsic_dim, ambient_dim))
    data = torch.matmul(latent_data, transformation)
    return data     # num_points x ambient_dim


def brute_force_cpu(data, query, batch_size=1024, num_workers=0):
    nn_dataloader1 = torch.utils.data.DataLoader(
              dataset=torch.utils.data.TensorDataset(query),
              batch_size=batch_size,
              num_workers=num_workers,
              shuffle=False)
    
    NN_1_to_2 = torch.empty((len(query)), dtype=torch.int64)
    prev = 0
    
    a = datetime.datetime.now()
    
    for _, d in enumerate(nn_dataloader1):
        selected_entries = d[0]
        bs = selected_entries.shape[0]

        distances = torch.norm(selected_entries.unsqueeze(1) - data.unsqueeze(0), p=2, dim=-1) 
        _, min_indices = torch.min(distances, axis=-1)

        NN_1_to_2[prev : prev + bs] = min_indices
        prev = prev + bs
        
    b = datetime.datetime.now()

    with open('time.txt', 'a') as f:
        print('-----------------------------------------------', file=f)
        print(f'Brute Force Time (CPU): {(b-a).total_seconds()}s', file=f)
        print(f"Peak CPU memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2):.2f} GB", file=f)
        print(f'batch_size={batch_size}, num_workers={num_workers}', file=f)


def brute_force_gpu(data, query, batch_size=1024, num_workers=0):
    device=torch.device('cuda:0')
    data = data.to(device)
    query = query.to(device)
    
    nn_dataloader1 = torch.utils.data.DataLoader(
              dataset=torch.utils.data.TensorDataset(query),
              batch_size=batch_size,
              num_workers=num_workers,
              shuffle=False)
    
    NN_1_to_2 = torch.empty((len(query)), dtype=torch.int64)
    prev = 0
    
    a = datetime.datetime.now()
    
    for _, d in enumerate(nn_dataloader1):
        selected_entries = d[0]
        bs = selected_entries.shape[0]

        distances = torch.norm(selected_entries.unsqueeze(1) - data.unsqueeze(0), p=2, dim=-1) 
        _, min_indices = torch.min(distances, axis=-1)

        NN_1_to_2[prev : prev + bs] = min_indices
        prev = prev + bs
        
    b = datetime.datetime.now()
    
    with open('time.txt', 'a') as f:
        print('-----------------------------------------------', file=f)
        print(f'Brute Force Time (GPU): {(b-a).total_seconds()}s', file=f)
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB", file=f)
        print(f'batch_size={batch_size}, num_workers={num_workers}', file=f)
    

def dci_knn(data, query, block_size=100, thread_size=10, num_comp_indices=2, num_simp_indices=10, 
            num_outer_iterations=5000, device=torch.device('cuda:0')):
    data = data.to(device)
    query = query.to(device)
    
    num_neighbours = 1
    dim = data.shape[1]
    
    dci_db = DCI(dim, num_comp_indices, num_simp_indices, block_size, thread_size, device=0)
    
    a = datetime.datetime.now()
    dci_db.add(data)
    indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
    
    # print("Nearest Indices:", indices)
    # print("Indices Distances:", dists)
    
    dci_db.clear()
    b = datetime.datetime.now()
    
    with open('time.txt', 'a') as f:
        print('-----------------------------------------------', file=f)
        print(f'DCIKNN Time: {(b-a).total_seconds()}s', file=f)
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB", file=f)
        print(f'block_size={block_size}, thread_size={thread_size}, num_comp_indices={num_comp_indices}, num_simp_indices={num_simp_indices}, num_outer_iterations={num_outer_iterations}', file=f)


def main():
    # Data Generation Hyperparameters                                                                                                           
    dim = 391
    num_data = 10000
    num_queries = 10000

    intrinsic_dim = 400
    data_and_queries = gen_data(dim, intrinsic_dim, num_data + num_queries)

    data = data_and_queries[:num_data, :].detach().clone()
    query = data_and_queries[num_data:, :].detach().clone()
    
    # brute_force_gpu(data, query, batch_size=1024, num_workers=0)
    # brute_force_gpu(data, query, batch_size=1024, num_workers=4)
    
    # brute_force_cpu(data, query, batch_size=1024, num_workers=0)
    # brute_force_cpu(data, query, batch_size=1024, num_workers=4)
    
    # dci_knn(data, query, block_size=100, thread_size=10, num_comp_indices=2, num_simp_indices=10, num_outer_iterations=5000)
    # dci_knn(data, query, block_size=10, thread_size=10, num_comp_indices=2, num_simp_indices=10, num_outer_iterations=5000)
    # dci_knn(data, query, block_size=10, thread_size=1, num_comp_indices=2, num_simp_indices=10, num_outer_iterations=5000)
    # dci_knn(data, query, block_size=10, thread_size=10, num_comp_indices=1, num_simp_indices=10, num_outer_iterations=5000)
    # dci_knn(data, query, block_size=10, thread_size=10, num_comp_indices=1, num_simp_indices=1, num_outer_iterations=5000)
    # dci_knn(data, query, block_size=10, thread_size=10, num_comp_indices=1, num_simp_indices=1, num_outer_iterations=500)
    dci_knn(data, query, block_size=10, thread_size=10, num_comp_indices=1, num_simp_indices=1, num_outer_iterations=50)
    
    
if __name__ == '__main__':
    main()
