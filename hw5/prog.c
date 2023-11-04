#define _GNU_SOURCE
#include <mpi.h>
#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int is_exit_on_error = 1;
static void __attribute__((destructor)) abort_on_error()
{
  // 确保其一进程 err/errx() 调用时终止整个进程组
  if(is_exit_on_error) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
  int np, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // 从命令行参数接收 root 值
  if(argc != 2) errx(EXIT_FAILURE, "expect 2 arguments, got %d", argc);
  int root = atoi(argv[1]);
  if(root < 0) errx(EXIT_FAILURE, "expect non-negative root, got %d", root);

  // 获取与父进程组连接的通信子
  MPI_Comm parent = MPI_COMM_NULL;
  MPI_Comm_get_parent(&parent);

  // 从父进程组接收数组长度
  // 匹配：MPI_Scatter(dist, 1, MPI_INT, NULL, 0, MPI_INT, MPI_ROOT, inter_comm); 
  int length = -1;
  MPI_Scatter(NULL, 0, MPI_INT, &length, 1, MPI_INT, root, parent);
  //printf("[%2d] length: %d\n", rank, length);
  if(length < 0) errx(EXIT_FAILURE, "expect non-negative length, got %d", length);

  // 数组即将在本进程组内的 np 个进程的地址空间中顺序分布式存储
  // 每个进程各自知道自己拥有的片段的长度，但它们未必相等
  // 首先计算片段长度前缀和，供下面进行数据交换的代码所使用
  // 然后由前缀和获取数组总长度，以确定需要分配的缓冲区大小
  int my_presum_length = -1, presum_length[np];
  memset(presum_length, -1, sizeof presum_length);
  MPI_Scan(&length, &my_presum_length, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  //printf("[%2d] presum length: %d\n", rank, my_presum_length);
  MPI_Allgather(&my_presum_length, 1, MPI_INT, presum_length, 1, MPI_INT, MPI_COMM_WORLD);
  int total_length = presum_length[np - 1];
  //printf("[%2d] total length: %d\n", rank, total_length);
  int slice_length = (total_length - rank + np - 1) / np;
  //printf("[%2d] slice length: %d\n", rank, slice_length);

  // 从父进程组接收数组内容
  // 匹配：MPI_Scatterv(a, dist, displ, MPI_INT, NULL, 0, MPI_INT, MPI_ROOT, inter_comm);        
  int *buf = (int *)malloc(length * sizeof(int));
  if(buf == NULL) errx(EXIT_FAILURE, "error allocating memory");
  MPI_Scatterv(NULL, NULL, NULL, MPI_INT, buf, length, MPI_INT, root, parent);
  //printf("[%2d] data received from parent\n", rank);

  // 在进程组内调整数组的分布式存储方式
  // 当前本地缓冲区中的数据为 ARRAY[presum_length[rank - 1] : presum_length[rank]]
  // 需要将 ARRAY[rank : total_length : np] 转移至本地缓冲区
  int scounts[np], rcounts[np];
  int sdispls[np], rdispls[np];
  MPI_Datatype stypes[np], rtypes[np];
  int first_receiver = rank == 0 ? 0 : presum_length[rank - 1] % np;
  for(int offset = 0; offset < np; ++offset) {
    int receiver = (first_receiver + offset) % np;
    if(offset < length) {
      scounts[receiver] = 1;
      sdispls[receiver] = offset * sizeof(int);
      MPI_Type_vector((length - offset + np - 1) / np, 1, np, MPI_INT, &stypes[receiver]);
    } else {
      scounts[receiver] = 0;
      sdispls[receiver] = length * sizeof(int);
      MPI_Type_vector(0, 1, np, MPI_INT, &stypes[receiver]);
    }
    MPI_Type_commit(&stypes[receiver]);
  }
  for(int sender = 0, offset = 0; sender < np; ++sender) {
    int end = (presum_length[sender] - rank + np - 1) / np;
    if(end < 0) end = 0;
    rcounts[sender] = end - offset;
    rdispls[sender] = offset * sizeof(int);
    rtypes[sender] = MPI_INT;
    offset = end;
  }
  int *slice_buf = (int *)malloc(slice_length * sizeof(int));
  if(slice_buf == NULL) errx(EXIT_FAILURE, "error allocating memory");
  MPI_Alltoallw(buf, scounts, sdispls, stypes, slice_buf, rcounts, rdispls, rtypes, MPI_COMM_WORLD);
  for(int receiver = 0; receiver < np; ++receiver) MPI_Type_free(&stypes[receiver]);
  //printf("[%2d] data rearranged in the group\n", rank);

  // 向父进程发送加工后的数组内容
  // 匹配：MPI_Gatherv(NULL, 0, MPI_INT, b, dist, displ, MPI_INT, MPI_ROOT, inter_comm);    
  MPI_Gatherv(slice_buf, slice_length, MPI_INT, NULL, NULL, NULL, MPI_INT, root, parent);
  //printf("[%2d] data sent to parent\n", rank);

  free(slice_buf);
  free(buf);
  MPI_Finalize();
  is_exit_on_error = 0;
  return 0;
}
