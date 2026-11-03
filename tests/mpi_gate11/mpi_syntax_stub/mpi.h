#ifndef MPI_H
#define MPI_H
typedef int MPI_Comm;
#define MPI_SUCCESS 0
#define MPI_DOUBLE 1
int MPI_Comm_rank(MPI_Comm,int*);
int MPI_Comm_size(MPI_Comm,int*);
int MPI_Allgatherv(const void*,int,int,void*,const int*,const int*,int,MPI_Comm);
#endif
