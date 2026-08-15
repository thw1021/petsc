#include <mpi.h>
#include <stdio.h>
int main(int c,char**v){int r,s,one=1,sum=0;MPI_Init(&c,&v);MPI_Comm_rank(MPI_COMM_WORLD,&r);
MPI_Comm_size(MPI_COMM_WORLD,&s);MPI_Allreduce(&one,&sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
printf("rank %d/%d allreduce=%d\n",r,s,sum);MPI_Finalize();return 0;}
