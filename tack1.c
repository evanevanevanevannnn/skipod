#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ARG(...) __VA_ARGS__

#define SIZE 4

#define COORDS(name) int name[2]

#define COORDS_INIT(name, init) \
    int name[2];\
    name[0] = init[0];\
    name[1] = init[1];\

#define COORDS_EQ(c1, c2) ((c1[0] == c2[0]) && (c1[1] == c2[1]))

#define MPI_UPDATE_MAX(send_array, recv_array, size)                            \
    {                                                                           \
        int buff[2];                                                            \
        MPI_Status status;                                                      \
        MPI_Request request;                                                    \
        int send_procs[size][2] = send_array;                                   \
        int recv_procs[size][2] = recv_array;                                   \
        for (int i = 0; i < size; ++i) {                                        \
            COORDS_INIT(sender_coords, send_procs[i]);                          \
            COORDS_INIT(recver_coords, recv_procs[i]);                          \
            if (COORDS_EQ(coords, sender_coords)) {                             \
                int recver_rank;                                                \
                MPI_Cart_rank(comm, recver_coords, &recver_rank);               \
                buff[0] = max_N, buff[1] = max_rank;                            \
                MPI_Isend(&buff, 2, MPI_INT, recver_rank, 0, comm, &request);   \
            }                                                                   \
        }                                                                       \
        for (int i = 0; i < size; ++i) {                                        \
            COORDS_INIT(sender_coords, send_procs[i]);                          \
            COORDS_INIT(recver_coords, recv_procs[i]);                          \
            if (COORDS_EQ(coords, recver_coords)) {                             \
                int sender_rank, new_N, new_rank;                               \
                MPI_Cart_rank(comm, sender_coords, &sender_rank);               \
                MPI_Recv(buff, 2, MPI_INT, sender_rank, 0, comm, &status);      \
                new_N = buff[0], new_rank = buff[1];                            \
                if (new_N > max_N || (new_N == max_N && new_rank < rank)) {     \
                    max_rank = new_rank;                                        \
                    max_N = new_N;                                              \
                }                                                               \
            }                                                                   \
        }                                                                       \
        MPI_Barrier(comm);                                                      \
    }                                                                           \

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, tasks;
    MPI_Comm comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);

    srand(time(NULL) + rank);
    int N = rand() % 1000;

    // создание транспьютерной матрицы
    MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){SIZE, SIZE}, (int[]){0, 0}, 0, &comm);

    COORDS(coords);
    MPI_Cart_coords(comm, rank, 2, coords);

    printf("Coordinates for process %d: (%d, %d)\n", rank, coords[0], coords[1]);
    printf("N[%d][%d] = %d\n", coords[0], coords[1], N);

    int max_N = N;
    int max_rank = rank;

    // обмен значениями за 8 шагов
    // шаг 1
    MPI_UPDATE_MAX(
        ARG({{0, 0}, {0, 1}, {0, 2}, {0, 3}, {3, 0}, {3, 1}, {3, 3}, {3, 3}}),
        ARG({{1, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 3}, {2, 3}}),
        8
    )

    // шаг 2
    MPI_UPDATE_MAX(
        ARG({{1, 0}, {1, 3}, {2, 0}, {2, 3}}),
        ARG({{1, 1}, {1, 2}, {2, 1}, {2, 2}}),
        4
    )

    // шаг 3
    MPI_UPDATE_MAX(
        ARG({{1, 1}, {1, 1}, {1, 2}, {1, 2}, {2, 1}, {2, 1}, {2, 2}, {2, 2}}),
        ARG({{2, 1}, {1, 2}, {1, 1}, {2, 2}, {1, 1}, {2, 2}, {1, 2}, {2, 1}}),
        8
    )

    // шаг 4
    MPI_UPDATE_MAX(
        ARG({{1, 1}, {1, 1}, {1, 2}, {1, 2}, {2, 1}, {2, 1}, {2, 2}, {2, 2}}),
        ARG({{2, 1}, {1, 2}, {1, 1}, {2, 2}, {1, 1}, {2, 2}, {1, 2}, {2, 1}}),
        8
    )

    // шаг 5
    MPI_UPDATE_MAX(
        ARG({{1, 1}, {1, 2}, {2, 1}, {2, 2}}),
        ARG({{1, 0}, {1, 3}, {2, 0}, {2, 3}}),
        4
    )

    // шаг 6
    MPI_UPDATE_MAX(
        ARG({{1, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 3}, {2, 3}}),
        ARG({{0, 0}, {0, 1}, {0, 2}, {0, 3}, {3, 0}, {3, 1}, {3, 3}, {3, 3}}),
        8
    )

    if (coords[0] == 0 && coords[1] == 0) {
        COORDS(max_N_coords);
        MPI_Cart_coords(comm, max_rank, 2, max_N_coords);
        printf("Max N: %d, sender coords: %d %d\n", max_N, max_N_coords[0], max_N_coords[1]);
    }

    MPI_Finalize();
    return 0;
}