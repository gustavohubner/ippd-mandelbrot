#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

float norm(float val, float rmin, float rmax, float vmin, float vmax)
{
    return ((vmax - vmin) * ((val - rmin) / (rmax - rmin))) + vmin;
}

int x = 1000;
int y = 1000;
int maxIter = 30;

int mandel(int i, int j)
{
    float x0 = norm(i, 0, x, -2.00, 0.47);
    float y0 = norm(j, 0, y, -1.12, 1.12);

    float xx = 0.0;
    float yy = 0.0;
    int iteration = 0;

    float xtemp = 0.0;

    while ((iteration < maxIter) && (((xx * xx) + (yy * yy) + y0) < 4))
    {
        xtemp = xx * xx - yy * yy + x0;
        yy = 2 * xx * yy + y0;
        xx = xtemp;
        iteration = iteration + 1;
    }
    return iteration;
}

int main(int argc, char ** argv)
{
    int worldSize, myRank, aux, dest, areaa[2];
    MPI_Status st;
    MPI_Init(&argc, &argv);
    MPI_Request req;
    FILE *a;

    if (argc == 4){
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        maxIter = atoi(argv[3]);
        printf("Tamanho %dx%d\n%d Iterações\n",x,y,maxIter);
    } else {
        printf("Usar: %s [Numero de linhas] [Numero de colunas] [Numero de iteracoes]\n", argv[0]);
        exit(1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (worldSize == 1)
        aux = 0;
    else
    {
        if (myRank == 0)
        {
            for (int i = 1; i < worldSize; i++)
            {
                areaa[0] = (x / worldSize) * i;
                areaa[1] = ((x / worldSize) * (i + 1)) - 1;
                MPI_Send((void *)areaa, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
                printf("Enviado area para %d\n", i);
            }

            areaa[0] = 0;
            areaa[1] = ((x / worldSize) * 1) - 1;
        }
        else
        {
            MPI_Recv(areaa, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            printf("Recebi %d %d rank %d\n", areaa[0], areaa[1], myRank);
        }
    }
    int *output = malloc(sizeof(int) * (y * (areaa[1] - areaa[0] + 1)));

#pragma omp parallel for num_threads(8) collapse(2)
    for (int i = 0; i <= areaa[1] - areaa[0]; i++)
    {
        for (int j = 0; j < y; j++)
        {
            output[(i * y) + j] = mandel(areaa[0] + i, j);
        }
    }

    if (myRank == 0)
    {
        a = fopen("saida.ppm", "w");
        fprintf(a, "P3\n%d %d\n255\n", x, y);

        for (int i = 0; i <= areaa[1] - areaa[0]; i++)
        {
            for (int j = 0; j < y; j++)
            {
                int cor = ((float)output[(i * y) + j] / (float)maxIter) * 255 ;
                fprintf(a, "%d %d %d\n", cor, cor / 2, cor / 3);
            }
        }

        for (int k = 1; k < worldSize; k++)
        {
            printf("HOST 0 RECEBENDO DE HOST %d\n", k);
            MPI_Recv((void *)output, (y * (areaa[1] - areaa[0] + 1)), MPI_INT, k, 0, MPI_COMM_WORLD, &st);
            printf("HOST 0 RESPOSTA RECEBIDA DE HOST %d\n", k);

            for (int i = 0; i <= areaa[1] - areaa[0]; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    int cor = ((float)output[(i * y) + j] / (float)maxIter) * 255 ;
                    fprintf(a, "%d %d %d\n", cor, cor / 2, cor / 3);
                }
            }
        }
        fclose(a);
    }
    else
    {
        MPI_Send((void *)output, (y * (areaa[1] - areaa[0] + 1)), MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Host %d enviado para host 0\n", myRank);
    }

    MPI_Finalize();
    printf("Host %d finalizou\n", myRank);
    return 0;
}