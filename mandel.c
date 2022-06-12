#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include <math.h>

// normaliza um valor val dentro de [rmin, rmax] para um valor dentro de [vmin, vmax]
float norm(float val, float rmin, float rmax, float vmin, float vmax)
{
    return ((vmax - vmin) * ((val - rmin) / (rmax - rmin))) + vmin;
}

int x, y, maxIter;

int mandel(int i, int j)
{
	// o ponto de entrada é normalizado para coordenadas do fractal
    float x0 = norm(i, 0, x, -2.00, 0.47);
    float y0 = norm(j, 0, y, -1.12, 1.12);

    float xx = 0.0;
    float yy = 0.0;
    int iteration = 0;

    float xtemp = 0.0;

    while ((iteration < maxIter) && (((xx * xx) + (yy * yy) + y0) < 4))
    {
		// funcao: f(x) = x^2 + c
		
		// c = a + b*i (numero complexo)
		// c^2 = a^2 + 2*(a)*(b*i) + (b*i)^2 
		//     = a^2 + 2abi - b^2
		
		// primeira iteracao
		// f(0) = 0^2 + c = c
		
		// segunda iteracao
		// f(c) = c^2 + c
		// f(c) = (a^2 + 2abi - b^2) + (a + bi)
		// f(c) = (a^2 - b^2 + a) + (2ab + b)*i
		
		// N iteracao: xN é o x na iteracao N (x é um numero complexo, xN = aN + bNi)
		// f(xN) = xN^2 + c
		// f(xN) = (aN^2 + 2aNbNi - bN^2) + (a + bi)
		// f(xN) = (aN^2 - bN^2 + a) + (2aNbN + b)*i
		
		// numero complexo no plano:
		//        x = (aN^2 - bN^2 + a) -> parte inteira
		//        y = (2aNbN + b)       -> parte complexa
        xtemp = xx * xx - yy * yy + x0;
        yy = 2 * xx * yy + y0;
        xx = xtemp;
        iteration = iteration + 1;
    }
    return iteration;
}

int main(int argc, char **argv)
{
    int worldSize, myRank, dest, areaa[2], offset;
    MPI_Status st;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Request req;
    FILE *a;

    if (argc == 4)
    {
        y = atoi(argv[1]);
        x = atoi(argv[2]);
        maxIter = atoi(argv[3]);
        if (myRank == 0) 
		{
			if (y < 1) {
				printf("Numero de linhas invalido, novo y = 1500\n");
				y = 1500;
			}
			if (x < 1) {
				printf("Numero de colunas invalido, novo x = 1500\n");
				x = 1500;
			}
			if (maxIter < 1) {
				printf("Numero de iteracoes invalido, novo maxIter = 1000\n");
				maxIter = 1000;
			}
			printf("Tamanho %dx%d\n%d Iterações\n", x, y, maxIter);
		} 
		else 
		{
			if (y < 1) 
				y = 1500;
			if (x < 1) 
				x = 1500;
			if (maxIter < 1) 
				maxIter = 1000;
		}
    }
    else
    {
        if (myRank == 0)
            printf("Usar: mpirun %s [Numero de linhas] [Numero de colunas] [Numero de iteracoes]\n", argv[0]);
        exit(1);
    }

    if (worldSize != 1)
    {

        if (myRank == 0)
        {
            int end = x - (((int)(x / worldSize)) * (worldSize - 1));
            offset = end - ((int)(x / worldSize));
            for (int i = 1; i < worldSize; i++)
            {
                areaa[0] = (x / worldSize) * i + offset;
                areaa[1] = ((x / worldSize) * (i + 1)) - 1 + offset;
                MPI_Send((void *)areaa, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
                printf("Host 0 - Enviado dimensões para Host %d\n", i);
            }
            areaa[0] = 0;
            areaa[1] = ((x / worldSize) * 1) - 1 + offset;

            printf("Host 0 - Usando limites [0, %d]\n", areaa[1] - areaa[0]);
        }
        else
        {
            MPI_Recv(areaa, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            printf("Host %d - Recebido limites [%d, %d] de Host 0\n", myRank, areaa[0], areaa[1]);
        }
		
		int *output = malloc(sizeof(int) * (y * (areaa[1] - areaa[0] + 1)));

#pragma omp parallel for num_threads(8) collapse(2) schedule(dynamic, 1024)
		for (int i = 0; i <= areaa[1] - areaa[0]; i++)
		{
			for (int j = 0; j < y; j++)
			{
				if (i == 0 && j == 0)
					printf("Host %d - OpenMP Threads: %d\n", myRank, omp_get_num_threads());
				output[(i * y) + j] = mandel(areaa[0] + i, j);
			}
		}

		if (myRank == 0)
		{
			a = fopen("saida.ppm", "w");
			printf("Host 0 - Resultado obtido \n");
			(void)fprintf(a, "P6\n%d %d\n255\n", y, x);

			for (int i = 0; i <= areaa[1] - areaa[0]; i++)
			{
				for (int j = 0; j < y; j++)
				{
					int cor = (int)(((float)output[(i * y) + j] / (float)maxIter) * 255);

					unsigned char cores[3] = {(int)(cor) % 255, (int)((cor % 254) / 1.5), 0};
					(void)fwrite(cores, 1, 3, a);
				}
			}

			for (int k = 1; k < worldSize; k++)
			{
				MPI_Recv((void *)output, (y * (areaa[1] - areaa[0] + 1 - offset)), MPI_INT, k, 0, MPI_COMM_WORLD, &st);
				printf("Host 0 - Resultado recebido de Host %d\n", k);

				for (int i = 0; i <= areaa[1] - areaa[0] - offset; i++)
				{
					for (int j = 0; j < y; j++)
					{
						int cor = (int)(((float)output[(i * y) + j] / (float)maxIter) * 255);
						unsigned char cores[3] = {(int)(cor) % 255, (int)((cor % 254) / 1.5), 0};
						(void)fwrite(cores, 1, 3, a);
					}
				}
			}
			(void)fclose(a);
		}
		else
		{
			MPI_Send((void *)output, (y * (areaa[1] - areaa[0] + 1)), MPI_INT, 0, 0, MPI_COMM_WORLD);
			printf("Host %d - Resultado enviado para Host 0\n", myRank);
		}
		free(output);
		printf("Host %d - Finalizado\n", myRank);
	} 
	else 
	{
		printf("Apenas um host detectado, favor usar pelo menos dois\n");
	}
    MPI_Finalize();
    return 0;
}
