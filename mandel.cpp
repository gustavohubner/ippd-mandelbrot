#include <iostream>
#include <omp.h>

using namespace std;

float norm(float val, float rmin, float rmax, float vmin, float vmax)
{
    return ((vmax - vmin) * ((val - rmin) / (rmax - rmin))) + vmin;
}

#define x 1000
#define y 2000
#define maxIter 10000

int data[x][y];

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

int main()
{

#pragma omp parallel for num_threads(8) collapse(2)
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {

            data[i][j] = (mandel(i, j) == maxIter);
        }
    }

    return 0;
}