
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include <mpi.h>
#include <omp.h>

#define N 1094716
#define D 6
#define K 6
#define MAX_ITER 100
#define TOL 0.0001f

#define DATA_FILE "kmeans_input.bin"
#define CENTROID_FILE "initial_centroids.bin"


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------------------------------
    // 1. Distribute records among MPI processes
    // --------------------------------------------------

    int base = N / size;
    int remainder = N % size;

    int localN = base + (rank < remainder ? 1 : 0);

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            int rows = base + (i < remainder ? 1 : 0);

            sendcounts[i] = rows * D;
            displs[i] = offset;

            offset += sendcounts[i];
        }
    }

    // --------------------------------------------------
    // 2. Load dataset and initial centroids on root
    // --------------------------------------------------

    float *globalData = NULL;

    float centroids[K][D];

    if (rank == 0)
    {
        globalData = malloc((size_t)N * D * sizeof(float));

        if (globalData == NULL)
        {
            fprintf(stderr, "Dataset allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        FILE *fp = fopen(DATA_FILE, "rb");

        if (fp == NULL)
        {
            fprintf(stderr, "Cannot open %s\n", DATA_FILE);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t readCount = fread(
            globalData,
            sizeof(float),
            (size_t)N * D,
            fp
        );

        fclose(fp);

        if (readCount != (size_t)N * D)
        {
            fprintf(stderr, "Dataset read error.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fp = fopen(CENTROID_FILE, "rb");

        if (fp == NULL)
        {
            fprintf(stderr, "Cannot open %s\n", CENTROID_FILE);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        readCount = fread(
            centroids,
            sizeof(float),
            K * D,
            fp
        );

        fclose(fp);

        if (readCount != K * D)
        {
            fprintf(stderr, "Centroid read error.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Dataset loaded: %d records x %d features\n", N, D);
        printf("Initial centroids loaded.\n");
    }

    // --------------------------------------------------
    // 3. Scatter dataset
    // --------------------------------------------------

    float *localData = malloc((size_t)localN * D * sizeof(float));

    int *labels = malloc((size_t)localN * sizeof(int));

    if (localData == NULL || labels == NULL)
    {
        fprintf(stderr, "Rank %d: local allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatterv(
        globalData,
        sendcounts,
        displs,
        MPI_FLOAT,
        localData,
        localN * D,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        free(globalData);
        free(sendcounts);
        free(displs);
    }

    // All processes must use identical initial centroids
    MPI_Bcast(
        centroids,
        K * D,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    // --------------------------------------------------
    // 4. Allocate OpenMP thread-local accumulators
    // --------------------------------------------------

    int maxThreads = omp_get_max_threads();

    float *threadSums = calloc(
        (size_t)maxThreads * K * D,
        sizeof(float)
    );

    int *threadCounts = calloc(
        (size_t)maxThreads * K,
        sizeof(int)
    );

    float globalSums[K][D];

    int globalCounts[K];

    float newCentroids[K][D];

    if (threadSums == NULL || threadCounts == NULL)
    {
        fprintf(stderr, "Rank %d: thread allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int iterations = 0;

    // --------------------------------------------------
    // 5. Start timing
    // --------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    // --------------------------------------------------
    // 6. K-Means iterations
    // --------------------------------------------------

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        // Reset thread-local accumulators
        memset(
            threadSums,
            0,
            (size_t)maxThreads * K * D * sizeof(float)
        );

        memset(
            threadCounts,
            0,
            (size_t)maxThreads * K * sizeof(int)
        );

        // Each thread processes its own records
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            float *mySums = threadSums + (size_t)tid * K * D;

            int *myCounts = threadCounts + (size_t)tid * K;

            #pragma omp for schedule(static)
            for (int i = 0; i < localN; i++)
            {
                int bestCluster = 0;

                float minDistance = INFINITY;

                // Find nearest centroid
                for (int c = 0; c < K; c++)
                {
                    float distance = 0.0f;

                    for (int j = 0; j < D; j++)
                    {
                        float diff =
                            localData[(size_t)i * D + j]
                            - centroids[c][j];

                        distance += diff * diff;
                    }

                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        bestCluster = c;
                    }
                }

                labels[i] = bestCluster;

                // Thread-local accumulation
                myCounts[bestCluster]++;

                for (int j = 0; j < D; j++)
                {
                    mySums[bestCluster * D + j] +=
                        localData[(size_t)i * D + j];
                }
            }
        }

        // --------------------------------------------------
        // 7. Combine thread-local results within each process
        // --------------------------------------------------

        float localSums[K][D] = {0};

        int localCounts[K] = {0};

        for (int t = 0; t < maxThreads; t++)
        {
            for (int c = 0; c < K; c++)
            {
                localCounts[c] += threadCounts[t * K + c];

                for (int j = 0; j < D; j++)
                {
                    localSums[c][j] +=
                        threadSums[t * K * D + c * D + j];
                }
            }
        }

        // --------------------------------------------------
        // 8. MPI: combine results from ALL processes
        // --------------------------------------------------

        MPI_Allreduce(
            localSums,
            globalSums,
            K * D,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        MPI_Allreduce(
            localCounts,
            globalCounts,
            K,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        // --------------------------------------------------
        // 9. Update centroids
        // --------------------------------------------------

        float maxMovement = 0.0f;

        for (int c = 0; c < K; c++)
        {
            for (int j = 0; j < D; j++)
            {
                if (globalCounts[c] > 0)
                {
                    newCentroids[c][j] =
                        globalSums[c][j] / globalCounts[c];
                }
                else
                {
                    // Empty cluster: retain old centroid
                    newCentroids[c][j] = centroids[c][j];
                }
            }

            // Calculate centroid movement
            float movementSquared = 0.0f;

            for (int j = 0; j < D; j++)
            {
                float diff =
                    newCentroids[c][j] - centroids[c][j];

                movementSquared += diff * diff;
            }

            float movement = sqrtf(movementSquared);

            if (movement > maxMovement)
            {
                maxMovement = movement;
            }
        }

        // Copy updated centroids
        memcpy(
            centroids,
            newCentroids,
            sizeof(centroids)
        );

        iterations = iter + 1;

        if (maxMovement < TOL)
        {
            break;
        }
    }

    // --------------------------------------------------
    // 10. Stop timing
    // --------------------------------------------------

    double endTime = MPI_Wtime();

    double localElapsed = endTime - startTime;

    double executionTime = 0.0;

    // Slowest process determines total execution time
    MPI_Reduce(
        &localElapsed,
        &executionTime,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    // --------------------------------------------------
    // 11. Calculate final cluster sizes
    // (Excluded from execution timing)
    // --------------------------------------------------

    int localClusterSizes[K] = {0};

    int globalClusterSizes[K] = {0};

    for (int i = 0; i < localN; i++)
    {
        localClusterSizes[labels[i]]++;
    }

    MPI_Reduce(
        localClusterSizes,
        globalClusterSizes,
        K,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // --------------------------------------------------
    // 12. Print results
    // --------------------------------------------------

    if (rank == 0)
    {
        printf("\n========== HYBRID MPI + OPENMP K-MEANS ==========\n");

        printf("Records          : %d\n", N);
        printf("Features         : %d\n", D);
        printf("Clusters (K)     : %d\n", K);
        printf("MPI Processes    : %d\n", size);
        printf("OpenMP Threads   : %d\n", maxThreads);
        printf("Total Workers    : %d\n", size * maxThreads);
        printf("Iterations       : %d\n", iterations);
        printf("Tolerance        : %.6f\n", TOL);
        printf("Execution Time   : %.6f seconds\n", executionTime);

        printf("\nCluster sizes:\n");

        for (int c = 0; c < K; c++)
        {
            printf(
                "Cluster %d: %d\n",
                c,
                globalClusterSizes[c]
            );
        }
    }

    // --------------------------------------------------
    // 13. Cleanup
    // --------------------------------------------------

    free(localData);
    free(labels);
    free(threadSums);
    free(threadCounts);

    MPI_Finalize();

    return 0;
}
