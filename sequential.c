#define _POSIX_C_SOURCE 199309L
//gcc -O3 -std=c11 sequential.c -lm -o sequential
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define N 1094716
#define D 6
#define K 6

#define MAX_ITER 100
#define TOL 1e-4f

// Get elapsed wall-clock time in seconds
double get_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {

    // =========================================
    // 1. MEMORY ALLOCATION
    // =========================================

    float *data = malloc((size_t)N * D * sizeof(float));
    int *labels = malloc((size_t)N * sizeof(int));

    float centroids[K][D];
    float newCentroids[K][D];
    float sums[K][D];

    int counts[K];

    if (data == NULL || labels == NULL) {
        printf("Memory allocation failed!\n");
        free(data);
        free(labels);
        return 1;
    }

    // =========================================
    // 2. LOAD DATASET
    // =========================================

    FILE *fp = fopen("kmeans_input.bin", "rb");

    if (fp == NULL) {
        printf("Cannot open dataset file!\n");
        free(data);
        free(labels);
        return 1;
    }

    size_t totalElements = (size_t)N * D;

    if (fread(data, sizeof(float), totalElements, fp)
        != totalElements) {

        printf("Error reading dataset!\n");

        fclose(fp);
        free(data);
        free(labels);

        return 1;
    }

    fclose(fp);

    // =========================================
    // 3. LOAD INITIAL CENTROIDS
    // =========================================

    fp = fopen("initial_centroids.bin", "rb");

    if (fp == NULL) {
        printf("Cannot open centroid file!\n");
        free(data);
        free(labels);
        return 1;
    }

    if (fread(centroids, sizeof(float), K * D, fp)
        != K * D) {

        printf("Error reading centroids!\n");

        fclose(fp);
        free(data);
        free(labels);

        return 1;
    }

    fclose(fp);

    printf("Dataset loaded: %d records x %d features\n", N, D);
    printf("Initial centroids loaded.\n");

    // =========================================
    // 4. K-MEANS ALGORITHM
    // =========================================

    int iterations = 0;

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // Reset cluster sums and counts
        for (int c = 0; c < K; c++) {

            counts[c] = 0;

            for (int j = 0; j < D; j++) {
                sums[c][j] = 0.0f;
            }
        }

        // -------------------------------------
        // STEP 1: ASSIGN POINTS TO CLUSTERS
        // -------------------------------------

        for (int i = 0; i < N; i++) {

            int nearest = 0;
            float minDist = INFINITY;

            // Compare point with every centroid
            for (int c = 0; c < K; c++) {

                float dist = 0.0f;

                for (int j = 0; j < D; j++) {

                    float diff =
                        data[(size_t)i * D + j] - centroids[c][j];

                    dist += diff * diff;
                }

                if (dist < minDist) {
                    minDist = dist;
                    nearest = c;
                }
            }

            // Assign point to nearest cluster
            labels[i] = nearest;

            // Update cluster count
            counts[nearest]++;

            // Accumulate feature sums
            for (int j = 0; j < D; j++) {
                sums[nearest][j] +=
                    data[(size_t)i * D + j];
            }
        }

        // -------------------------------------
        // STEP 2: UPDATE CENTROIDS
        // -------------------------------------

        float maxMovement = 0.0f;

        for (int c = 0; c < K; c++) {

            // Keep previous centroid if cluster is empty
            if (counts[c] == 0) {

                for (int j = 0; j < D; j++) {
                    newCentroids[c][j] = centroids[c][j];
                }

                continue;
            }

            float movementSquared = 0.0f;

            for (int j = 0; j < D; j++) {

                // Calculate mean of each feature
                newCentroids[c][j] =
                    sums[c][j] / counts[c];

                // Calculate centroid movement
                float diff =
                    newCentroids[c][j] - centroids[c][j];

                movementSquared += diff * diff;
            }

            float movement = sqrtf(movementSquared);

            if (movement > maxMovement) {
                maxMovement = movement;
            }
        }

        // -------------------------------------
        // STEP 3: UPDATE ALL CENTROIDS
        // -------------------------------------

        for (int c = 0; c < K; c++) {

            for (int j = 0; j < D; j++) {
                centroids[c][j] = newCentroids[c][j];
            }
        }

        iterations = iter + 1;

        // -------------------------------------
        // STEP 4: CHECK CONVERGENCE
        // -------------------------------------

        if (maxMovement < TOL) {
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    // =========================================
    // 5. END OF TIMED ALGORITHM
    // =========================================

    double executionTime = get_time(start, end);

    // =========================================
    // 6. CALCULATE FINAL CLUSTER SIZES
    // =========================================

    int finalCounts[K] = {0};

    for (int i = 0; i < N; i++) {
        finalCounts[labels[i]]++;
    }

    // =========================================
    // 7. DISPLAY RESULTS
    // =========================================

    printf("\n========== SEQUENTIAL K-MEANS ==========\n");

    printf("Records          : %d\n", N);
    printf("Features         : %d\n", D);
    printf("Clusters (K)     : %d\n", K);
    printf("Iterations       : %d\n", iterations);
    printf("Tolerance        : %.6f\n", TOL);

    printf("Execution Time   : %.6f seconds\n",
           executionTime);

    printf("\nCluster sizes:\n");

    for (int c = 0; c < K; c++) {

        printf("Cluster %d: %d\n",
               c, finalCounts[c]);
    }

    // =========================================
    // 8. FREE MEMORY
    // =========================================

    free(data);
    free(labels);

    return 0;
}
