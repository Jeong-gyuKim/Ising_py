#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_D 100
#define MAX_E 1000

#define INIT_L 2
#define FINAL_L 10
#define STEP_L 1

#define T0 2.7
#define DT_SCAN 1.0
#define DT_STEP 0.5

#define N 1000

#define UP_RATE 0.5

typedef struct {
    double energy;
    double *magnetizations;
    int count;
} Histogram;

double energy[MAX_E], hist_E[MAX_E], hist_M[MAX_E], hist_M2[MAX_E], hist_M4[MAX_E];

// Function prototypes
int cmp(const void *a, const void *b);
void initialize_lattice(int L, int** lattice, int* magnetization, double* energy);
double compute_total_energy(int** lattice, int L);
double calculate_energy_change(int** lattice, int x, int y, int L);
void wolff_algorithm(int L, double T, double* energies, double* magnetizations);
int compute_histogram(double* energies, double* magnetizations);
void compute_statistics(const double* temp_range, int temp_size, const int* L_range, int L_size);
void write_csv(const char* filename, double T, int L, double* energy, double* hist_E, double* hist_M, double* hist_M2, double* hist_M4, int size);

int main() {
    int temp_size = (int)((2 * DT_SCAN) / DT_STEP) + 1;
    double* T_range = (double*)malloc(temp_size * sizeof(double));
    int t;
    for (t = 0; t < temp_size; t++) {
        T_range[t] = T0 - DT_SCAN + t * DT_STEP;
    }

    int L_size = (FINAL_L - INIT_L) / STEP_L + 1;
    int* L_range = (int*)malloc(L_size * sizeof(int));
    int l;
    for (l = 0; l < L_size; l++) {
        L_range[l] = INIT_L + l * STEP_L;
    }

    compute_statistics(T_range, temp_size, L_range, L_size);

    free(T_range);
    free(L_range);

    return 0;
}

void compute_statistics(const double* temp_range, int temp_size, const int* L_range, int L_size) {
    int D = 0;
    int l, t;
    for (l = 0; l < L_size; l++) {
        int L = L_range[l];
        for (t = 0; t < temp_size; t++) {
            double T = temp_range[t];

            printf("#Writing file data/hist%d.csv\n", D);

            double* energies = (double*)malloc(N * sizeof(double));
            double* magnetizations = (double*)malloc(N * sizeof(double));

            wolff_algorithm(L, T, energies, magnetizations);

            int hist_count = compute_histogram(energies, magnetizations);

            char filename[50];
            sprintf(filename, "data/hist%d.csv", D);
            write_csv(filename, T, L, energy, hist_E, hist_M, hist_M2, hist_M4, hist_count);

            free(energies);
            free(magnetizations);

            printf("#T1 = %.1f  Nsite = %d\n", T, L * L);
            D++;
        }
    }
}

int cmp(const void *a, const void *b) {
    double diff = ((Histogram *)a)->energy - ((Histogram *)b)->energy;
    return (diff > 0) - (diff < 0);
}

int compute_histogram(double* energies, double* magnetizations) {
    int i, j;
    for (i = 0; i < N; i++) {
        magnetizations[i] = fabs(magnetizations[i]);
    }

    Histogram histogram[MAX_E];
    int hist_count = 0;

    for (i = 0; i < N; i++) {
        double e = energies[i];
        double m = magnetizations[i];

        int found = 0;

        for (j = 0; j < hist_count; j++) {
            if (histogram[j].energy == e) {
                histogram[j].magnetizations[histogram[j].count++] = m;
                found = 1;
                break;
            }
        }

        if (!found) {
            histogram[hist_count].energy = e;
            histogram[hist_count].magnetizations = malloc(N * sizeof(double));
            histogram[hist_count].magnetizations[0] = m;
            histogram[hist_count].count = 1;
            hist_count++;
        }
    }

    qsort(histogram, hist_count, sizeof(Histogram), cmp);

    for (i = 0; i < hist_count; i++) {
        double sum = 0.0, sum_sq = 0.0, sum_quad = 0.0;
        int count = histogram[i].count;
        for (j = 0; j < count; j++) {
            double m = histogram[i].magnetizations[j];
            sum += m;
            sum_sq += m * m;
            sum_quad += m * m * m * m;
        }
        energy[i] = histogram[i].energy;
        hist_E[i] = count;
        hist_M[i] = sum / count;
        hist_M2[i] = sum_sq / count;
        hist_M4[i] = sum_quad / count;
    }

    return hist_count;
}

void initialize_lattice(int L, int** lattice, int* magnetization, double* energy) {
    srand(time(NULL));
    *magnetization = 0;

    int i, j;
    for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
            double random_value = (double)rand() / RAND_MAX;
            lattice[i][j] = (random_value < UP_RATE) ? 1 : -1;
            *magnetization += lattice[i][j];
        }
    }
    *energy = compute_total_energy(lattice, L);
}

double compute_total_energy(int** lattice, int L) {
    double total_energy = 0.0;

    int i, j;
    for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
            int right = lattice[i][(j + 1) % L];
            int down = lattice[(i + 1) % L][j];
            total_energy -= lattice[i][j] * (right + down);
        }
    }

    return total_energy;
}

double calculate_energy_change(int** lattice, int x, int y, int L) {
    int left = lattice[x][(y - 1 + L) % L];
    int right = lattice[x][(y + 1) % L];
    int up = lattice[(x - 1 + L) % L][y];
    int down = lattice[(x + 1) % L][y];

    int neighbor_sum = left + right + up + down;
    return 2 * lattice[x][y] * neighbor_sum;
}

void wolff_algorithm(int L, double T, double* energies, double* magnetizations) {
    int** lattice = (int**)malloc(L * sizeof(int*));
    int i;
    for (i = 0; i < L; i++) {
        lattice[i] = (int*)malloc(L * sizeof(int));
    }

    int magnetization;
    double energy;
    initialize_lattice(L, lattice, &magnetization, &energy);

    double p = 1.0 - exp(-2.0 / T);

    int step;
    for (step = 0; step < N; step++) {
        int x = rand() % L;
        int y = rand() % L;

        int s_i = lattice[x][y];
        int** stack = (int**)malloc(L * L * sizeof(int*));
        int stack_size = 0;

        stack[stack_size] = (int*)malloc(2 * sizeof(int));
        stack[stack_size][0] = x;
        stack[stack_size][1] = y;
        stack_size++;

        while (stack_size > 0) {
            int* current = stack[--stack_size];
            int cx = current[0];
            int cy = current[1];
            free(current);

            if (lattice[cx][cy] == s_i) {

                int neighbors[4][2] = {
                    {cx, (cy - 1 + L) % L},
                    {cx, (cy + 1) % L},
                    {(cx - 1 + L) % L, cy},
                    {(cx + 1) % L, cy}
                };

                for (i = 0; i < 4; i++) {
                    int nx = neighbors[i][0];
                    int ny = neighbors[i][1];
                    if (lattice[nx][ny] == s_i && ((double)rand() / RAND_MAX) < p) {
                        
                        energy += calculate_energy_change(lattice, cx, cy, L);
                        lattice[cx][cy] = -s_i;
                        magnetization += -2 * s_i;

                        stack[stack_size] = (int*)malloc(2 * sizeof(int));
                        stack[stack_size][0] = nx;
                        stack[stack_size][1] = ny;
                        stack_size++;
                    }
                }
            }
        }
        
        for (i = 0; i < stack_size; i++) {
            free(stack[i]);
        }
        free(stack);

        energies[step] = energy;
        magnetizations[step] = magnetization;
    }

    for (i = 0; i < L; i++) {
        free(lattice[i]);
    }
    free(lattice);
}

void write_csv(const char* filename, double T, int L, double* energy, double* hist_E, double* hist_M, double* hist_M2, double* hist_M4, int size) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    fprintf(file, "T,%.1f,Nsite,%d\n", T, L * L);
    fprintf(file, "energy,hist_E,hist_M,hist_M2,hist_M4\n");

    int i;
    for (i = 0; i < size; i++) {
        fprintf(file, "%.1f,%e,%e,%e,%e\n", energy[i], hist_E[i], hist_M[i], hist_M2[i], hist_M4[i]);
    }

    fclose(file);
}
