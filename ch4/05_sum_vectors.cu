/*--------------------------------------------------------------------*/
/* 05_sum_vectors.cu                                                  */
/* Author: Windsor Nguyen                                             */
/*--------------------------------------------------------------------*/

#include "../book.h"

#define N 10

// Sequentially adding
void add(int *a, int *b, int *c) {
    int tid = 0; // CPU 0 starts at 0
    
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main() {
    int a[N];
    int b[N];
    int c[N];

    // Fill arrays with some numbers
    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    // Add and display the results
    add(a, b, c);
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}