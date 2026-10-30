/* ============================================================
 *  Matrix Operations in C (2D arrays)
 *  Compile: gcc matrix.c -o matrix -lm
 *  Features: addition, subtraction, multiplication,
 *            scalar multiply, transpose, trace,
 *            determinant, cofactor, adjoint, inverse
 * ============================================================ */
#include <stdio.h>
#include <math.h>

#define MAX 10          /* max rows/cols supported            */
#define EPS 1e-9        /* tolerance for "zero" comparisons   */

/* ------------------------------------------------------------
 *  Basic I/O helpers
 * ---------------------------------------------------------- */
void readMatrix(double a[MAX][MAX], int r, int c, const char *name)
{
    printf("Enter elements of matrix %s (%d x %d):\n", name, r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            scanf("%lf", &a[i][j]);
}

void printMatrix(double a[MAX][MAX], int r, int c)
{
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            double v = (fabs(a[i][j]) < EPS) ? 0.0 : a[i][j];
            printf("%10.4f ", v);
        }
        printf("\n");
    }
}

void zeroMatrix(double a[MAX][MAX], int r, int c)
{
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            a[i][j] = 0.0;
}

/* ------------------------------------------------------------
 *  Addition
 * ---------------------------------------------------------- */
int addMatrix(double a[MAX][MAX], double b[MAX][MAX], int r, int c,
              double res[MAX][MAX])
{
    if (r <= 0 || c <= 0) return 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            res[i][j] = a[i][j] + b[i][j];
    return 1;
}

/* ------------------------------------------------------------
 *  Subtraction
 * ---------------------------------------------------------- */
int subMatrix(double a[MAX][MAX], double b[MAX][MAX], int r, int c,
              double res[MAX][MAX])
{
    if (r <= 0 || c <= 0) return 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            res[i][j] = a[i][j] - b[i][j];
    return 1;
}

/* ------------------------------------------------------------
 *  Multiplication  (r1 x c1) * (r2 x c2), needs c1 == r2
 * ---------------------------------------------------------- */
int mulMatrix(double a[MAX][MAX], double b[MAX][MAX],
              int r1, int c1, int r2, int c2, double res[MAX][MAX])
{
    if (c1 != r2) return 0;                     /* incompatible */

    zeroMatrix(res, r1, c2);
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c2; j++) {
            double sum = 0.0;
            for (int k = 0; k < c1; k++)
                sum += a[i][k] * b[k][j];
            res[i][j] = sum;
        }
    return 1;
}

/* ------------------------------------------------------------
 *  Scalar multiplication
 * ---------------------------------------------------------- */
void scalarMul(double a[MAX][MAX], int r, int c, double s,
               double res[MAX][MAX])
{
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            res[i][j] = a[i][j] * s;
}

/* ------------------------------------------------------------
 *  Transpose
 * ---------------------------------------------------------- */
void transpose(double a[MAX][MAX], int r, int c, double res[MAX][MAX])
{
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            res[j][i] = a[i][j];
}

/* ------------------------------------------------------------
 *  Trace (sum of main diagonal) — square only
 * ---------------------------------------------------------- */
double trace(double a[MAX][MAX], int n)
{
    double t = 0.0;
    for (int i = 0; i < n; i++)
        t += a[i][i];
    return t;
}

/* ------------------------------------------------------------
 *  Determinant — Gaussian elimination with partial pivoting
 *  O(n^3), numerically stable.
 * ---------------------------------------------------------- */
double determinant(double a[MAX][MAX], int n)
{
    if (n == 0) return 1.0;          /* det of empty matrix */
    if (n == 1) return a[0][0];

    double m[MAX][MAX];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = a[i][j];

    double det = 1.0;

    for (int col = 0; col < n; col++) {
        /* find pivot row */
        int piv = col;
        for (int i = col + 1; i < n; i++)
            if (fabs(m[i][col]) > fabs(m[piv][col]))
                piv = i;

        if (fabs(m[piv][col]) < EPS) return 0.0;   /* singular */

        if (piv != col) {                          /* swap rows */
            for (int j = 0; j < n; j++) {
                double t = m[col][j];
                m[col][j] = m[piv][j];
                m[piv][j] = t;
            }
            det = -det;
        }

        det *= m[col][col];

        /* eliminate below */
        for (int i = col + 1; i < n; i++) {
            double f = m[i][col] / m[col][col];
            for (int j = col; j < n; j++)
                m[i][j] -= f * m[col][j];
        }
    }
    return det;
}

/* ------------------------------------------------------------
 *  Cofactor matrix  C[i][j] = (-1)^(i+j) * minor(i,j)
 * ---------------------------------------------------------- */
void cofactorMatrix(double a[MAX][MAX], int n, double cof[MAX][MAX])
{
    if (n == 1) { cof[0][0] = 1.0; return; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double minor[MAX][MAX];
            int mi = 0;

            for (int r = 0; r < n; r++) {
                if (r == i) continue;
                int mj = 0;
                for (int c = 0; c < n; c++) {
                    if (c == j) continue;
                    minor[mi][mj++] = a[r][c];
                }
                mi++;
            }
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            cof[i][j] = sign * determinant(minor, n - 1);
        }
    }
}

/* ------------------------------------------------------------
 *  Adjoint = transpose of cofactor matrix
 * ---------------------------------------------------------- */
void adjointMatrix(double a[MAX][MAX], int n, double adj[MAX][MAX])
{
    double cof[MAX][MAX];
    cofactorMatrix(a, n, cof);
    transpose(cof, n, n, adj);
}

/* ------------------------------------------------------------
 *  Inverse — Gauss-Jordan elimination on [A | I]
 *  Returns 1 on success, 0 if the matrix is singular.
 * ---------------------------------------------------------- */
int inverse(double a[MAX][MAX], int n, double inv[MAX][MAX])
{
    double aug[MAX][2 * MAX];

    /* build augmented matrix */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j]     = a[i][j];
            aug[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < n; col++) {
        /* partial pivoting */
        int piv = col;
        for (int i = col + 1; i < n; i++)
            if (fabs(aug[i][col]) > fabs(aug[piv][col]))
                piv = i;

        if (fabs(aug[piv][col]) < EPS) return 0;   /* singular */

        if (piv != col)
            for (int j = 0; j < 2 * n; j++) {
                double t = aug[col][j];
                aug[col][j] = aug[piv][j];
                aug[piv][j] = t;
            }

        /* normalize pivot row */
        double pv = aug[col][col];
        for (int j = 0; j < 2 * n; j++)
            aug[col][j] /= pv;

        /* eliminate the column in every other row */
        for (int i = 0; i < n; i++) {
            if (i == col) continue;
            double f = aug[i][col];
            for (int j = 0; j < 2 * n; j++)
                aug[i][j] -= f * aug[col][j];
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv[i][j] = aug[i][j + n];

    return 1;
}

/* ============================================================
 *  Menu-driven main
 * ============================================================ */
int main(void)
{
    double A[MAX][MAX], B[MAX][MAX], R[MAX][MAX];
    int r1, c1, r2, c2, choice;

    do {
        printf("\n========== MATRIX MENU ==========\n");
        printf(" 1. Addition\n");
        printf(" 2. Subtraction\n");
        printf(" 3. Multiplication\n");
        printf(" 4. Scalar multiplication\n");
        printf(" 5. Transpose\n");
        printf(" 6. Trace\n");
        printf(" 7. Determinant\n");
        printf(" 8. Cofactor matrix\n");
        printf(" 9. Adjoint matrix\n");
        printf("10. Inverse\n");
        printf(" 0. Exit\n");
        printf("Enter your choice: ");
        if (scanf("%d", &choice) != 1) break;

        switch (choice) {

        /* ---------- 1 & 2 : add / subtract ---------- */
        case 1: case 2:
            printf("Rows and columns: ");
            scanf("%d %d", &r1, &c1);
            readMatrix(A, r1, c1, "A");
            readMatrix(B, r1, c1, "B");

            if (choice == 1) addMatrix(A, B, r1, c1, R);
            else             subMatrix(A, B, r1, c1, R);

            printf("\nResult (%s):\n", choice == 1 ? "A + B" : "A - B");
            printMatrix(R, r1, c1);
            break;

        /* ---------- 3 : multiplication ---------- */
        case 3:
            printf("A rows cols: "); scanf("%d %d", &r1, &c1);
            printf("B rows cols: "); scanf("%d %d", &r2, &c2);
            readMatrix(A, r1, c1, "A");
            readMatrix(B, r2, c2, "B");

            if (!mulMatrix(A, B, r1, c1, r2, c2, R)) {
                printf("Error: columns of A must equal rows of B.\n");
            } else {
                printf("\nA * B:\n");
                printMatrix(R, r1, c2);
            }
            break;

        /* ---------- 4 : scalar multiply ---------- */
        case 4: {
            double s;
            printf("Rows and columns: ");
            scanf("%d %d", &r1, &c1);
            readMatrix(A, r1, c1, "A");
            printf("Scalar: ");
            scanf("%lf", &s);
            scalarMul(A, r1, c1, s, R);
            printf("\n%g * A:\n", s);
            printMatrix(R, r1, c1);
            break;
        }

        /* ---------- 5 : transpose ---------- */
        case 5:
            printf("Rows and columns: ");
            scanf("%d %d", &r1, &c1);
            readMatrix(A, r1, c1, "A");
            transpose(A, r1, c1, R);
            printf("\nTranspose:\n");
            printMatrix(R, c1, r1);
            break;

        /* ---------- 6 : trace ---------- */
        case 6:
            printf("Order n (square matrix): ");
            scanf("%d", &r1);
            readMatrix(A, r1, r1, "A");
            printf("\nTrace = %.4f\n", trace(A, r1));
            break;

        /* ---------- 7 : determinant ---------- */
        case 7:
            printf("Order n (square matrix): ");
            scanf("%d", &r1);
            readMatrix(A, r1, r1, "A");
            printf("\nDeterminant = %.4f\n", determinant(A, r1));
            break;

        /* ---------- 8 : cofactor ---------- */
        case 8:
            printf("Order n (square matrix): ");
            scanf("%d", &r1);
            readMatrix(A, r1, r1, "A");
            cofactorMatrix(A, r1, R);
            printf("\nCofactor matrix:\n");
            printMatrix(R, r1, r1);
            break;

        /* ---------- 9 : adjoint ---------- */
        case 9:
            printf("Order n (square matrix): ");
            scanf("%d", &r1);
            readMatrix(A, r1, r1, "A");
            adjointMatrix(A, r1, R);
            printf("\nAdjoint matrix:\n");
            printMatrix(R, r1, r1);
            break;

        /* ---------- 10 : inverse ---------- */
        case 10:
            printf("Order n (square matrix): ");
            scanf("%d", &r1);
            readMatrix(A, r1, r1, "A");
            if (!inverse(A, r1, R))
                printf("\nMatrix is singular — no inverse exists.\n");
            else {
                printf("\nInverse:\n");
                printMatrix(R, r1, r1);
            }
            break;

        case 0:
            printf("Bye.\n");
            break;

        default:
            printf("Invalid choice.\n");
        }

    } while (choice != 0);

    return 0;
}