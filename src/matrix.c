#include "matrix.h"
#include <math.h>

size_t matrixSize(const Matrix *m) {
  Vec4 dim = m->dim;
  return dim.x * dim.y * dim.z * dim.w;
}

Matrix *newMatrix(Vec4 dim) {
  Matrix *m = malloc(sizeof(Matrix));

  if (m == NULL)
    return NULL;

  m->dim = dim;
  m->data = malloc(sizeof(float) * matrixSize(m));

  if (m->data == NULL) {
    free(m);
    return NULL;
  }

  return m;
}

void deleteMatrix(Matrix *m) {
  free(m->data);
  free(m);
}

size_t idxFromRowCol(const Matrix *m, Vec4 idx) {
  size_t c = m->dim.w;
  size_t b = c * m->dim.z;
  size_t a = b * m->dim.y;
  return a * idx.x + b * idx.y + c * idx.z + idx.w;
}

void setMatrixValue(Matrix *m, Vec4 idx, float val) {
  m->data[idxFromRowCol(m, idx)] = val;
}

float getMatrixValue(const Matrix *m, Vec4 idx) {
  return m->data[idxFromRowCol(m, idx)];
}

void initMatrix(Matrix *m, float vals[]) {
  if (!m || !vals)
    return;

  size_t size = matrixSize(m);

  for (size_t i = 0; i < size; i++) {
    m->data[i] = vals[i];
  }
}

void randomMatrix(Matrix *m, float min, float max) {
  if (!m)
    return;

  float range = max - min;
  float div = range / RAND_MAX;

  size_t size = matrixSize(m);
  for (size_t i = 0; i < size; i++) {
    m->data[i] = min + div * rand();
  }
}

void fillMatrix(Matrix *m, float fill) {
  size_t size = matrixSize(m);

  for (size_t i = 0; i < size; i++) {
    m->data[i] = fill;
  }
}

bool matrixScale(const Matrix *m, float scale, Matrix *out) {
  if (!m || !out) {
    fprintf(stderr, "[ERROR] skipped matrixScale [m=%p] [out=%p]\n", m, out);
    return false;
  }

  if (matrixSize(m) != matrixSize(out)) {
    fprintf(stderr, "[ERROR] skipped matrixScale: mismatched size\n");
    return false;
  }

  size_t size = matrixSize(m);

  for (size_t i = 0; i < size; i++) {
    float res = m->data[i] * scale;
    out->data[i] = (res > MAXFLOAT) ? MAXFLOAT : res;
  }
  return true;
}

bool matrixMul(const Matrix *a, const Matrix *b, Matrix *out) {
  if (!a || !b || !out) {
    fprintf(stderr, "[ERROR] skipped matrixMul: [a=%p] [b=%p] [out=%p]\n", a, b,
            out);
    return false;
  }

  if (a->dim.w != b->dim.x && a->dim.z == b->dim.y) {
    fprintf(stderr, "[ERROR] skipped matrixMul: dimension mismatch\n");
    return false;
  }

  fillMatrix(out, 0.0);

  for (size_t i = 0; i < a->dim.x; i++) {
    for (size_t j = 0; j < a->dim.y; j++) {
      for (size_t k = 0; k < b->dim.z; k++) {
        for (size_t l = 0; l < b->dim.w; l++) {
          Vec4 currIdx = {i, j, k, l};

          for (size_t r = 0; r < a->dim.z; r++) {
            for (size_t c = 0; c < a->dim.w; c++) {
              Vec4 idxA = (Vec4){i, j, r, c};
              Vec4 idxB = (Vec4){c, r, k, l};
              float val = getMatrixValue(a, idxA) * getMatrixValue(b, idxB) +
                          getMatrixValue(out, currIdx);
              setMatrixValue(out, currIdx, val);
            }
          }
        }
      }
    }
  }

  return true;
}

bool matrixAdd(const Matrix *a, const Matrix *b, Matrix *out) {
  if (!a || !b || !out || matrixSize(a) != matrixSize(b) ||
      matrixSize(a) != matrixSize(out)) {
    fprintf(stderr, "[ERROR] skipped matrixAdd\n");
    exit(1);
    return false;
  }

  size_t size = matrixSize(a);

  for (size_t i = 0; i < size; i++) {
    float res = a->data[i] + b->data[i];
    out->data[i] = (res > MAXFLOAT) ? MAXFLOAT : res;
  }

  return true;
}

float matrixSum(const Matrix *in) {
  if (!in)
    return 0.0;

  size_t size = matrixSize(in);
  float total = 0.0;

  for (size_t i = 0; i < size; i++) {
    float f = in->data[i];
    total += f;
  }

  return total;
}

void displayMatrix(const char *title, const Matrix *m) {
  printf("%s:\n", title);
  Vec4 dim = m->dim;

  for (size_t l = 0; l < dim.x; l++) {
    for (size_t i = 0; i < dim.y; i++) {
      for (size_t j = 0; j < dim.z; j++) {
        for (size_t k = 0; k < dim.w; k++) {
          Vec4 idx = {
              .x = l,
              .y = i,
              .z = j,
              .w = k,
          };
          printf("%8.2f ", getMatrixValue(m, idx));
        }
        if (dim.w > 1)
          putchar('\n');
      }
      if (dim.z > 1)
        putchar('\n');
    }
    if (dim.y > 1)
      putchar('\n');
  }
  putchar('\n');
}
