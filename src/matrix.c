#include "matrix.h"

Matrix *newMatrix(size_t rows, size_t cols) {
  Matrix *m = malloc(sizeof(Matrix));

  if (m == NULL)
    return NULL;

  m->rows = rows;
  m->cols = cols;
  m->data = malloc(sizeof(float) * rows * cols);

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

size_t idxFromRowCol(const Matrix *m, size_t r, size_t c) {
  return (r * m->cols) + c;
}

void setMatrixValue(Matrix *m, size_t r, size_t c, float val) {
  m->data[idxFromRowCol(m, r, c)] = val;
}

float getMatrixValue(const Matrix *m, size_t r, size_t c) {
  return m->data[idxFromRowCol(m, r, c)];
}

void initMatrix(Matrix *m, float vals[]) {
  if (!m || !vals)
    return;

  size_t size = m->rows * m->cols;

  for (size_t i = 0; i < size; i++) {
    m->data[i] = vals[i];
  }
}

void randomMatrix(Matrix *m, float min, float max) {
  if (!m)
    return;

  float range = max - min;
  float div = range / RAND_MAX;

  size_t size = m->rows * m->cols;
  for (size_t i = 0; i < size; i++) {
    m->data[i] = min + div * rand();
  }
}

void fillMatrix(Matrix *m, float fill) {
  size_t size = m->rows * m->cols;

  for (size_t i = 0; i < size; i++) {
    m->data[i] = fill;
  }
}

void identityMatrix(Matrix *m, float scale) {
  fillMatrix(m, 0.0);
  size_t l = (m->rows <= m->cols) ? m->rows : m->cols;

  for (size_t i = 0; i < l; i++) {
    setMatrixValue(m, i, i, scale);
  }
}

bool matrixScale(const Matrix *m, float scale, Matrix *out) {
  if (!m || !out) {
    fprintf(stderr, "[FATAL] skipped matrixScale [m=%p] [out=%p]\n", m, out);
    return false;
  }

  if (m->rows != out->rows || m->cols != out->cols) {
    fprintf(stderr, "[FATAL] skipped matrixScale %lu x %lu vs %lu x %lu\n",
            m->rows, m->cols, out->rows, out->cols);
    return false;
  }

  size_t size = m->rows * m->cols;

  for (size_t i = 0; i < size; i++) {
    float res = m->data[i] * scale;
    out->data[i] = (res > MAXFLOAT) ? MAXFLOAT : res;
  }
  return true;
}

bool matrixMul(const Matrix *a, const Matrix *b, Matrix *out) {
  if (!a || !b || !out || a->cols != b->rows || out->rows != a->rows ||
      out->cols != b->cols) {
    fprintf(stderr, "[FATAL] skipped matrixMul\n");
    return false;
  }

  fillMatrix(out, 0.0);

  for (size_t r = 0; r < a->rows; r++) {
    for (size_t c = 0; c < b->cols; c++) {
      for (size_t k = 0; k < a->cols; k++) {
        float val = getMatrixValue(a, r, k) * getMatrixValue(b, k, c) +
                     getMatrixValue(out, r, c);
        setMatrixValue(out, r, c, val);
      }
    }
  }

  return true;
}

bool matrixAdd(const Matrix *a, const Matrix *b, Matrix *out) {
  if (!a || !b || !out || a->rows != b->rows || a->cols != b->cols ||
      a->rows != out->rows || a->cols != out->cols) {
    fprintf(stderr, "[FATAL] skipped matrixAdd\n");
    exit(1);
    return false;
  }

  size_t size = a->rows * a->cols;

  for (size_t i = 0; i < size; i++) {
    float res = a->data[i] + b->data[i];
    out->data[i] = (res > MAXFLOAT) ? MAXFLOAT : res;
  }

  return true;
}

float matrixSum(const Matrix *in) {
  if (!in)
    return 0.0;

  size_t size = in->rows * in->cols;
  float total = 0.0;

  for (size_t i = 0; i < size; i++) {
    float f = in->data[i];
    total += f;
  }

  return total;
}

void displayMatrix(const char *title, const Matrix *m) {
  printf("%s:\n", title);

  for (size_t r = 0; r < m->rows; r++) {
    for (size_t c = 0; c < m->cols; c++) {
      printf("\t%-7.2f ", getMatrixValue(m, r, c));
    }
    putchar('\n');
  }
  putchar('\n');
}
