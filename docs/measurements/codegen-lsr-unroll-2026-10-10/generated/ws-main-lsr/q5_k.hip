
extern "C" {
__global__ void sarek_kern(int* __restrict__ er, int sarek_er_length, int* __restrict__ slots, int sarek_slots_length, int rec_words, int o_d, int o_dmin, int o_scales, int o_qh, int o_qs, float* __restrict__ x, int sarek_x_length, int x_stride, float* __restrict__ y, int sarek_y_length, int rows, int nb) {
  __shared__ float red[256];
  int tid = threadIdx.x;
  int lane = (tid & 31);
  int row = ((blockIdx.x * 8) + (tid / 32));
  int base = (slots[blockIdx.y] * rec_words);
  int xo = (blockIdx.y * x_stride);
  int yo = (blockIdx.y * rows);
  float acc = 0.0f;
  {
    if ((row < rows)) {
      int item = lane;
      while ((item < (nb * 8))) {
        int b = (item / 8);
        int j = (item - (b * 8));
        int blk = ((row * nb) + b);
        int sb = (blk * 3);
        int ja = (j + 4);
        int jb = ((j < 4) ? j : (j - 4));
        int w_j = er[(((base + o_scales) + sb) + (j / 4))];
        int w_ja = er[(((base + o_scales) + sb) + (ja / 4))];
        int w_jb = er[(((base + o_scales) + sb) + (jb / 4))];
        int sh_j = ((j - ((j / 4) * 4)) * 8);
        int sh_ja = ((ja - ((ja / 4) * 4)) * 8);
        int sh_jb = ((jb - ((jb / 4) * 4)) * 8);
        int b_j = (((sh_j == 0) ? w_j : ((w_j >> sh_j) ^ ((w_j >> 31) << ((32 - sh_j) & 31)))) & 255);
        int b_ja = (((sh_ja == 0) ? w_ja : ((w_ja >> sh_ja) ^ ((w_ja >> 31) << ((32 - sh_ja) & 31)))) & 255);
        int b_jb = (((sh_jb == 0) ? w_jb : ((w_jb >> sh_jb) ^ ((w_jb >> 31) << ((32 - sh_jb) & 31)))) & 255);
        int hi_jb = ((6 == 0) ? b_jb : ((b_jb >> 6) ^ ((b_jb >> 31) << ((32 - 6) & 31))));
        int hi_j = ((6 == 0) ? b_j : ((b_j >> 6) ^ ((b_j >> 31) << ((32 - 6) & 31))));
        int sc = 0;
        int m = 0;
        if ((j < 4)) {
          sc = (b_j & 63);
          m = (b_ja & 63);
        } else {
          sc = ((b_ja & 15) | (hi_jb << 4));
          m = (((4 == 0) ? b_ja : ((b_ja >> 4) ^ ((b_ja >> 31) << ((32 - 4) & 31)))) | (hi_j << 4));
        }
        int p = (j / 2);
        int hi = (j - (p * 2));
        int nsh = (hi * 4);
        int xb = (((xo + (b * 256)) + (p * 64)) + (hi * 32));
        int qb = (((base + o_qs) + (blk * 32)) + (p * 8));
        int hb = ((base + o_qh) + (blk * 8));
        float sqx = 0.0f;
        float sx = 0.0f;
        for (int wi = 0; wi <= 7; wi++) {
          int w = er[(qb + wi)];
          int wh = er[(hb + wi)];
          int lo = (((nsh == 0) ? w : ((w >> nsh) ^ ((w >> 31) << ((32 - nsh) & 31)))) & 252645135);
          int hbits = (((j == 0) ? wh : ((wh >> j) ^ ((wh >> 31) << ((32 - j) & 31)))) & 16843009);
          int qw = (lo + (hbits << 4));
          int xi = (xb + (wi * 4));
          float x0 = x[xi];
          float x1 = x[(xi + 1)];
          float x2 = x[(xi + 2)];
          float x3 = x[(xi + 3)];
          float q0 = (float)((qw & 255));
          float q1 = (float)((((8 == 0) ? qw : ((qw >> 8) ^ ((qw >> 31) << ((32 - 8) & 31)))) & 255));
          float q2 = (float)((((16 == 0) ? qw : ((qw >> 16) ^ ((qw >> 31) << ((32 - 16) & 31)))) & 255));
          float q3 = (float)((((24 == 0) ? qw : ((qw >> 24) ^ ((qw >> 31) << ((32 - 24) & 31)))) & 255));
          sqx = ((((sqx + (q0 * x0)) + (q1 * x1)) + (q2 * x2)) + (q3 * x3));
          sx = ((((sx + x0) + x1) + x2) + x3);
        }
        acc = ((acc + ((__int_as_float(er[((base + o_d) + blk)]) * (float)(sc)) * sqx)) - ((__int_as_float(er[((base + o_dmin) + blk)]) * (float)(m)) * sx));
        item = (item + 32);
      }
    }
    red[tid] = acc;
  }
  __syncthreads();
  {
    if ((lane < 16)) {
      red[tid] = (red[tid] + red[(tid + 16)]);
    }
  }
  __syncthreads();
  {
    if ((lane < 8)) {
      red[tid] = (red[tid] + red[(tid + 8)]);
    }
  }
  __syncthreads();
  {
    if ((lane < 4)) {
      red[tid] = (red[tid] + red[(tid + 4)]);
    }
  }
  __syncthreads();
  {
    if ((lane < 2)) {
      red[tid] = (red[tid] + red[(tid + 2)]);
    }
  }
  __syncthreads();
  if (((lane == 0) ? (row < rows) : 0)) {
    y[(yo + row)] = (red[tid] + red[(tid + 1)]);
  }
}
}
