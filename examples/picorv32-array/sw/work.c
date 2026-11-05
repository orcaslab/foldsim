// The tiles' workload: sort the tile's 64 words (loaded at 0x800), checksum them, sieve the primes below
// 2048 (a bit map at 0xa00), pass the checksum around the ring, and combine it with the neighbour's
// (multiply, divide, remainder). Every result is an output token (OUT).
#define OUT  (*(volatile unsigned *)0x10000000)
#define ID   (*(volatile unsigned *)0x10000004)
#define MBOX (*(volatile unsigned *)0x10000008)
#define N 64
int main(void) {
  unsigned *d = (unsigned *)0x800;
  unsigned char *sieve = (unsigned char *)0xa00;
  unsigned id = ID, s = 0, primes = 0, r;
  for (int i = 1; i < N; i++) {
    unsigned v = d[i];
    int j = i - 1;
    while (j >= 0 && d[j] > v) { d[j + 1] = d[j]; j--; }
    d[j + 1] = v;
  }
  for (int i = 0; i < N; i++) s = s * 31 + d[i];
  OUT = d[0];
  OUT = d[N - 1];
  OUT = s;
  for (int i = 0; i < 256; i++) sieve[i] = 0;
  for (unsigned p = 2; p < 2048; p++) {
    if (sieve[p >> 3] & (1u << (p & 7))) continue;
    primes++;
    for (unsigned q = p * p; q < 2048; q += p) sieve[q >> 3] |= 1u << (q & 7);
  }
  OUT = primes;
  MBOX = s ^ (id << 24) | 1;
  while ((r = MBOX) == 0) ;
  OUT = r;
  OUT = r * (id + 3);
  OUT = r / (id + 7);
  OUT = r % 1000;
  return 0;
}
