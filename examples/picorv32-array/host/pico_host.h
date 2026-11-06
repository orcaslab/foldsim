// pico_host.h: the PicoRV32 image's host (pico_host.cpp) as a library over a register transport.
#pragma once

#include <foldsim_regs.h>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pico {

// pico_v80's CSR reads, by address: what the host reads back and checks.
constexpr const char* kResultNames[] = {"checksum", "trap", "tokens", "cycles", "last0", "last1", "last2", "last3"};
// Device memory's base (the idle memory port's), as every foldsim image takes it.
constexpr uint64_t kMemBase = 0x4000000000ull;

// The kernel's registers (foldsim_regs.h).
class Transport {
public:
  virtual ~Transport() = default;
  virtual int init() = 0;
  virtual int reg_write(uint32_t off, uint32_t v) = 0;
  virtual int reg_read(uint32_t off, uint32_t* v) = 0;
  // Whether the kernel runs between accesses (the card) or only while the host accesses it (in-process models).
  virtual bool runs_on_its_own() const = 0;
};

class Host {
public:
  explicit Host(Transport& t) : t_(t) {}
  int open();
  int run(const std::vector<std::pair<uint32_t, uint32_t>>& load, std::map<std::string, uint32_t>* got);
  int check(const std::map<std::string, uint32_t>& got, const std::map<std::string, uint32_t>& golden);

  double timeout_s = 30.0;  // no target progress for this long: stopped
  double max_s = 600.0;     // a wait longer than this, progress or not: stopped
  uint32_t reset_cycles = 64;
  double wall_ms = 0.0;
  uint64_t target_cycles = 0, host_cycles = 0, commands = 0;
  std::string error;

private:
  int fail(const std::string& what);
  int rd(uint32_t off, uint32_t* v);
  int wr(uint32_t off, uint32_t v);
  template <typename Done>
  int poll(Done done, const char* what);
  int command(uint32_t op, uint32_t addr, uint32_t data);
  int wait_idle(const char* what);
  int csr_read(uint32_t addr, uint32_t* v);
  uint64_t counter(uint32_t lo);
  Transport& t_;
};

std::vector<std::pair<uint32_t, uint32_t>> read_load(const std::string& path);
std::map<std::string, uint32_t> read_golden(const std::string& path);
std::string hex(uint32_t v);
int main_with(Transport& t, int argc, char** argv);

}  // namespace pico
