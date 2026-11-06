// pico_host: runs the PicoRV32 array's foldsim image (pico_v80.toml) and checks it against the Verilator golden.
//
//   pico_host --load <load.hex> --golden <golden.txt> [--timeout-s S] [--max-s S] [--reset-cycles N]
//
// The kernel is foldsim's (rtl/kernel): its control endpoint takes commands through its AXI4-Lite registers
// (foldsim_regs.h): a CSR write or read of the target (pico_v80's host_req), start (its go), a target reset; its
// status says when the target is busy (pico_v80's running). The host resets the target, writes every line of
// load.hex (each tile's program and data), starts it, waits for it to go idle, reads back the checksum of every
// output token, the trapped tiles, the token count, the cycles the cores ran and each tile's last token, and
// compares them with golden.txt (golden_tb.sv on the unfolded RTL). It reports the wall time from start to idle and
// the target cycles the kernel counted.
//
// Transports (one per build): -DPICO_HOST_VRT, the V80 through SLASH's VRT as Vortex's foldsim runtime reaches it
// (VRT_VBIN_PATH, VRT_DEVICE_BDF or the first V80 on the bus; the system's /lib/x86_64-linux-gnu VRT, never
// /opt/xilinx/slash); -DPICO_HOST_EMUL, the kernel in Verilator in-process (foldsim_emul, as the avedsim target);
// neither, a library (pico_host.h) the unit test drives with a mock transport.
#include "pico_host.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace pico {

std::vector<std::pair<uint32_t, uint32_t>> read_load(const std::string& path) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty())
      continue;
    if (line.size() != 11)
      throw std::runtime_error(path + ": a line is not <addr:3 hex><data:8 hex>: " + line);
    out.emplace_back(uint32_t(std::stoul(line.substr(0, 3), nullptr, 16)),
                     uint32_t(std::stoul(line.substr(3), nullptr, 16)));
  }
  if (out.empty())
    throw std::runtime_error(path + ": no writes");
  return out;
}

std::map<std::string, uint32_t> read_golden(const std::string& path) {
  std::map<std::string, uint32_t> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    out[line.substr(0, eq)] = uint32_t(std::stoul(line.substr(eq + 1), nullptr, 16));
  }
  for (const char* k : kResultNames)
    if (!out.count(k))
      throw std::runtime_error(path + ": no " + k);
  return out;
}

int Host::fail(const std::string& what) {
  if (error.empty())
    error = what;
  fprintf(stderr, "[pico] %s\n", what.c_str());
  return -1;
}

int Host::rd(uint32_t off, uint32_t* v) {
  if (t_.reg_read(off, v) != 0)
    return fail("register read failed");
  return 0;
}

int Host::wr(uint32_t off, uint32_t v) {
  if (t_.reg_write(off, v) != 0)
    return fail("register write failed");
  return 0;
}

// Polls the status until done(status); fails on a memory error or when the target's cycles stop advancing for
// timeout_s (progress, not wall time, is the measure).
template <typename Done>
int Host::poll(Done done, const char* what) {
  uint32_t s = 0, last = 0, now = 0;
  if (rd(FOLDSIM_REG_TCYCLES_LO, &last) != 0)
    return -1;
  auto progress = std::chrono::steady_clock::now();
  const auto start = progress;
  for (uint64_t n = 0;; ++n) {
    if (rd(FOLDSIM_REG_STATUS, &s) != 0)
      return -1;
    if (s & FOLDSIM_STATUS_MEM_ERR)
      return fail("the memory endpoint reports an error (status 0x" + hex(s) + ")");
    if (done(s))
      return 0;
    if ((n & 0x3f) == 0x3f) {
      if (rd(FOLDSIM_REG_TCYCLES_LO, &now) != 0)
        return -1;
      if (now != last) {
        last = now;
        progress = std::chrono::steady_clock::now();
      } else if (std::chrono::duration<double>(std::chrono::steady_clock::now() - progress).count() > timeout_s) {
        return fail(std::string("the target has not ") + what + " and has not advanced in " +
                    std::to_string(timeout_s) + " s (status 0x" + hex(s) + ")");
      }
      // A target that advances but never finishes (a core that never traps) is stopped at max_s of wall time.
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > max_s)
        return fail(std::string("the target has not ") + what + " in " + std::to_string(max_s) +
                    " s of wall time (status 0x" + hex(s) + ")");
      if (t_.runs_on_its_own())
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
}

int Host::command(uint32_t op, uint32_t addr, uint32_t data) {
  if (poll([](uint32_t s) { return !(s & FOLDSIM_STATUS_FULL); }, "taken a command") != 0)
    return -1;
  if (wr(FOLDSIM_REG_CMD_ADDR, addr) != 0 || wr(FOLDSIM_REG_CMD_DATA, data) != 0 || wr(FOLDSIM_REG_CMD, op) != 0)
    return -1;
  ++commands;
  return 0;
}

int Host::wait_idle(const char* what) {
  return poll([](uint32_t s) { return !(s & FOLDSIM_STATUS_BUSY); }, what);
}

int Host::csr_read(uint32_t addr, uint32_t* v) {
  if (command(FOLDSIM_CMD_DCR_READ, addr, 0) != 0 ||
      poll([](uint32_t s) { return (s & FOLDSIM_STATUS_RSP) != 0; }, "answered a read") != 0)
    return -1;
  return rd(FOLDSIM_REG_DCR_RSP, v);
}

uint64_t Host::counter(uint32_t lo) {
  uint32_t h = 0, l = 0, h2 = 0;
  rd(lo + 4, &h);
  rd(lo, &l);
  rd(lo + 4, &h2);
  if (h2 != h)
    rd(lo, &l);
  return (uint64_t(h2) << 32) | l;
}

int Host::open() {
  uint32_t id = 0, version = 0, s = 0;
  if (rd(FOLDSIM_REG_ID, &id) != 0 || id != FOLDSIM_ID)
    return fail("the kernel is not a foldsim simulator (id 0x" + hex(id) + ")");
  if (rd(FOLDSIM_REG_VERSION, &version) != 0 || version != FOLDSIM_VERSION)
    return fail("the kernel's register map is version " + std::to_string(version) + ", this host's " +
                std::to_string(FOLDSIM_VERSION));
  // The memory port is idle (pico_v80 has no memory outside its tiles); its base is set as for any image.
  if (wr(FOLDSIM_REG_MEM_BASE_LO, uint32_t(kMemBase)) != 0 || wr(FOLDSIM_REG_MEM_BASE_HI, uint32_t(kMemBase >> 32)) != 0 ||
      wr(FOLDSIM_REG_CTRL, 1) != 0)
    return -1;
  // A target left by an earlier run is reset (refresh off).
  if (command(FOLDSIM_CMD_RESET, 0, reset_cycles) != 0 || wait_idle("finished its reset") != 0 ||
      rd(FOLDSIM_REG_STATUS, &s) != 0)
    return -1;
  return 0;
}

int Host::run(const std::vector<std::pair<uint32_t, uint32_t>>& load, std::map<std::string, uint32_t>* got) {
  for (auto& [a, d] : load)
    if (command(FOLDSIM_CMD_DCR_WRITE, a, d) != 0)
      return -1;
  if (wait_idle("taken the loads") != 0)
    return -1;
  const uint64_t t0 = counter(FOLDSIM_REG_TCYCLES_LO), h0 = counter(FOLDSIM_REG_HCYCLES_LO);
  auto w0 = std::chrono::steady_clock::now();
  if (command(FOLDSIM_CMD_START, 0, 0) != 0 || wait_idle("finished its run") != 0)
    return -1;
  wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
  target_cycles = counter(FOLDSIM_REG_TCYCLES_LO) - t0;
  host_cycles = counter(FOLDSIM_REG_HCYCLES_LO) - h0;
  for (uint32_t a = 0; a < sizeof(kResultNames) / sizeof(kResultNames[0]); ++a) {
    uint32_t v = 0;
    if (csr_read(a, &v) != 0)
      return -1;
    (*got)[kResultNames[a]] = v;
  }
  return 0;
}

int Host::check(const std::map<std::string, uint32_t>& got, const std::map<std::string, uint32_t>& golden) {
  int bad = 0;
  for (const char* k : kResultNames) {
    const bool ok = got.at(k) == golden.at(k);
    printf("[pico] %-8s 0x%08x %s 0x%08x\n", k, got.at(k), ok ? "==" : "!=", golden.at(k));
    bad += !ok;
  }
  return bad;
}

std::string hex(uint32_t v) {
  char b[16];
  snprintf(b, sizeof(b), "%x", v);
  return b;
}

int main_with(Transport& t, int argc, char** argv) {
  std::string load_f, golden_f;
  Host host(t);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(a + " takes a value");
      return argv[++i];
    };
    if (a == "--load")
      load_f = next();
    else if (a == "--golden")
      golden_f = next();
    else if (a == "--timeout-s")
      host.timeout_s = std::stod(next());
    else if (a == "--max-s")
      host.max_s = std::stod(next());
    else if (a == "--reset-cycles")
      host.reset_cycles = uint32_t(std::stoul(next()));
    else
      throw std::runtime_error("unknown argument " + a);
  }
  if (load_f.empty() || golden_f.empty())
    throw std::runtime_error("usage: pico_host --load <load.hex> --golden <golden.txt> [--timeout-s S] [--max-s S]");
  const auto load = read_load(load_f);
  const auto golden = read_golden(golden_f);
  if (t.init() != 0)
    return 2;
  std::map<std::string, uint32_t> got;
  if (host.open() != 0 || host.run(load, &got) != 0)
    return 2;
  const int bad = host.check(got, golden);
  printf("[pico] %zu writes, %lu commands; run: %.3f ms wall, %lu target cycles (cores ran %u), %lu host cycles, "
         "host/target %.2f\n", load.size(), (unsigned long)host.commands, host.wall_ms,
         (unsigned long)host.target_cycles, got.at("cycles"), (unsigned long)host.host_cycles,
         host.target_cycles ? double(host.host_cycles) / double(host.target_cycles) : 0.0);
  printf("[pico] %s\n", bad ? "FAILED" : "PASSED");
  return bad ? 1 : 0;
}

}  // namespace pico

#if defined(PICO_HOST_VRT)
#include <v80_pcie.h>
#include <vrt/device.hpp>
#include <vrt/kernel.hpp>
#include <memory>

// The V80 through VRT, as Vortex's foldsim runtime (sw/runtime/foldsim TARGET=hw) reaches it: the image programmed,
// the kernel clocked at the rate it was timed for (or PICO_CLOCK_HZ, lower), the kernel's registers.
class VrtTransport : public pico::Transport {
public:
  int init() override {
    const char* vbin = getenv("VRT_VBIN_PATH");
    if (vbin == nullptr || !*vbin) {
      fprintf(stderr, "[pico] VRT_VBIN_PATH must name the image's .vbin\n");
      return -1;
    }
    std::string bdf = getenv("VRT_DEVICE_BDF") ? getenv("VRT_DEVICE_BDF") : "";
    if (bdf.empty())
      bdf = vortex::discover_v80_bdf();
    if (bdf.empty()) {
      fprintf(stderr, "[pico] no V80 on the bus; set VRT_DEVICE_BDF\n");
      return -1;
    }
    try {
      device_ = std::make_unique<vrt::Device>(bdf, vbin, /*program=*/true);
      const uint64_t timed = device_->getMaxFrequency();
      uint64_t want = timed;
      if (const char* s = getenv("PICO_CLOCK_HZ")) {
        const uint64_t hz = strtoull(s, nullptr, 0);
        if (hz == 0 || hz > timed) {
          fprintf(stderr, "[pico] PICO_CLOCK_HZ=%s: must be in (0, %lu]\n", s, (unsigned long)timed);
          park();
          return -1;
        }
        want = hz;
      }
      // The clock wizard settles on the nearest rate it reaches, which can be a step above the request.
      const uint64_t limit = want;
      uint64_t hz = 0;
      for (int i = 0; i < 8; ++i) {
        device_->setFrequency(want);
        hz = device_->getFrequency();
        if (hz == 0 || hz <= limit)
          break;
        want -= (hz - want) + limit / 100;
      }
      if (hz == 0 || hz > limit) {
        fprintf(stderr, "[pico] kernel clock reads %lu Hz, the image is timed for %lu Hz\n", (unsigned long)hz,
                (unsigned long)timed);
        park();
        return -1;
      }
      printf("[pico] kernel clock %lu Hz\n", (unsigned long)hz);
      kernel_ = std::make_unique<vrt::Kernel>(*device_, "foldsim_afu_0");
    } catch (const std::exception& e) {
      fprintf(stderr, "[pico] VRT exception: %s\n", e.what());
      park();
      return -1;
    }
    return 0;
  }
  int reg_write(uint32_t off, uint32_t v) override {
    try {
      kernel_->write(off, v);
    } catch (const std::exception& e) {
      fprintf(stderr, "[pico] VRT exception: %s\n", e.what());
      return -1;
    }
    return 0;
  }
  int reg_read(uint32_t off, uint32_t* v) override {
    try {
      *v = kernel_->read(off);
    } catch (const std::exception& e) {
      fprintf(stderr, "[pico] VRT exception: %s\n", e.what());
      return -1;
    }
    return 0;
  }
  bool runs_on_its_own() const override { return true; }

private:
  // An image that does not come up stays loaded: clocked below any image's timing until the next load.
  void park() noexcept {
    try {
      if (device_)
        device_->setFrequency(50000000);
    } catch (...) {
    }
  }
  std::unique_ptr<vrt::Device> device_;
  std::unique_ptr<vrt::Kernel> kernel_;
};
#endif

#if defined(PICO_HOST_EMUL)
#include <foldsim_emul.h>

// The kernel in Verilator, in-process (foldsim_emul: the kernel clock runs while the host touches a register).
class EmulTransport : public pico::Transport {
public:
  int init() override { return sim_.init(); }
  int reg_write(uint32_t off, uint32_t v) override { return sim_.reg_write(off, v); }
  int reg_read(uint32_t off, uint32_t* v) override { return sim_.reg_read(off, v); }
  bool runs_on_its_own() const override { return false; }

private:
  vortex::foldsim_emul sim_;
};
#endif

#if defined(PICO_HOST_VRT) || defined(PICO_HOST_EMUL)
int main(int argc, char** argv) {
  try {
#if defined(PICO_HOST_VRT)
    VrtTransport t;
#else
    EmulTransport t;
#endif
    return pico::main_with(t, argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "[pico] %s\n", e.what());
    return 2;
  }
}
#endif
